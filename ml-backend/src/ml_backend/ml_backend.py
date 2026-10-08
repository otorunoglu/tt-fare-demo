import os
import logging

# Configure logging FIRST, before any imports that might set up their own loggers
# This sets the root logger level which propagates to all child loggers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    force=True,  # Force reconfiguration even if logging was already set up
)

# Set root logger to INFO to catch any loggers that don't inherit properly
logging.getLogger().setLevel(logging.INFO)

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml_backend.registry import load_backend_for_project
from ml_backend.routes.common.health import health_router
from ml_backend.routes.smartstable import smartstable_router


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logger = logging.getLogger("MLBackend")


# ---------------------------------------------------------------------
# Label Studio connection
# ---------------------------------------------------------------------

# Import ls client from dedicated module to avoid circular imports
from ml_backend.ls_client import get_ls_client

# Create local reference for backwards compatibility within this module
ls = get_ls_client()


# ---------------------------------------------------------------------
# FastAPI initialization
# ---------------------------------------------------------------------

app = FastAPI(
    title="SmartStable ML Backend",
    description="Unified ML backend with multi-project support",
    version="1.0.0",
)

# CORS if LS runs on different origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def check_generated_types():
    """
    Check if the generated types in smartstablemodel are up to date with the config.
    """
    try:
        from smartstablemodel.tools.generate_types import check_and_update

        logger.info("Checking smartstablemodel generated types...")
        check_and_update()
    except ImportError:
        logger.warning(
            "Could not import smartstablemodel.tools.generate_types. Skipping type check."
        )
    except Exception as e:
        logger.error(f"Error checking generated types: {e}")


@app.on_event("startup")
async def start_alert_watcher():
    """
    Launch the background watcher that turns incoming edge-device alert clips into
    Label Studio tasks. Fire-and-forget; a reference is kept so it isn't GC'd.
    """
    import asyncio

    from ml_backend.alert_watcher import run_alert_watcher

    app.state.alert_watcher_task = asyncio.create_task(run_alert_watcher())
    logger.info("Alert watcher task scheduled.")


# ---------------------------------------------------------------------
# Mount project-specific routers
# ---------------------------------------------------------------------


# Mount routers
app.include_router(health_router, prefix="", tags=["common"])
app.include_router(smartstable_router, prefix="/api/smartstable", tags=["smartstable"])


# ---------------------------------------------------------------------
# Core prediction endpoint used by Label Studio
# ---------------------------------------------------------------------

from fastapi.responses import Response


class PredictRequest(BaseModel):
    tasks: list  # Label Studio sends tasks as a list


from threading import Lock

# Track tasks currently being processed to prevent duplicate processing
_processing_tasks: set = set()
_processing_tasks_lock = Lock()


@app.post("/predict", status_code=202)
async def predict(payload: PredictRequest, background: BackgroundTasks):
    tasks = payload.tasks
    if not tasks:
        raise HTTPException(400, "No tasks in request")

    task = tasks[0]
    task_id = task.get("id")
    project_id = task.get("project")

    # Log full task for debugging if project is missing
    if project_id is None:
        logger.debug(f"Task payload missing 'project' field. Full task: {task}")
        # Try to get project_id from task data or use default project 1
        project_id = task.get("data", {}).get("project") or 1
        logger.info(f"Using fallback project_id={project_id} for task {task_id}")

    # Check if task is already being processed
    with _processing_tasks_lock:
        if task_id in _processing_tasks:
            logger.debug(f"Task {task_id} already in progress, skipping")
            return Response(status_code=202)
        _processing_tasks.add(task_id)

    logger.info(f"Predict request for task {task_id} in project {project_id}")

    backend = load_backend_for_project(project_id)
    if backend is None:
        # Fall back to default backend
        logger.warning(
            f"No backend for project {project_id}, falling back to SmartStable backend"
        )
        backend = load_backend_for_project(1)

    if backend is None:
        with _processing_tasks_lock:
            _processing_tasks.discard(task_id)
        raise HTTPException(404, f"No backend available")

    # Wrapper to remove task from processing set when done
    def run_and_cleanup():
        try:
            backend.handle_predict_request(task_id, ls)
        finally:
            with _processing_tasks_lock:
                _processing_tasks.discard(task_id)

    background.add_task(run_and_cleanup)

    # Return 202 Accepted with empty body - tells LS "I'm working on it"
    return Response(status_code=202)


# ---------------------------------------------------------------------
# Label Studio setup endpoint
# ---------------------------------------------------------------------


@app.post("/setup")
async def setup(payload: dict):
    """
    LS calls this when attaching the ML backend.
    Returns the model version and available labels.

    Note: Label Studio sometimes sends weird values in the 'project' field
    (like model versions). We try to extract project_id but fall back to
    a default backend if not available.
    """
    logger.debug(f"Setup payload received: {payload}")

    # Handle different payload formats from Label Studio
    # LS may send: {"project": {"id": 1, ...}} or {"project": 1} or {"project": "1"}
    project = payload.get("project")
    project_id = None

    if isinstance(project, dict):
        project_id = project.get("id")
    elif isinstance(project, int):
        project_id = project
    elif isinstance(project, str):
        # Try to parse as int, but handle floats/versions gracefully
        try:
            if "." not in project:
                project_id = int(project)
        except ValueError:
            pass

    # If project_id still not found, try other common payload locations
    if project_id is None:
        project_id = payload.get("project_id")

    # Try to get project-specific backend, or fall back to default (project 1)
    backend = load_backend_for_project(project_id) if project_id else None

    if backend is None:
        # Fall back to default backend (SmartStable, project 1)
        # Note: LS often sends garbage in 'project' field during setup.
        # This is fine - LS will only use labels matching its label config anyway.
        logger.info(
            f"No backend for project {project_id}, using default SmartStable backend"
        )
        backend = load_backend_for_project(1)

    if backend is None:
        raise HTTPException(500, "No backend available")

    return backend.get_setup_response()


# ---------------------------------------------------------------------
# Webhook endpoint for Label Studio training trigger
# ---------------------------------------------------------------------

# Track if training is in progress
_training_in_progress = False
_training_lock = Lock()


@app.post("/webhook")
async def webhook_handler(payload: dict, background: BackgroundTasks):
    """
    Webhook called by Label Studio when 'Start Training' is clicked.
    """
    global _training_in_progress

    action = payload.get("action", "unknown")
    logger.info(f"Webhook received: action={action}")

    if action != "START_TRAINING":
        return {"message": f"Webhook action '{action}' acknowledged", "status": "ok"}

    project_data = payload.get("project", {})
    project_id = project_data.get("id") if isinstance(project_data, dict) else None

    if not project_id:
        raise HTTPException(400, "Missing project id in webhook")

    # Check if training data exists
    total_annotations = project_data.get("total_annotations_number", 0)
    tasks_with_annotations = project_data.get("num_tasks_with_annotations", 0)

    logger.info(
        f"Training request: project={project_id}, annotations={total_annotations}, tasks={tasks_with_annotations}"
    )

    if total_annotations == 0:
        raise HTTPException(400, "No training data (annotations) available")

    # Prevent concurrent training
    with _training_lock:
        if _training_in_progress:
            logger.warning("Training already in progress, skipping")
            return {"message": "Training already in progress", "status": "busy"}
        _training_in_progress = True

    backend = load_backend_for_project(project_id)
    if backend is None:
        with _training_lock:
            _training_in_progress = False
        raise HTTPException(404, f"No backend registered for project {project_id}")

    def run_training_and_cleanup():
        global _training_in_progress
        try:
            backend.run_training(project_id, ls)
        finally:
            with _training_lock:
                _training_in_progress = False

    background.add_task(run_training_and_cleanup)

    return {
        "message": "Training started",
        "status": "training",
        "project_id": project_id,
        "total_annotations": total_annotations,
        "tasks_with_annotations": tasks_with_annotations,
    }


# ---------------------------------------------------------------------
# App entrypoint (for uvicorn)
# ---------------------------------------------------------------------


def create_app():
    return app


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("ML_BACKEND_PORT", "30040"))
    logger.info(f"Starting ML Backend on 0.0.0.0:{port}")

    # Log registered routes
    logger.info("=== REGISTERED ROUTES ===")
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            logger.info(f"  {route.path} -> {route.methods}")  # type: ignore
    logger.info("=== END ROUTES ===")

    uvicorn.run(app, host="0.0.0.0", port=port)
