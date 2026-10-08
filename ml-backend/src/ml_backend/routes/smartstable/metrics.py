# ml_backend/routes/smartstable/metrics.py
"""
SmartStable metrics and evaluation endpoints.

Provides endpoints for:
- Model evaluation on test tasks
- Metrics retrieval
- Performance monitoring
"""

import logging
from typing import Optional, Literal, Any
from datetime import datetime
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ml_backend.smartstable_core.manager_loader import (
    get_multistall_manager,
    get_model_version_string,
)
from ml_backend.smartstable_core.evaluation import (
    evaluate_on_test_tasks,
    extract_ground_truth_labels,
    run_synthetic_evaluation,
    generate_synthetic_scenario,
    evaluate_saved_scenario,
)
from ml_backend.ls_client import get_ls_client

logger = logging.getLogger("MetricsRouter")

metrics_router = APIRouter(prefix="/metrics", tags=["metrics"])


# ─── Request/Response Models ──────────────────────────────────────────────────


class EvaluationRequest(BaseModel):
    """Request to evaluate model on test tasks."""

    project_id: int = Field(..., description="Label Studio project ID")
    iou_threshold: float = Field(
        0.3, ge=0.0, le=1.0, description="IoU threshold for segment matching"
    )
    log_to_mlflow: bool = Field(True, description="Whether to log results to MLflow")


class EvaluationResponse(BaseModel):
    """Response from model evaluation."""

    success: bool
    model_version: str
    model_source: Optional[str] = None
    evaluation_time: Optional[str] = None
    num_test_tasks: int = 0
    num_evaluated: int = 0
    metrics: Optional[dict] = None
    task_results: Optional[list] = None
    message: Optional[str] = None


class SyntheticEvaluationRequest(BaseModel):
    """Request to create and evaluate a synthetic track."""

    run_id: Optional[str] = Field(
        None,
        description="MLflow Run ID to get the test pool from (defaults to currently loaded model's run)",
    )
    project_id: int = Field(..., description="Label Studio project ID to resolve audio")
    duration: float = Field(
        600.0, ge=60.0, le=3600.0, description="Duration of synthetic track in seconds"
    )
    events_per_min: float = Field(
        5.0, ge=1.0, le=30.0, description="Average events per minute"
    )


class FullEvaluationRequest(BaseModel):
    """Request for a full evaluation including standard and synthetic tests."""

    project_id: int = Field(..., description="Label Studio project ID")
    iou_threshold: float = Field(
        0.3, ge=0.0, le=1.0, description="Minimum IoU for segment matching"
    )
    run_id: Optional[str] = Field(
        None, description="Optional MLflow run ID for test pool retrieval"
    )
    include_synthetic: bool = Field(
        True, description="Whether to include synthetic evaluation"
    )
    synthetic_duration: float = Field(
        600.0, ge=60.0, le=3600.0, description="Duration of synthetic track"
    )
    synthetic_events_per_min: float = Field(
        5.0, ge=1.0, le=30.0, description="Average events per minute for synthesis"
    )
    log_to_mlflow: bool = Field(True, description="Whether to log results to MLflow")


class ScenarioBlock(BaseModel):
    type: str = Field(
        ..., description="Block type (e.g. 'eating_window', 'distress_cluster')"
    )
    labels: list[str] = Field(..., description="List of labels to sample from")
    start_hour: float = Field(..., description="Start hour (0-24)")
    end_hour: float = Field(..., description="End hour (0-24)")
    events_per_minute: float = Field(..., description="Density of events in this block")


class ScenarioGenerationRequest(BaseModel):
    name: str = Field(..., description="Name of the scenario for saving to disk")
    project_id: int = Field(
        ..., description="Label Studio project ID to fetch samples from"
    )
    run_id: Optional[str] = Field(
        None, description="MLflow run ID for test pool retrieval"
    )
    total_duration_hours: float = Field(
        ..., description="Total length of the scenario in hours"
    )
    behavior_blocks: list[ScenarioBlock] = Field(
        ..., description="Custom windows for behaviors"
    )
    background_events_per_minute: float = Field(
        1.0, description="Density of scattered 'normal' background noise"
    )


class ScenarioEvaluationRequest(BaseModel):
    scenario_name: str = Field(
        ..., description="Name of the scenario to load and evaluate"
    )
    log_to_mlflow: bool = Field(True, description="Whether to log results to MLflow")


class ShortSegmentDistributionRequest(BaseModel):
    """Request for labeled short-segment distribution analysis."""

    project_id: int = Field(..., description="Label Studio project ID")
    task_scope: Literal["training", "test", "all"] = Field(
        "training",
        description=(
            "Task purpose scope: 'training' (default, matches training pipeline), "
            "'test', or 'all'"
        ),
    )
    shorter_than_seconds: Optional[float] = Field(
        None,
        gt=0.0,
        description=(
            "Optional short-segment threshold override in seconds. If omitted, "
            "the service uses the model segment duration constant."
        ),
    )
    include_current_model_metrics: bool = Field(
        False,
        description=(
            "If true, runs current-model evaluation on test tasks and attaches "
            "per-class precision/recall/f1/support where available."
        ),
    )
    iou_threshold_for_metrics: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="IoU threshold used when include_current_model_metrics=true",
    )
    max_tasks: Optional[int] = Field(
        None,
        ge=1,
        description="Optional cap on number of tasks processed after scope filtering",
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────


@metrics_router.get("/debug")
def debug_metrics():
    """Health check for metrics endpoint."""
    return {"status": "ok", "message": "metrics route reachable"}


@metrics_router.post("/generate-scenario")
def generate_scenario_endpoint(request: ScenarioGenerationRequest):
    """
    Generate a highly configurable synthetic scenario and save it to disk.
    Will not be used for training.
    """
    try:
        ls = get_ls_client()
        logger.info(f"Generating synthetic scenario: {request.name}")

        result = generate_synthetic_scenario(
            request=request,
            ls_client=ls,
        )
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Scenario generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@metrics_router.post("/evaluate-scenario")
def evaluate_scenario_endpoint(request: ScenarioEvaluationRequest):
    """
    Evaluate the current model on a previously generated synthetic scenario.
    """
    try:
        logger.info(f"Evaluating synthetic scenario: {request.scenario_name}")

        result = evaluate_saved_scenario(
            scenario_name=request.scenario_name,
            log_to_mlflow=request.log_to_mlflow,
        )
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Scenario evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@metrics_router.get("/model-info")
def get_model_info_endpoint():
    """Get information about the currently loaded model."""
    from ml_backend.smartstable_core.manager_loader import get_model_info

    model_info = get_model_info()
    model_version = get_model_version_string()

    manager = get_multistall_manager()
    manager_status = "loaded" if manager is not None else "not loaded"

    return {
        "model_version": model_version,
        "model_info": model_info,
        "manager_status": manager_status,
    }


@metrics_router.post("/evaluate-test-tasks", response_model=EvaluationResponse)
def evaluate_test_tasks_endpoint(request: EvaluationRequest):
    """
    Evaluate the current production model on all test tasks.

    This endpoint:
    1. Fetches all tasks marked as 'test' (via task_purpose annotation) from the project
    2. Runs the current model on each test task
    3. Compares predictions to ground truth annotations
    4. Calculates precision, recall, F1 scores (overall and per-label)
    5. Optionally logs results to MLflow

    Test tasks are those with a `task_purpose` Choices annotation set to 'test'.
    Tasks without this annotation or with 'train' are excluded.

    The evaluation uses IoU (Intersection over Union) to match predicted
    segments to ground truth segments before comparing labels.
    """
    try:
        ls = get_ls_client()

        result = evaluate_on_test_tasks(
            project_id=request.project_id,
            ls_client=ls,
            iou_threshold=request.iou_threshold,
            log_to_mlflow=request.log_to_mlflow,
        )

        return EvaluationResponse(**result)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@metrics_router.get("/test-task-count")
def get_test_task_count(
    project_id: int = Query(..., description="Label Studio project ID"),
):
    """
    Get the count of test tasks in a project.

    Useful to check how many test tasks are available before running evaluation.
    """
    from ml_backend.util.label_studio_helper import (
        list_tasks_by_project,
        get_test_tasks,
    )

    try:
        ls = get_ls_client()

        all_tasks = list_tasks_by_project(
            project_id, ls_client=ls, only_annotated=True, exclude_test_tasks=False
        )

        test_tasks = get_test_tasks(all_tasks)
        training_tasks_count = len(all_tasks) - len(test_tasks)

        return {
            "project_id": project_id,
            "total_annotated_tasks": len(all_tasks),
            "test_tasks": len(test_tasks),
            "training_tasks": training_tasks_count,
        }

    except Exception as e:
        logger.error(f"Failed to count tasks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@metrics_router.post("/evaluate-synthetic")
def evaluate_synthetic_track_endpoint(request: SyntheticEvaluationRequest):
    """
    Generate and evaluate a synthetic soundscape using a model's test pool.

    This ensures the model is tested on data it has NEVER seen during training.
    """
    from ml_backend.smartstable_core.evaluation import run_synthetic_evaluation

    try:
        ls = get_ls_client()
        result = run_synthetic_evaluation(
            run_id=request.run_id,
            project_id=request.project_id,
            duration=request.duration,
            events_per_min=request.events_per_min,
            ls_client=ls,
        )
        return result
    except Exception as e:
        logger.error(f"Synthetic evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@metrics_router.post("/evaluate-full")
def evaluate_full_endpoint(request: FullEvaluationRequest):
    """
    Run a full evaluation:
    1. Standard evaluation on Label Studio test tasks
    2. Optional synthetic evaluation using the model's test pool
    3. Upload all artifacts to a single MLflow run
    """
    results = {
        "success": True,
        "model_version": get_model_version_string(),
        "timestamp": datetime.now().isoformat(),
    }

    try:
        ls = get_ls_client()

        # 1. Standard Evaluation
        logger.info(f"Starting standard evaluation for project {request.project_id}...")
        results["standard_evaluation"] = evaluate_on_test_tasks(
            project_id=request.project_id,
            ls_client=ls,
            iou_threshold=request.iou_threshold,
            log_to_mlflow=request.log_to_mlflow,
        )

        # 2. Synthetic Evaluation
        if request.include_synthetic:
            # If standard eval created a run, use it. Otherwise uses request.run_id or None.
            log_target = results["standard_evaluation"].get("mlflow_run_id")

            logger.info("Starting synthetic evaluation...")
            results["synthetic_evaluation"] = run_synthetic_evaluation(
                run_id=request.run_id,
                project_id=request.project_id,
                duration=request.synthetic_duration,
                events_per_min=request.synthetic_events_per_min,
                ls_client=ls,
                log_to_run_id=log_target,
            )

        # 3. Upload real task plots to the model's training run as well
        #    (so they appear alongside the synthetic eval artifacts)
        if request.log_to_mlflow:
            try:
                from mlflow import MlflowClient
                import os

                # Determine the target run: same one synthetic eval used
                target_run = None
                if request.include_synthetic:
                    synth_result = results.get("synthetic_evaluation", {})
                    target_run = synth_result.get("run_id")
                if not target_run:
                    from ml_backend.smartstable_core.manager_loader import (
                        get_model_info,
                    )

                    target_run = get_model_info().get("mlflow_run_id")

                if target_run:
                    client = MlflowClient()
                    std_eval = results.get("standard_evaluation", {})
                    
                    import json
                    import tempfile
                    metrics = std_eval.get("metrics", {})
                    if metrics:
                        # Log standard metrics as file
                        metrics_file_path = os.path.join(tempfile.gettempdir(), "labeled_tasks_metrics.json")
                        with open(metrics_file_path, "w") as f:
                            json.dump(metrics, f, indent=4)
                        client.log_artifact(target_run, metrics_file_path, artifact_path="evaluation_metrics")
                        
                        # Log overall metrics
                        overall = metrics.get("overall", {})
                        if overall:
                            client.log_metric(target_run, "labeled_precision", overall.get("precision", 0.0))
                            client.log_metric(target_run, "labeled_recall", overall.get("recall", 0.0))
                            client.log_metric(target_run, "labeled_f1", overall.get("f1", 0.0))
                            
                        # Log per-label metrics
                        per_label = metrics.get("per_label", {})
                        for label, label_metrics in per_label.items():
                            safe_label = label.replace(" ", "_").replace("-", "_")
                            client.log_metric(target_run, f"labeled_{safe_label}_precision", label_metrics.get("precision", 0.0))
                            client.log_metric(target_run, f"labeled_{safe_label}_recall", label_metrics.get("recall", 0.0))
                            client.log_metric(target_run, f"labeled_{safe_label}_f1", label_metrics.get("f1", 0.0))

                    task_results = std_eval.get("task_results", [])
                    uploaded = 0
                    for res in task_results:
                        plot_path = res.get("plot_path")
                        if plot_path and os.path.exists(plot_path):
                            client.log_artifact(
                                target_run, plot_path, artifact_path="test_task_plots"
                            )
                            uploaded += 1
                    logger.info(
                        f"Uploaded {uploaded} real task plots and standard evaluation metrics to training run {target_run}"
                    )
            except Exception as e:
                logger.warning(f"Failed to upload real task plots to training run: {e}")

        return results

    except Exception as e:
        logger.error(f"Full evaluation failed: {e}", exc_info=True)
        results["success"] = False
        results["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))


def _resolve_short_threshold_seconds(override_value: Optional[float]) -> tuple[float, str]:
    """Resolve short-segment threshold from request override or model constant."""
    if override_value is not None:
        return float(override_value), "request.shorter_than_seconds"

    try:
        from smartstablemodel.multi_stall_manager import WIN_SECONDS

        return float(WIN_SECONDS), "smartstablemodel.multi_stall_manager.WIN_SECONDS"
    except Exception:
        return 2.0, "fallback.default_2.0"


def _task_id(task: Any) -> Any:
    if isinstance(task, dict):
        return task.get("id")
    return getattr(task, "id", None)


def _filter_tasks_by_scope(tasks: list[Any], task_scope: str) -> list[Any]:
    from ml_backend.util.label_studio_helper import filter_training_tasks, get_test_tasks

    if task_scope == "training":
        return filter_training_tasks(tasks, exclude_test=True)
    if task_scope == "test":
        return get_test_tasks(tasks)
    return tasks


def _build_short_segment_distribution(tasks: list[Any], short_threshold_seconds: float) -> dict:
    """Aggregate per-label duration stats and short-segment counts."""
    per_label = defaultdict(
        lambda: {
            "total_segments": 0,
            "short_segments": 0,
            "total_duration_seconds": 0.0,
            "short_duration_seconds": 0.0,
            "min_duration_seconds": None,
            "max_duration_seconds": None,
            "example_task_ids": set(),
        }
    )

    tasks_with_labels = 0
    total_segments = 0
    total_short_segments = 0

    for task in tasks:
        gt_segments = extract_ground_truth_labels(task)
        if not gt_segments:
            continue

        tasks_with_labels += 1
        tid = _task_id(task)
        for seg in gt_segments:
            label = str(seg.get("label", "unknown"))
            start_time = float(seg.get("start_time", 0.0) or 0.0)
            end_time = float(seg.get("end_time", 0.0) or 0.0)
            duration = max(0.0, end_time - start_time)
            is_short = duration < short_threshold_seconds

            agg = per_label[label]
            agg["total_segments"] += 1
            agg["total_duration_seconds"] += duration
            if is_short:
                agg["short_segments"] += 1
                agg["short_duration_seconds"] += duration
            if agg["min_duration_seconds"] is None or duration < agg["min_duration_seconds"]:
                agg["min_duration_seconds"] = duration
            if agg["max_duration_seconds"] is None or duration > agg["max_duration_seconds"]:
                agg["max_duration_seconds"] = duration
            if tid is not None and len(agg["example_task_ids"]) < 5:
                agg["example_task_ids"].add(tid)

            total_segments += 1
            if is_short:
                total_short_segments += 1

    classes = []
    for label, agg in per_label.items():
        total = agg["total_segments"]
        short = agg["short_segments"]
        classes.append(
            {
                "label": label,
                "total_segments": total,
                "short_segments": short,
                "short_ratio": (short / total) if total > 0 else 0.0,
                "total_duration_seconds": agg["total_duration_seconds"],
                "short_duration_seconds": agg["short_duration_seconds"],
                "mean_duration_seconds": (
                    agg["total_duration_seconds"] / total if total > 0 else 0.0
                ),
                "min_duration_seconds": agg["min_duration_seconds"] or 0.0,
                "max_duration_seconds": agg["max_duration_seconds"] or 0.0,
                "example_task_ids": sorted(agg["example_task_ids"]),
            }
        )

    classes.sort(key=lambda c: (-c["short_segments"], -c["total_segments"], c["label"]))

    return {
        "tasks_with_labels": tasks_with_labels,
        "total_labeled_segments": total_segments,
        "total_short_segments": total_short_segments,
        "short_ratio": (total_short_segments / total_segments) if total_segments > 0 else 0.0,
        "classes": classes,
    }


def _attach_current_model_metrics(
    *,
    model_architecture: str,
    class_rows: list[dict],
) -> dict:
    """Attach current model per-class metrics from champion model MLflow run."""
    from ml_backend.smartstable_core.mlflow_support import get_production_model_run_metrics

    metric_payload = get_production_model_run_metrics(
        model_architecture=model_architecture
    )
    per_label = (metric_payload or {}).get("per_label") or {}

    def _safe_label(label: str) -> str:
        return label.replace(" ", "_").replace("-", "_")

    for row in class_rows:
        label = row["label"]
        metric_row = per_label.get(label) or per_label.get(_safe_label(label))
        if metric_row is None:
            row["precision"] = None
            row["recall"] = None
            row["f1"] = None
            row["eval_support"] = None
            row["metrics_source"] = "not_present_in_production_run_metrics"
            continue

        row["precision"] = metric_row.get("precision")
        row["recall"] = metric_row.get("recall")
        row["f1"] = metric_row.get("f1")
        row["eval_support"] = metric_row.get("support")
        row["metrics_source"] = "mlflow.production_model_run"

    return {
        "evaluation_success": bool((metric_payload or {}).get("success")),
        "evaluation_model_version": (metric_payload or {}).get("model_version"),
        "mlflow_run_id": (metric_payload or {}).get("run_id"),
        "metric_prefix_used": (metric_payload or {}).get("metric_prefix_used"),
        "available_metric_prefixes": (metric_payload or {}).get(
            "available_metric_prefixes", []
        ),
    }


@metrics_router.post("/short-segment-distribution")
def short_segment_distribution_endpoint(request: ShortSegmentDistributionRequest):
    """
    Analyze labeled segment duration distribution by class.

    Default scope is training-used tasks (tasks not marked as test), matching
    the current SmartStable training pipeline semantics.
    """
    from ml_backend.util.label_studio_helper import list_tasks_by_project

    try:
        ls = get_ls_client()
        short_threshold_seconds, threshold_source = _resolve_short_threshold_seconds(
            request.shorter_than_seconds
        )

        all_tasks = list_tasks_by_project(
            request.project_id,
            ls_client=ls,
            only_annotated=True,
            exclude_test_tasks=False,
        )
        scoped_tasks = _filter_tasks_by_scope(all_tasks, request.task_scope)

        if request.max_tasks is not None:
            scoped_tasks = scoped_tasks[: request.max_tasks]

        if not scoped_tasks:
            raise HTTPException(
                status_code=404,
                detail=f"No annotated tasks found for scope '{request.task_scope}'",
            )

        distribution = _build_short_segment_distribution(scoped_tasks, short_threshold_seconds)
        if distribution["total_labeled_segments"] == 0:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No labeled segments found in selected task scope. "
                    "Check annotations or use a different task_scope."
                ),
            )

        metrics_join = None
        if request.include_current_model_metrics:
            from ml_backend.smartstable_core.manager_loader import get_model_info

            model_info = get_model_info()
            model_architecture = model_info.get("model_architecture", "panns")
            metrics_join = _attach_current_model_metrics(
                model_architecture=model_architecture,
                class_rows=distribution["classes"],
            )

        logger.info(
            "Short-segment distribution computed: project_id=%s scope=%s tasks=%s labels=%s threshold=%.3fs",
            request.project_id,
            request.task_scope,
            len(scoped_tasks),
            len(distribution["classes"]),
            short_threshold_seconds,
        )

        return {
            "success": True,
            "project_id": request.project_id,
            "task_scope": request.task_scope,
            "threshold_seconds": short_threshold_seconds,
            "threshold_source": threshold_source,
            "include_current_model_metrics": request.include_current_model_metrics,
            "total_annotated_tasks": len(all_tasks),
            "tasks_in_scope": len(scoped_tasks),
            "tasks_with_labels": distribution["tasks_with_labels"],
            "total_labeled_segments": distribution["total_labeled_segments"],
            "total_short_segments": distribution["total_short_segments"],
            "short_ratio": distribution["short_ratio"],
            "classes": distribution["classes"],
            "metrics_join": metrics_join,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Short-segment distribution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
