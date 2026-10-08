# ml_backend/routes/smartstable/training_routes.py

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional

from ml_backend.smartstable_core.manager_loader import (
    get_multistall_manager as get_manager,
)
from ml_backend.ls_client import get_ls_client
from ml_backend.smartstable_core.training import (
    _blob_store_training,
    create_dataset,
    smartstable_train,
    get_dataset_status,
    get_untrained_classes,
)
from smartstablemodel.soundscape_classifier import SoundscapeClassifier
from ml_backend.util.label_studio_helper import ls_url_to_local_path
from smartstablemodel.labels import get_label_registry
from smartstablemodel.config import load_metamodel_config

training_router = APIRouter()


def group_consecutive_segments(segments, max_gap=5.0):
    """
    Group consecutive segments into events.
    Expects segments to be dicts with "start", "end", "top_label", "probabilities".
    """
    if not segments:
        return []

    # Sort by start time
    segments.sort(key=lambda x: x["start"])

    events = []
    # Initialize first event
    first_score = segments[0]["probabilities"].get(segments[0]["top_label"], 0.0)
    current_event = {
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "scores": [first_score],
    }

    for seg in segments[1:]:
        # Check if this segment is close enough to the current event
        if seg["start"] - current_event["end"] <= max_gap:
            # Extend current event
            current_event["end"] = seg["end"]
            score = seg["probabilities"].get(seg["top_label"], 0.0)
            current_event["scores"].append(score)
        else:
            # Finish current event and start new one
            import numpy as np

            current_event["avg_score"] = float(np.mean(current_event["scores"]))
            del current_event["scores"]  # clean up
            events.append(current_event)

            score = seg["probabilities"].get(seg["top_label"], 0.0)
            current_event = {
                "start": seg["start"],
                "end": seg["end"],
                "scores": [score],
            }

    # Don't forget the last event
    import numpy as np

    current_event["avg_score"] = float(np.mean(current_event["scores"]))
    del current_event["scores"]
    events.append(current_event)

    return events


class TrainingRequest(BaseModel):
    project_id: int
    model_architecture: str = "panns"  # "panns", "yamnet"


class InferenceTestRequest(BaseModel):
    task_id: int
    model_architecture: str = "panns"
    apply_labels: bool = False


@training_router.get("/training/debug")
def debug_training():
    return {"status": "ok", "message": "training route reachable"}


@training_router.post("/training")
def trigger_training(req: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Trigger background training for the specified project and architecture.
    """
    manager = get_manager()
    ls = get_ls_client()

    if not manager or not ls:
        raise HTTPException(
            status_code=503, detail="Manager or LS client not initialized"
        )

    # Launch in background
    background_tasks.add_task(
        smartstable_train,
        manager=manager,
        project_id=req.project_id,
        ls=ls,
        model_architecture=req.model_architecture,
    )

    return {
        "status": "accepted",
        "message": f"Training started for project {req.project_id} with {req.model_architecture}",
    }


class DatasetStatusRequest(BaseModel):
    project_id: int | None = None  # informational; manifests aren't project-scoped yet
    version: str = "latest"


@training_router.post("/training/dataset/status")
def dataset_status(req: DatasetStatusRequest):
    status = get_dataset_status(version=req.version)
    if status is None:
        raise HTTPException(
            status_code=404, detail=f"No manifest found for version '{req.version}'"
        )
    return status


@training_router.post("/training/dataset/untrained")
def dataset_untrained(req: DatasetStatusRequest):
    ls = get_ls_client()
    out = get_untrained_classes(req.project_id, ls_client=ls)
    return out


class TrainFromManifestRequest(BaseModel):
    project_id: int
    model_architecture: str = "panns"  # "panns", "yamnet"
    manifest_version: str = "latest"  # or specific version
    seed: int = 42


@training_router.post("/training/from-manifest")
def train_from_manifest(req: TrainFromManifestRequest):
    manager = get_manager()
    history, code = _blob_store_training(
        manifest_version=req.manifest_version,
        multi_stall_manager=manager,
        model_architecture=req.model_architecture,
        seed=req.seed if hasattr(req, "seed") else 42,
    )
    if code != 200:
        raise HTTPException(status_code=code, detail="Training failed; see logs")
    return {"status": "ok", "manifest_version": req.manifest_version}


class CreateDatasetRequest(BaseModel):
    project_id: int
    test_split: float = 0.2
    seed: int
    version: Optional[str] = "new"
    dataset_path: Optional[str] = None
    test_group_overrides: Optional[dict] = None
    comment: Optional[str] = None


@training_router.post("/training/dataset/create")
def create_new_dataset(req: CreateDatasetRequest):
    manager = get_manager()
    result = create_dataset(
        project_id=req.project_id,
        manager=manager,
        test_split=req.test_split,
        seed=req.seed,
        version=req.version or "new",
        test_group_overrides=getattr(req, "test_group_overrides", None),
        comment=getattr(req, "comment", None),
    )
    if result is False:
        raise HTTPException(status_code=500, detail="Dataset creation failed; see logs")
    return {"status": "ok", **result}  # includes the manifest path


@training_router.post("/predict/test")
def test_inference(req: InferenceTestRequest):
    """
    Test inference on a specific task using a freshly loaded model of the requested architecture.
    Optionally pushes results to Label Studio.
    """
    manager = get_manager()
    ls = get_ls_client()

    if not ls:
        raise HTTPException(status_code=503, detail="LS client not initialized")

    # 1. Fetch task
    try:
        task = ls.tasks.get(req.task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {req.task_id} not found")

        # Handle task.data which might be a dict or a serializer object
        task_data = task.data

        # Helper to safely get value from dict or object
        def get_val(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        audio_url = get_val(task_data, "audio")
        if not audio_url:
            raise HTTPException(status_code=400, detail="Task has no audio data")

        local_path = ls_url_to_local_path(audio_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch task: {e}")

    # 2. Instantiate Classifier (this loads the latest model from disk for this architecture)
    try:
        # We assume the models are saved with prefixes 'stable_panns' or 'stable_yamnet'
        # and SoundscapeClassifier finds the latest one automatically.
        classifier = SoundscapeClassifier(model_architecture=req.model_architecture)

        if not hasattr(classifier, "model") or classifier.model is None:
            raise HTTPException(
                status_code=404,
                detail=f"No trained model found for architecture {req.model_architecture}",
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # 3. Running Inference
    import numpy as np

    try:
        # Use existing audio processor from manager
        processor = manager.audio_processor

        # Load full audio and create sliding windows
        audio, _ = processor.load_and_resample_audio(local_path)
        windows, times, _ = processor.create_sliding_windows(
            audio, segment_duration=2.0
        )

        # Run inference on batch
        results = []
        if len(windows) > 0:
            probs = classifier.infer_audio_segment(
                np.stack(windows), return_probabilities=True, file_path=local_path
            )

            # Combine results
            for (start, end), p in zip(times, probs):
                results.append(
                    {
                        "start": start,
                        "end": end,
                        "probabilities": p,
                        "top_label": max(p, key=p.get) if p else "unknown",
                    }
                )

        response = {
            "model_version": classifier.get_model_version(),
            "architecture": req.model_architecture,
            "results": results,
        }

        # 4. Apply Labels if requested
        if req.apply_labels:
            # Construct LS annotations
            # We will create a prediction or annotation? User said "add labels to the task".
            # Usually predictions.

            ls_results = []
            registry = get_label_registry()
            labels_to_show = registry.get_labels_to_show_in_ls()

            # Group results by label
            segments_by_label = {}
            for res in results:
                label = res["top_label"]
                if label not in segments_by_label:
                    segments_by_label[label] = []
                segments_by_label[label].append(res)

            # Load global config once
            metamodel_config = load_metamodel_config()
            classifier_thresholds = metamodel_config.model_parameters.get(
                "classifier_thresholds", {}
            )
            # Get architecture default
            default_thr = float(classifier_thresholds.get(req.model_architecture, 0.9))

            # Build grouped events
            for label, label_segments in segments_by_label.items():
                if label not in labels_to_show:
                    continue

                # Get correct UI group name (from_name)
                from_name = registry.get_label_group_by_value(label)
                if not from_name:
                    from_name = "label"

                events = group_consecutive_segments(label_segments)

                for ev in events:
                    # Use global architecture default threshold
                    threshold = default_thr

                    if ev["avg_score"] < threshold:
                        continue

                    ls_results.append(
                        {
                            "from_name": from_name,
                            "to_name": "audio",
                            "type": "labels",
                            "value": {
                                "start": ev["start"],
                                "end": ev["end"],
                                "labels": [label],
                            },
                            "score": ev["avg_score"],
                        }
                    )

            if ls_results:
                # push prediction
                ls.predictions.create(
                    task=req.task_id,
                    result=ls_results,
                    score=float(np.mean([r["score"] for r in ls_results])),
                    model_version=f"{req.model_architecture}_{classifier.get_model_version()}",
                )
                response["labels_applied"] = True
                response["predictions_count"] = len(ls_results)
            else:
                response["labels_applied"] = False
                response["message"] = "No non-normal labels found to apply."

        return response

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
