# ml_backend/smartstable_core/evaluation.py
"""
SmartStable model evaluation on Label Studio test tasks.

This module provides functionality to:
- Evaluate the current production model on test tasks
- Compare predictions to ground truth annotations
- Log evaluation metrics to MLflow
"""

import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import defaultdict

from ml_backend.util.label_studio_helper import (
    list_tasks_by_project,
    get_test_tasks,
    ls_url_to_local_path,
)
from ml_backend.smartstable_core.manager_loader import (
    get_multistall_manager,
    get_model_info,
    get_model_version_string,
)
from ml_backend.smartstable_core.mlflow_support import (
    is_mlflow_enabled,
    init_mlflow,
    download_test_pool_manifest,
    log_synthetic_evaluation_run,
)
from ml_audio_core.synthesizer import SoundscapeSynthesizer
from ml_audio_core.evaluation import calculate_segment_based_metrics
from ml_audio_core.audio_processing import extract_loudness_features

logger = logging.getLogger("SmartStableEvaluation")


# ─── Annotation Extraction ────────────────────────────────────────────────────


def extract_ground_truth_labels(task) -> List[Dict[str, Any]]:
    """
    Extract ground truth labels from a Label Studio task's annotations.

    Args:
        task: A Label Studio task object or dict

    Returns:
        List of dicts with {label, start, end} for each annotated segment
    """
    from smartstablemodel.labels import get_label_remapping

    remapping = get_label_remapping()

    annotations = []

    # Get annotations from task
    if hasattr(task, "annotations") and task.annotations:
        annotations = task.annotations
    elif isinstance(task, dict) and "annotations" in task:
        annotations = task.get("annotations", [])

    ground_truth = []

    for ann in annotations:
        # Get the result array
        results = []
        if isinstance(ann, dict):
            results = ann.get("result", [])
        elif hasattr(ann, "result"):
            results = ann.result or []

        for result in results:
            result_dict = result if isinstance(result, dict) else result.__dict__

            # Only process label annotations (not choices, ratings, etc.)
            result_type = result_dict.get("type", "")
            if result_type != "labels":
                continue

            value = result_dict.get("value", {})
            labels = value.get("labels", [])
            start = value.get("start", 0)
            end = value.get("end", 0)

            for label in labels:
                mapped_label = remapping.get(label, label)
                ground_truth.append(
                    {"label": mapped_label, "start_time": start, "end_time": end}
                )

    # Merge adjacent segments with same label
    if not ground_truth:
        return []

    ground_truth.sort(key=lambda x: x["start_time"])

    merged = []
    # We use a dict to track 'last seen' per label to handle overlapping different labels
    last_by_label = {}

    for gt in ground_truth:
        label = gt["label"]
        if label in last_by_label:
            last = last_by_label[label]
            if gt["start_time"] <= last["end_time"] + 0.1:
                last["end_time"] = max(last["end_time"], gt["end_time"])
                continue

        # New segment for this label
        new_gt = gt.copy()
        merged.append(new_gt)
        last_by_label[label] = new_gt

    return merged


def extract_predicted_labels(segments) -> List[Dict[str, Any]]:
    """
    Extract predicted labels from model output segments and merge adjacent ones.

    Args:
        segments: List of prediction segments from MultiStallDetectorManager

    Returns:
        List of merged dicts with {label, start_time, end_time, probability}
    """
    if not segments:
        return []

    # Sort just in case, though usually they are chronological
    sorted_segs = sorted(segments, key=lambda x: getattr(x, "start_time", 0))

    merged = []
    current = None

    for seg in sorted_segs:
        label = seg.predicted_label
        start = seg.start_time
        end = seg.end_time
        prob = getattr(seg, "probability", 0.5)

        if current is None:
            current = {
                "label": label,
                "start_time": start,
                "end_time": end,
                "probability": prob,
            }
        elif label == current["label"] and start <= current["end_time"] + 0.1:
            # Merge adjacent
            current["end_time"] = max(current["end_time"], end)
            current["probability"] = max(current["probability"], prob)
        else:
            merged.append(current)
            current = {
                "label": label,
                "start_time": start,
                "end_time": end,
                "probability": prob,
            }

    if current:
        merged.append(current)

    return merged


# ─── Segment Matching ─────────────────────────────────────────────────────────


def calculate_iou(seg1: Dict, seg2: Dict) -> float:
    """Calculate Intersection over Union for two time segments."""
    start1, end1 = seg1["start_time"], seg1["end_time"]
    start2, end2 = seg2["start_time"], seg2["end_time"]

    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)

    if intersection_start >= intersection_end:
        return 0.0

    intersection = intersection_end - intersection_start
    union = (end1 - start1) + (end2 - start2) - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def match_predictions_to_ground_truth(
    predictions: List[Dict], ground_truth: List[Dict], iou_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Match predictions to ground truth segments using IoU.

    Args:
        predictions: List of predicted segments
        ground_truth: List of ground truth segments
        iou_threshold: Minimum IoU to consider a match

    Returns:
        Dict with matched pairs, unmatched predictions (false positives),
        and unmatched ground truth (false negatives)
    """
    matched_predictions = set()
    matched_ground_truth = set()
    matches = []

    # For each ground truth, find best matching prediction
    for gt_idx, gt in enumerate(ground_truth):
        best_match = None
        best_iou = iou_threshold
        best_pred_idx = None

        for pred_idx, pred in enumerate(predictions):
            if pred_idx in matched_predictions:
                continue

            iou = calculate_iou(pred, gt)
            if iou >= best_iou:
                best_match = pred
                best_iou = iou
                best_pred_idx = pred_idx

        if best_match is not None:
            matches.append(
                {
                    "ground_truth": gt,
                    "prediction": best_match,
                    "iou": best_iou,
                    "correct": gt["label"] == best_match["label"],
                }
            )
            matched_predictions.add(best_pred_idx)
            matched_ground_truth.add(gt_idx)

    # Unmatched predictions are false positives
    false_positives = [
        p for i, p in enumerate(predictions) if i not in matched_predictions
    ]

    # Unmatched ground truth are false negatives
    false_negatives = [
        gt for i, gt in enumerate(ground_truth) if i not in matched_ground_truth
    ]

    return {
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


# ─── Metrics Calculation ──────────────────────────────────────────────────────


def calculate_evaluation_metrics(
    all_matches: List[Dict],
    all_false_positives: List[Dict],
    all_false_negatives: List[Dict],
) -> Dict[str, Any]:
    """
    Calculate overall evaluation metrics.

    Returns:
        Dict with precision, recall, f1, accuracy, and per-label metrics
    """
    # Overall segment-level metrics
    total_matches = len(all_matches)
    correct_matches = sum(1 for m in all_matches if m["correct"])
    total_predictions = total_matches + len(all_false_positives)
    total_ground_truth = total_matches + len(all_false_negatives)

    # Precision: of all predictions, how many were correct matches
    precision = correct_matches / total_predictions if total_predictions > 0 else 0.0

    # Recall: of all ground truth, how many were correctly detected
    recall = correct_matches / total_ground_truth if total_ground_truth > 0 else 0.0

    # F1 score
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Per-label metrics
    label_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for match in all_matches:
        gt_label = match["ground_truth"]["label"]
        pred_label = match["prediction"]["label"]

        if match["correct"]:
            label_metrics[gt_label]["tp"] += 1
        else:
            label_metrics[gt_label]["fn"] += 1
            label_metrics[pred_label]["fp"] += 1

    for fp in all_false_positives:
        label_metrics[fp["label"]]["fp"] += 1

    for fn in all_false_negatives:
        label_metrics[fn["label"]]["fn"] += 1

    # Calculate per-label precision, recall, f1
    label_results = {}
    for label, counts in label_metrics.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]

        label_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        label_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        label_f1 = (
            2 * label_precision * label_recall / (label_precision + label_recall)
            if (label_precision + label_recall) > 0
            else 0.0
        )

        label_results[label] = {
            "precision": label_precision,
            "recall": label_recall,
            "f1": label_f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "support": tp + fn,  # total ground truth for this label
        }

    return {
        "overall": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_correct": correct_matches,
            "total_predictions": total_predictions,
            "total_ground_truth": total_ground_truth,
            "false_positives": len(all_false_positives),
            "false_negatives": len(all_false_negatives),
        },
        "per_label": dict(label_results),
    }


# ─── Main Evaluation Function ─────────────────────────────────────────────────


def evaluate_on_test_tasks(
    project_id: int,
    ls_client=None,
    iou_threshold: float = 0.3,
    log_to_mlflow: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate the current production model on all test tasks.

    Args:
        project_id: Label Studio project ID
        ls_client: Optional Label Studio client
        iou_threshold: Minimum IoU for segment matching
        log_to_mlflow: Whether to log results to MLflow

    Returns:
        Dict with evaluation metrics and task-level details
    """
    if ls_client is None:
        from ml_backend.ls_client import get_ls_client

        ls_client = get_ls_client()

    manager = get_multistall_manager()
    if manager is None:
        raise RuntimeError("MultiStallDetectorManager not initialized")

    model_info = get_model_info()
    model_version = get_model_version_string()

    logger.info(
        f"Starting evaluation on project {project_id} using model: {model_version}"
    )

    # Fetch all annotated tasks, then filter to test tasks only
    all_tasks = list_tasks_by_project(
        project_id,
        ls_client=ls_client,
        only_annotated=True,
        exclude_test_tasks=False,  # We want ALL tasks, then filter
    )

    test_tasks = get_test_tasks(all_tasks)

    if not test_tasks:
        logger.warning(f"No test tasks found in project {project_id}")
        return {
            "success": False,
            "message": "No test tasks found",
            "model_version": model_version,
        }

    logger.info(f"Found {len(test_tasks)} test tasks for evaluation")

    # Evaluate each task
    all_matches = []
    all_false_positives = []
    all_false_negatives = []
    task_results = []

    for task in test_tasks:
        task_id = getattr(task, "id", None) or task.get("id")
        data = task.data if hasattr(task, "data") else task.get("data", {})
        audio_url = data.get("audio", "")

        try:
            local_path = ls_url_to_local_path(audio_url)
            stable_id, stall_id = manager.extract_stall_info_from_filename(local_path)

            # Get predictions
            segments = manager.process_audio_file(
                local_path, stable_id=stable_id, stall_id=stall_id
            )
            # Get ground truth
            ground_truth = extract_ground_truth_labels(task)

            logger.info(
                f"Task {task_id}: Extracted {len(ground_truth)} GT segments and {len(segments)} raw segments from manager."
            )
            predictions = extract_predicted_labels(segments)
            logger.info(f"Task {task_id}: {len(predictions)} merged predictions.")

            # Match predictions to ground truth
            match_result = match_predictions_to_ground_truth(
                predictions, ground_truth, iou_threshold
            )

            all_matches.extend(match_result["matches"])
            all_false_positives.extend(match_result["false_positives"])
            all_false_negatives.extend(match_result["false_negatives"])

            task_result = {
                "task_id": task_id,
                "audio": os.path.basename(local_path),
                "num_predictions": len(predictions),
                "num_ground_truth": len(ground_truth),
                "num_matches": len(match_result["matches"]),
                "num_correct": sum(1 for m in match_result["matches"] if m["correct"]),
                "num_false_positives": len(match_result["false_positives"]),
                "num_false_negatives": len(match_result["false_negatives"]),
            }
            task_results.append(task_result)

            logger.debug(
                f"Task {task_id}: {task_result['num_correct']}/{task_result['num_ground_truth']} correct"
            )

        except Exception as e:
            logger.error(f"Failed to evaluate task {task_id}: {e}", exc_info=True)
            task_results.append({"task_id": task_id, "error": str(e)})
            continue

        # Generate Task Plot
        try:
            from ml_backend.smartstable_core.mlflow_support import (
                _generate_sed_event_roll,
            )
            import tempfile

            # Predicts are in 'matching_result['matches']' etc? No, we have 'predictions' list.
            # predictions is List[Dict] with {label, start, end, probability}

            # Ground truth is 'ground_truth'

            # Audio is at 'audio_path'
            # Load audio for plotting (librosa style) - might be slow but requested.
            # _generate_sed_event_roll takes (audio, sr, gt, preds, duration, path)
            # We have audio loaded in 'audio' numpy array from MultiStallManager?
            # No, 'smartstable_predict' or similar handles it.
            # We called 'manager.process_file' which returns segments.
            # We don't have the raw audio array here unless we load it.
            # Loading it again for plotting is safer than refactoring everything.

            plot_path = os.path.join(tempfile.gettempdir(), f"task_{task_id}_eval.png")
            # Load audio for viz
            # Use manager's processor to load audio (via Rust backend)
            y, sr = manager.audio_processor.load_and_resample_audio(local_path)
            duration = len(y) / sr

            _generate_sed_event_roll(
                y, sr, ground_truth, predictions, duration, plot_path
            )

            # Add to results for logging later
            task_results[-1]["plot_path"] = plot_path

        except Exception as pe:
            logger.warning(f"Failed to generate plot for task {task_id}: {pe}")

    # Calculate overall metrics
    metrics = calculate_evaluation_metrics(
        all_matches, all_false_positives, all_false_negatives
    )

    result = {
        "success": True,
        "model_version": model_version,
        "model_source": model_info.get("source", "unknown"),
        "evaluation_time": datetime.now().isoformat(),
        "iou_threshold": iou_threshold,
        "num_test_tasks": len(test_tasks),
        "num_evaluated": len([t for t in task_results if "error" not in t]),
        "metrics": metrics,
        "task_results": task_results,
    }

    # Log to MLflow
    if log_to_mlflow and is_mlflow_enabled():
        run_id = _log_evaluation_to_mlflow(result, model_info)
        result["mlflow_run_id"] = run_id

    logger.info(
        f"Evaluation complete: F1={metrics['overall']['f1']:.3f}, "
        f"Precision={metrics['overall']['precision']:.3f}, "
        f"Recall={metrics['overall']['recall']:.3f}"
    )

    return result


def _log_evaluation_to_mlflow(result: Dict[str, Any], model_info: Dict[str, Any]):
    """Log evaluation results to MLflow."""
    try:
        import mlflow

        init_mlflow("smartstable_evaluation")

        run_name = f"test_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            # Log model info
            mlflow.set_tags(
                {
                    "evaluation_type": "test_set",
                    "model_version": result["model_version"],
                    "model_source": result.get("model_source", "unknown"),
                }
            )

            # Log overall metrics
            overall = result["metrics"]["overall"]
            mlflow.log_metrics(
                {
                    "test_precision": overall["precision"],
                    "test_recall": overall["recall"],
                    "test_f1": overall["f1"],
                    "test_total_predictions": overall["total_predictions"],
                    "test_total_ground_truth": overall["total_ground_truth"],
                    "test_correct": overall["total_correct"],
                    "test_false_positives": overall["false_positives"],
                    "test_false_negatives": overall["false_negatives"],
                    "test_num_tasks": result["num_test_tasks"],
                }
            )

            # Log per-label metrics
            for label, label_metrics in result["metrics"]["per_label"].items():
                safe_label = label.replace(" ", "_").replace("-", "_")
                mlflow.log_metrics(
                    {
                        f"test_{safe_label}_precision": label_metrics["precision"],
                        f"test_{safe_label}_recall": label_metrics["recall"],
                        f"test_{safe_label}_f1": label_metrics["f1"],
                        f"test_{safe_label}_support": label_metrics["support"],
                    }
                )

            # Log params
            mlflow.log_params(
                {
                    "iou_threshold": result["iou_threshold"],
                    "num_test_tasks": result["num_test_tasks"],
                }
            )

            # Log per-task plots if they exist
            task_results = result.get("task_results", [])
            logged_plots = 0
            for res in task_results:
                plot_path = res.get("plot_path")
                if plot_path and os.path.exists(plot_path):
                    mlflow.log_artifact(plot_path, artifact_path="test_task_plots")
                    logged_plots += 1

            logger.info(
                f"Logged evaluation to MLflow run '{run_name}': "
                f"{logged_plots}/{len(task_results)} task plots uploaded"
            )
            return mlflow.active_run().info.run_id

    except Exception as e:
        logger.error(f"Failed to log to MLflow: {e}", exc_info=True)
        return None


# ─── Synthetic Evaluation ───────────────────────────────────────────────────


def run_synthetic_evaluation(
    run_id: Optional[str] = None,
    project_id: int = 1,
    duration: float = 600.0,
    events_per_min: float = 5.0,
    ls_client=None,
    log_to_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a synthetic evaluation track from a model's test pool and evaluate it.
    """
    if ls_client is None:
        from ml_backend.ls_client import get_ls_client

        ls_client = get_ls_client()

    manager = get_multistall_manager()
    if manager is None:
        raise RuntimeError("MultiStallDetectorManager not initialized")

    # Auto-detect run_id if not provided
    if not run_id:
        model_info = get_model_info()
        run_id = model_info.get("mlflow_run_id")
        logger.info(f"DEBUG: get_model_info returned: {model_info}")
        logger.info(f"DEBUG: Auto-detected run_id: {run_id}")

        if not run_id:
            raise ValueError(
                "No run_id provided and currently loaded model has no MLflow run_id."
            )
        logger.info(f"Auto-detected run_id from current model: {run_id}")

    classifier = manager.get_soundscape_classifier()
    processor = manager.audio_processor

    # 1. Download manifest
    manifest = download_test_pool_manifest(run_id)
    if not manifest:
        raise ValueError(f"Could not download test pool manifest for run {run_id}")

    # 2. Build filename -> absolute path mapping from LS tasks
    logger.info(f"Building filename mapping for project {project_id}...")

    # 2. Build filename -> absolute path mapping by scanning local directory
    # OPTIMIZATION: Instead of fetching 10k tasks via LS API, scan the mounted volume directly.
    logger.info("Scanning local audio directory for filename mapping...")

    filename_to_path = {}
    search_dirs = [
        "/label-studio/data/audio",
        "/label-studio/data",  # Fallback/General
    ]

    scanned_count = 0
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue

        for root, _, files in os.walk(search_dir):
            for f in files:
                # We map basename -> full path
                # If duplicates exist, later ones overwrite (usually fine if filenames are unique IDs)
                if f.endswith((".wav", ".flac", ".mp3")):
                    filename_to_path[f] = os.path.join(root, f)
                    scanned_count += 1

    logger.info(
        f"Built mapping for {len(filename_to_path)} files (scanned {scanned_count})"
    )

    # 3. Create AudioProcessor (needed for loading)
    # processor = manager.audio_processor (already obtained)

    # 4. Generate Synthetic Track using ONLY files in the test manifest
    logger.info("Generating synthetic evaluation track...")

    # Manifest is Dict[label, List[sample_id]]
    # sample_id is "filename#time_offset"

    # We need to map: "filename" -> "/abs/path/to/filename"

    missing_files = 0
    found_files = 0
    valid_test_events = []

    # Iterate properly with labels
    for label, samples in manifest.items():
        for sample_id in samples:
            # Handle timestamps with potential hash in filename?
            # Use rsplit to separate only the last part (offset)
            if "#" not in sample_id:
                logger.warning(f"Invalid sample_id format: {sample_id}")
                continue

            fname, offset_str = sample_id.rsplit("#", 1)

            if fname in filename_to_path and os.path.exists(filename_to_path[fname]):
                valid_test_events.append(
                    {
                        "label": label,
                        "source_path": filename_to_path[fname],
                        "source_offset": float(offset_str),
                        "duration": 4.0,  # Standard duration
                    }
                )
                found_files += 1
            else:
                missing_files += 1

    if found_files == 0:
        raise ValueError(
            f"Could not find any local audio files matching test manifest! Missing: {missing_files}"
        )

    logger.info(
        f"Found {found_files} valid test sample segments (missing {missing_files})"
    )

    # 3. Create Synthesizer
    synthesizer = SoundscapeSynthesizer(sr=processor.sr)

    # The original code had a `sample_pool` which was a dict of `label -> list of (audio, sid)`.
    # The new `valid_test_events` is a list of dicts, which is more suitable for `generate_random_soundscape`
    # if it's adapted to take `foreground_events` directly.
    # For now, we'll adapt the `generate_random_soundscape` call to use `valid_test_events`.
    # The `total_loaded` count should reflect `found_files`.
    total_loaded = found_files  # Update total_loaded to reflect the new logic

    # 4. Generate synthetic soundscape
    import random

    # Calculate how many events to insert
    num_events = int((duration / 60.0) * events_per_min)
    logger.info(
        f"Generating synthetic track with ~{num_events} events from {found_files} available samples."
    )

    foreground_events = []

    if valid_test_events:
        for _ in range(num_events):
            # Pick a random sample from the pre-validated list
            event_spec = random.choice(valid_test_events)

            try:
                # Load audio on demand (lazy loading)
                # event_spec has: source_path, source_offset, duration, label
                audio_clip, _ = processor.load_and_resample_audio(
                    event_spec["source_path"],
                    offset=event_spec["source_offset"],
                    duration=event_spec["duration"],
                )

                # Random start time in the track
                # Ensure it fits within track duration
                max_start = duration - event_spec["duration"]
                if max_start < 0:
                    start_time = 0
                else:
                    start_time = random.uniform(0, max_start)

                foreground_events.append(
                    {
                        "label": event_spec["label"],
                        "start_time": start_time,
                        "audio": audio_clip,
                        "metadata": {
                            "source": os.path.basename(event_spec["source_path"]),
                            "offset": event_spec["source_offset"],
                            "duration": event_spec["duration"],
                        },
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to load sample for synthetic track: {e}")

    # Create the track
    # We don't have a background loop loaded here, assuming silence or noise if handled by synthesizer?
    # Original code had `synthesizer = SoundscapeSynthesizer(sr=processor.sr)`.
    # Synthesizer generally adds silence/noise if no background provided?
    # Base implementation initializes canvas with zeros or bg.
    # We'll pass background_audio=None as before.
    audio_track, gt_events = synthesizer.create_track(
        duration_seconds=duration, foreground_events=foreground_events
    )

    # 5. Run evaluation
    # Split audio into 2-second windows with 50% overlap (matches production)
    windows, times, _ = processor.create_sliding_windows(
        audio_track, segment_duration=2.0, overlap=0.5
    )

    logger.info(f"Running inference on {len(windows)} segments...")
    # Convert list of windows to a single numpy array batch
    audio_batch = np.stack(windows)
    results = classifier.infer_audio_segment(
        audio_batch, batch_size=32, return_probabilities=True
    )

    # Load production thresholds and silence config
    from smartstablemodel.config import load_metamodel_config

    metamodel_config = load_metamodel_config()
    classifier_thresholds = metamodel_config.model_parameters.get(
        "classifier_thresholds", {}
    )
    silence_threshold = metamodel_config.model_parameters.get(
        "silence_threshold", -60.0
    )
    model_arch = getattr(classifier, "model_architecture", "panns")
    label_thr = float(classifier_thresholds.get(model_arch, 0.9))

    IGNORE_LABELS = {"unsure", "normal"}

    logger.info(
        f"Synthetic eval using production thresholds: "
        f"label_thr={label_thr}, silence_thr={silence_threshold} dBFS"
    )

    # Compute loudness for silence detection
    loudness_values = [extract_loudness_features(w) for w in windows]

    # Prepare GT and Predictions per segment
    # Also track confidence stats for windows that have GT events
    gt_segments = []
    pred_segments = []

    # Confidence distribution: for windows where GT has a real event,
    # record the model's best non-normal confidence to understand misses
    conf_buckets = {"silence": 0, "0.0-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, "0.9+": 0}
    conf_per_class = {}  # label -> list of confidences

    for i, (start_time, end_time) in enumerate(times):
        # GT: Find overlapping events
        gt_labels = []
        for event in gt_events:
            e_start = event["start_time"]
            e_end = event["end_time"]
            seg_overlap = max(0.0, min(end_time, e_end) - max(start_time, e_start))
            if seg_overlap > 0.1 * (end_time - start_time):
                gt_labels.append(event["label"])

        if not gt_labels:
            gt_labels = ["normal"]
        gt_segments.append(gt_labels)

        has_gt_event = any(lbl not in {"normal", "silence"} for lbl in gt_labels)

        # --- Prediction (aligned with production _process_batch) ---

        # Silence detection: if too quiet, force "silence" (matches production)
        if loudness_values[i] < silence_threshold:
            pred_segments.append(["silence"])
            if has_gt_event:
                conf_buckets["silence"] += 1
            continue

        p_dict = results[i]
        top_label, top_conf = max(p_dict.items(), key=lambda kv: kv[1])

        # Filter out IGNORE_LABELS ("normal", "unsure") — pick next best
        if top_label in IGNORE_LABELS:
            filtered = {k: v for k, v in p_dict.items() if k not in IGNORE_LABELS}
            if filtered:
                top_label, top_conf = max(filtered.items(), key=lambda kv: kv[1])
            else:
                pred_segments.append(["normal"])
                if has_gt_event:
                    conf_buckets["0.0-0.5"] += 1
                continue

        # Track confidence for windows with GT events
        if has_gt_event:
            if top_conf >= 0.9:
                conf_buckets["0.9+"] += 1
            elif top_conf >= 0.7:
                conf_buckets["0.7-0.9"] += 1
            elif top_conf >= 0.5:
                conf_buckets["0.5-0.7"] += 1
            else:
                conf_buckets["0.0-0.5"] += 1
            # Per-class tracking
            for gt_lbl in gt_labels:
                if gt_lbl in {"normal", "silence"}:
                    continue
                conf_per_class.setdefault(gt_lbl, []).append(
                    {"confidence": float(top_conf), "predicted_as": top_label}
                )

        # Apply production confidence threshold
        if top_conf >= label_thr:
            pred_segments.append([top_label])
        else:
            pred_segments.append(["normal"])

    # Log confidence distribution
    total_gt_windows = sum(conf_buckets.values())
    logger.info(
        f"Confidence distribution for {total_gt_windows} windows with GT events:"
    )
    for bucket, count in conf_buckets.items():
        pct = (count / total_gt_windows * 100) if total_gt_windows > 0 else 0
        logger.info(f"  {bucket:>8s}: {count:4d}  ({pct:5.1f}%)")

    # 6. Calculate SED Metrics
    sed_metrics = calculate_segment_based_metrics(gt_segments, pred_segments)

    # Append confidence stats to sed_metrics
    sed_metrics["confidence_distribution"] = conf_buckets
    sed_metrics["confidence_per_class"] = {
        lbl: {
            "mean_conf": float(np.mean([c["confidence"] for c in confs])),
            "median_conf": float(np.median([c["confidence"] for c in confs])),
            "above_threshold": sum(1 for c in confs if c["confidence"] >= label_thr),
            "total_windows": len(confs),
        }
        for lbl, confs in conf_per_class.items()
    }

    # Reconstruct predicted events for visualization
    # Filter out "normal" and "silence" — they are non-event labels
    NON_EVENT_LABELS = {"normal", "silence"}
    predicted_events = []
    if pred_segments:
        current_event = None
        for i, seg in enumerate(pred_segments):
            # seg is a list of labels, take the first one (top prediction)
            label = seg[0]
            start_time = times[i][0]
            end_time = times[i][1]

            # Skip non-event labels for visualization
            if label in NON_EVENT_LABELS:
                if current_event:
                    predicted_events.append(current_event)
                    current_event = None
                continue

            if current_event and current_event["label"] == label:
                # Extend current event
                current_event["end_time"] = end_time
            else:
                # Close previous event
                if current_event:
                    predicted_events.append(current_event)
                # Start new event
                current_event = {
                    "label": label,
                    "start_time": start_time,
                    "end_time": end_time,
                }

        # Append last event
        if current_event:
            predicted_events.append(current_event)

    logger.info(
        f"Synthetic eval: {len(predicted_events)} predicted events "
        f"(filtered from {len(times)} windows, threshold={label_thr})"
    )

    # 7. Log to MLflow
    # Use explicit target run ID if provided, otherwise fallback to source run ID
    target_run_id = log_to_run_id or run_id

    if target_run_id:
        try:
            log_synthetic_evaluation_run(
                run_id=target_run_id,
                sed_metrics=sed_metrics,
                audio_track=audio_track,
                annotations=gt_events,
                predicted_events=predicted_events,
                sr=processor.sr,
            )
        except Exception as e:
            logger.warning(f"Failed to log synthetic evaluation results: {e}")

    # 8. Package results
    return {
        "run_id": run_id,
        "model_version": get_model_version_string(),
        "duration": duration,
        "events_per_min": events_per_min,
        "num_samples_loaded": total_loaded,
        "sed_metrics": sed_metrics,
        "num_events_generated": len(gt_events),
    }


# ─── Synthetic Scenarios Generation & Evaluation ──────────────────────────────

import json
from pathlib import Path
from pydantic import BaseModel

SCENARIO_OUTPUT_DIR = Path("/label-studio/data/audio_test_files")


def generate_synthetic_scenario(request: Any, ls_client: Any) -> Dict[str, Any]:
    """
    Generate a full synthetic scenario (e.g., a whole night) based on configurable blocks,
    and save it to disk. Does not run evaluation.
    """
    SCENARIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_wav = SCENARIO_OUTPUT_DIR / f"{request.name}.wav"
    output_json = SCENARIO_OUTPUT_DIR / f"{request.name}.json"

    # 1. Download test pool manifest
    run_id = request.run_id
    if not run_id:
        from ml_backend.smartstable_core.manager_loader import get_model_info

        model_info = get_model_info()
        run_id = model_info.get("mlflow_run_id")
        logger.info(f"Using loaded model run_id={run_id} as fallback")

    if not run_id:
        raise RuntimeError(
            "Could not retrieve test pool from MLflow. Ensure a model is loaded or run_id is passed."
        )

    logger.info(f"Downloading test pool using run_id={run_id}")
    manifest = download_test_pool_manifest(run_id)
    if not manifest:
        raise RuntimeError(
            "Could not retrieve test pool from MLflow. Ensure the model has a test_pool_manifest artifact."
        )

    total_manifest_samples = sum(len(samples) for samples in manifest.values())
    if total_manifest_samples == 0:
        raise RuntimeError(
            "Test pool is empty. Please annotate test labels and train a model first."
        )
    logger.info(
        f"Manifest has {total_manifest_samples} sample IDs across {len(manifest)} labels"
    )
    logger.info(f"Available manifest labels: {list(manifest.keys())}")

    # 2. Build filename -> local path mapping (same as run_synthetic_evaluation)
    manager = get_multistall_manager()
    if manager is None:
        raise RuntimeError("MultiStallDetectorManager not initialized")
    processor = manager.audio_processor

    filename_to_path = {}
    search_dirs = ["/label-studio/data/audio", "/label-studio/data"]
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        for root, _, files in os.walk(search_dir):
            for f in files:
                if f.endswith((".wav", ".flac", ".mp3")):
                    filename_to_path[f] = os.path.join(root, f)
    logger.info(f"Built local file mapping with {len(filename_to_path)} audio files")

    # 3. Compute total duration and track empty arrays
    total_duration_sec = request.total_duration_hours * 3600
    foreground_events = []

    def _load_sample(sample_id: str, label: str):
        """Resolve a manifest sample_id to loaded audio. Returns (audio_array, sample_id) or None."""
        if "#" not in sample_id:
            logger.warning(f"Invalid sample_id format (no '#'): {sample_id}")
            return None
        fname, offset_str = sample_id.rsplit("#", 1)
        if fname not in filename_to_path:
            return None
        try:
            audio_clip, _ = processor.load_and_resample_audio(
                filename_to_path[fname],
                offset=float(offset_str),
                duration=4.0,
            )
            return audio_clip
        except Exception as e:
            logger.warning(f"Failed to load {sample_id}: {e}")
            return None

    # 4. Process each configuration block
    for block in request.behavior_blocks:
        start_sec = block.start_hour * 3600
        end_sec = block.end_hour * 3600

        if end_sec <= start_sec:
            logger.warning(f"Block '{block.type}' has end <= start. Skipping.")
            continue

        block_duration = end_sec - start_sec
        num_events = int((block_duration / 60.0) * block.events_per_minute)

        valid_labels = [
            lbl for lbl in block.labels if lbl in manifest and manifest[lbl]
        ]
        if not valid_labels:
            logger.warning(
                f"No samples available for requested labels {block.labels} in block {block.type}. Skipping."
            )
            continue

        for _ in range(num_events):
            label = np.random.choice(valid_labels)
            pool = manifest[label]
            sample_id = pool[np.random.randint(len(pool))]
            audio = _load_sample(sample_id, label)
            if audio is None:
                continue

            event_start = np.random.uniform(start_sec, max(start_sec, end_sec - 2.0))

            foreground_events.append(
                {
                    "audio": audio,
                    "label": label,
                    "start_time": event_start,
                    "metadata": {"sample_id": sample_id, "block_type": block.type},
                }
            )

    # 5. Process background 'normal' tracks uniformly across the entire timeline
    if (
        request.background_events_per_minute > 0
        and "normal" in manifest
        and manifest["normal"]
    ):
        num_bg = int((total_duration_sec / 60.0) * request.background_events_per_minute)
        for _ in range(num_bg):
            pool = manifest["normal"]
            sample_id = pool[np.random.randint(len(pool))]
            audio = _load_sample(sample_id, "normal")
            if audio is None:
                continue
            event_start = np.random.uniform(0, max(0, total_duration_sec - 2.0))
            foreground_events.append(
                {
                    "audio": audio,
                    "label": "normal",
                    "start_time": event_start,
                    "metadata": {"sample_id": sample_id, "block_type": "background"},
                }
            )

    # Sort events by time
    foreground_events.sort(key=lambda x: x["start_time"])

    # 5. Synthesize
    logger.info(
        f"Synthesizing {total_duration_sec}s track with {len(foreground_events)} events..."
    )
    synth = SoundscapeSynthesizer(sr=16000)
    mixed_audio, annotations = synth.create_track(
        duration_seconds=total_duration_sec,
        foreground_events=foreground_events,
    )

    # 6. Save to disk
    logger.info(f"Saving scenario to {output_wav}")
    synth.save_track(mixed_audio, annotations, output_wav)

    return {
        "scenario_name": request.name,
        "saved_wav": str(output_wav),
        "saved_json": str(output_json),
        "total_duration_hours": request.total_duration_hours,
        "num_events": len(annotations),
    }


def evaluate_saved_scenario(
    scenario_name: str, log_to_mlflow: bool = True
) -> Dict[str, Any]:
    """
    Load a pre-generated synthetic scenario from disk and evaluate the current model on it,
    logging results directly to the active MLflow run.
    """

    wav_path = SCENARIO_OUTPUT_DIR / f"{scenario_name}.wav"
    json_path = SCENARIO_OUTPUT_DIR / f"{scenario_name}.json"

    if not wav_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"Scenario {scenario_name} not found at {SCENARIO_OUTPUT_DIR}. Generate it first."
        )

    logger.info(f"Loading scenario {scenario_name}...")
    manager = get_multistall_manager()
    if not manager:
        raise RuntimeError("MultiStallDetectorManager not loaded.")
    processor = manager.audio_processor

    # Use Rust-based audio loading and resampling (align with production)
    audio_track, sr = processor.load_and_resample_audio(str(wav_path))

    with open(json_path, "r") as f:
        meta = json.load(f)
        ground_truth_events = meta.get("annotations", [])

    duration = meta.get("duration", len(audio_track) / sr)

    # Run evaluation matching `run_synthetic_evaluation`
    logger.info(f"Evaluating scenario {scenario_name} ({duration / 3600:.2f} hours)")

    # Setup windows just like run_synthetic_evaluation handles it
    windows, times, _ = processor.create_sliding_windows(
        audio_track,
        segment_duration=processor.win_seconds,
        overlap=processor.win_overlap,
    )

    # Ensure windows are correctly shaped and process them
    # Because of memory on 13 hrs (23,000 windows), _process_batch already batches internally.
    batch_size = 64
    pred_segments = []

    logger.info(f"Processing {len(windows)} windows...")

    conf_buckets = {
        "silence": 0,
        "0.0-0.5": 0,
        "0.5-0.7": 0,
        "0.7-0.9": 0,
        "0.9+": 0,
    }
    from smartstablemodel.config import load_metamodel_config

    metamodel_config = load_metamodel_config()
    label_thr = metamodel_config.model_parameters.get(
        "alert_labels_confidence_floor", 0.6
    )
    silence_threshold = metamodel_config.model_parameters.get(
        "silence_threshold", -60.0
    )
    if not isinstance(label_thr, float):
        label_thr = 0.6

    conf_per_class = defaultdict(list)

    # Times are already computed by create_sliding_windows

    # Convert to batched format matching production signature

    # For safety to not block event loop, we iterate over the batches here manually to compute the arrays
    # Although manager.infer_audio_segment handles this cleanly, _process_batch gives us confidence scores!
    all_scores = []
    for i in range(0, len(windows), batch_size):
        batch_windows = windows[i : i + batch_size]
        batch_times = times[i : i + batch_size]
        batch_scores = manager._process_batch(
            audio_list=batch_windows,
            times_list=batch_times,
            results=[],
            stable_id="synthetic_scenario",
            stall_id="default",
            file_path=str(wav_path),
        )
        all_scores.extend(batch_scores)

    for i, score_obj in enumerate(all_scores):
        start_time = score_obj.start_time
        end_time = score_obj.end_time

        # Check if GT event is in this window
        window_has_gt = False
        for gt in ground_truth_events:
            if gt["label"] in ["normal", "silence"]:
                continue
            if not (end_time <= gt["start_time"] or start_time >= gt["end_time"]):
                window_has_gt = True
                break

        best_lbl = score_obj.predicted_label
        best_conf = score_obj.probability

        if best_lbl == "silence":
            pred_segments.append(["silence"])
            if window_has_gt:
                conf_buckets["silence"] += 1
            continue

        pred_segments.append([best_lbl])

        if window_has_gt:
            if best_conf < 0.5:
                conf_buckets["0.0-0.5"] += 1
            elif best_conf < 0.7:
                conf_buckets["0.5-0.7"] += 1
            elif best_conf < 0.9:
                conf_buckets["0.7-0.9"] += 1
            else:
                conf_buckets["0.9+"] += 1

            for lbl, conf in score_obj.label_probabilities.items():
                if lbl != "normal":
                    conf_per_class[lbl].append(
                        {"time": start_time, "confidence": float(conf)}
                    )

    logger.info(f"Confidence distribution for windows with GT events:")
    for bucket, count in conf_buckets.items():
        logger.info(f"  {bucket:>8s}: {count:4d}")

    # Build GT segments for metrics matching
    gt_segments = []
    for t_start, t_end in times:
        window_gt_labels = []
        for gt in ground_truth_events:
            if not (t_end <= gt["start_time"] or t_start >= gt["end_time"]):
                window_gt_labels.append(gt["label"])

        gt_segments.append(window_gt_labels if window_gt_labels else ["normal"])

    # Ensure gaps in GT are explicitly marked silence if loudness matches
    for i, (window_audio, gt_lbls) in enumerate(zip(windows, gt_segments)):
        if ["normal"] == gt_lbls:  # Only gap
            loudness = extract_loudness_features(window_audio)
            if loudness < silence_threshold:
                gt_segments[i] = ["silence"]

    sed_metrics = calculate_segment_based_metrics(gt_segments, pred_segments)
    sed_metrics["confidence_distribution"] = conf_buckets
    sed_metrics["confidence_per_class"] = {
        lbl: {
            "mean_conf": float(np.mean([c["confidence"] for c in confs]))
            if confs
            else 0.0,
            "above_threshold": sum(1 for c in confs if c["confidence"] >= label_thr),
        }
        for lbl, confs in conf_per_class.items()
    }

    NON_EVENT_LABELS = {"normal", "silence"}
    predicted_events = []
    if pred_segments:
        current_event = None
        for i, seg in enumerate(pred_segments):
            label = seg[0]
            start_time = times[i][0]
            end_time = times[i][1]

            if label in NON_EVENT_LABELS:
                if current_event:
                    predicted_events.append(current_event)
                    current_event = None
                continue

            if current_event and current_event["label"] == label:
                current_event["end_time"] = end_time
            else:
                if current_event:
                    predicted_events.append(current_event)
                current_event = {
                    "label": label,
                    "start_time": start_time,
                    "end_time": end_time,
                }
        if current_event:
            predicted_events.append(current_event)

    logger.info(f"Scenario eval: {len(predicted_events)} predicted events found.")

    # MLFlow logging
    if log_to_mlflow:
        try:
            model_info = get_model_info()
            target_run_id = model_info.get("mlflow_run_id")

            if not target_run_id:
                # Fallback to an active run or current centroid training run
                from mlflow import active_run

                ar = active_run()
                if ar:
                    target_run_id = ar.info.run_id

            if target_run_id:
                log_synthetic_evaluation_run(
                    run_id=target_run_id,
                    sed_metrics=sed_metrics,
                    audio_track=audio_track,
                    annotations=ground_truth_events,
                    predicted_events=predicted_events,
                    sr=processor.sr,
                )
                logger.info(f"Logged scenario metrics to MLflow run {target_run_id}")
            else:
                logger.warning("Could not identify target MLflow run for logging.")
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}", exc_info=True)

    return {
        "scenario_name": scenario_name,
        "duration": duration,
        "sed_metrics": sed_metrics,
        "num_events_generated": len(ground_truth_events),
    }
