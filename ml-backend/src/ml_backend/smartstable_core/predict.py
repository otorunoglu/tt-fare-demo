import logging
import numpy as np
from typing import List

from ml_backend.util.label_studio_helper import ls_url_to_local_path
from ml_backend.smartstable_core.manager_loader import (
    get_model_info,
    get_model_version_string,
)
from smartstablemodel.labels import get_label_registry
from smartstablemodel.multi_stall_manager import MultiStallDetectorManager
from smartstablemodel.smartstable_types import MultiStallBatchResultReturn

logger = logging.getLogger("SmartStablePredict")


def group_consecutive_segments(
    segments: List[MultiStallBatchResultReturn],
    max_gap: float = 5.0,
    probability_key: str = "probability",
) -> list:
    """
    Group consecutive segments of the same label into merged events.

    Args:
        segments: List of prediction segments (should all have the same label)
        max_gap: Maximum gap in seconds between segments to still merge them
        probability_key: Attribute name to use for probability averaging

    Returns:
        List of merged events with start_time, end_time, and avg_probability
    """
    if not segments:
        return []

    # Sort by start time
    segments = sorted(segments, key=lambda x: x.start_time)

    events = []
    current_event = {
        "start_time": segments[0].start_time,
        "end_time": segments[0].end_time,
        "segments": [segments[0]],
        "probabilities": [getattr(segments[0], probability_key, 0.5)],
    }

    for segment in segments[1:]:
        # Check if this segment is close enough to the current event
        if segment.start_time - current_event["end_time"] <= max_gap:
            # Extend current event
            current_event["end_time"] = segment.end_time
            current_event["segments"].append(segment)
            current_event["probabilities"].append(
                getattr(segment, probability_key, 0.5)
            )
        else:
            # Finish current event and start new one
            current_event["avg_probability"] = float(
                np.mean(current_event["probabilities"])
            )
            events.append(current_event)

            current_event = {
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "segments": [segment],
                "probabilities": [getattr(segment, probability_key, 0.5)],
            }

    # Don't forget the last event
    current_event["avg_probability"] = float(np.mean(current_event["probabilities"]))
    events.append(current_event)

    return events


def build_ls_payload(
    segments: List[MultiStallBatchResultReturn], max_gap: float = 5.0
) -> list:
    """
    Build Label Studio payload with grouped consecutive segments.

    Groups segments by their predicted label, then merges consecutive segments
    of the same label within max_gap seconds into single events.

    Args:
        segments: List of prediction segments from the model
        max_gap: Maximum gap in seconds between segments to merge them

    Returns:
        List of Label Studio annotation objects
    """
    if not segments:
        return []

    labels_to_show = get_label_registry().get_labels_to_show_in_ls()

    # Filter to only segments with labels we want to show
    filtered_segments = [s for s in segments if s.predicted_label in labels_to_show]

    if not filtered_segments:
        return []

    logger.debug(f"Building LS payload for {len(filtered_segments)} segments")

    # Group segments by their predicted label
    label_groups = {}
    for segment in filtered_segments:
        label = segment.predicted_label or "unknown"
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(segment)

    logger.debug(
        f"Grouped into {len(label_groups)} label groups: {list(label_groups.keys())}"
    )

    ls_response = []

    # Create events for each label type
    for label, label_segments in label_groups.items():
        label_group = get_label_registry().get_label_group_by_value(label)

        # Merge consecutive segments of the same label
        events = group_consecutive_segments(
            label_segments, max_gap=max_gap, probability_key="probability"
        )

        logger.debug(
            f"Label '{label}': {len(label_segments)} segments merged into {len(events)} events"
        )

        for ev in events:
            ls_response.append(
                {
                    "from_name": label_group,
                    "to_name": "audio",
                    "type": "labels",
                    "value": {
                        "start": ev["start_time"],
                        "end": ev["end_time"],
                        "labels": [label],
                    },
                    "score": ev["avg_probability"],
                }
            )

    # --- Anomaly Handling ---
    # Process anomalies as a distinct 'label' type
    anomaly_segments = [s for s in segments if getattr(s, "is_anomaly", False)]
    if anomaly_segments:
        logger.debug(f"Found {len(anomaly_segments)} anomalous segments")

        # Merge consecutive anomaly segments
        # We can treat them as having label "anomaly_abnormal"
        for s in anomaly_segments:
            # Temp override for grouping helper
            # We rely on is_anomaly flag, but helper groups by list content
            pass

        anomaly_events = group_consecutive_segments(
            anomaly_segments, max_gap=max_gap, probability_key="anomaly_score"
        )

        # Assuming 'anomaly_abnormal' is in 'anomaly_labels' group
        # We should check if this label exists and which group it is
        try:
            # Use the value we set in labels.json
            anom_label = "anomaly_abnormal"
            anom_group = (
                get_label_registry().get_label_group_by_value(anom_label)
                or "anomaly_labels"
            )

            for ev in anomaly_events:
                ls_response.append(
                    {
                        "from_name": anom_group,
                        "to_name": "audio",
                        "type": "labels",
                        "value": {
                            "start": ev["start_time"],
                            "end": ev["end_time"],
                            "labels": [anom_label],  # "Abnormal"
                        },
                        "score": ev[
                            "avg_probability"
                        ],  # This will be the avg anomaly score
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to add anomaly events: {e}")

    return ls_response


def smartstable_predict(
    manager: MultiStallDetectorManager, task_id: int, ls, max_gap: float = 5.0
):
    """
    Run prediction on a Label Studio task and create predictions.

    Args:
        manager: The MultiStallDetectorManager instance
        task_id: Label Studio task ID
        ls: Label Studio client
        max_gap: Maximum gap in seconds between segments to merge them (default: 5.0)
    """
    try:
        # Log model info at prediction start
        model_info = get_model_info()
        model_version = get_model_version_string()
        logger.info(
            f"Starting prediction for task {task_id} using model: {model_version} (source: {model_info['source']})"
        )

        task = ls.tasks.get(task_id)
        audio_url = task.data["audio"]

        local_path = ls_url_to_local_path(audio_url)
        stable_id, stall_id = manager.extract_stall_info_from_filename(local_path)

        logger.info(
            f"Processing audio: {local_path} (stable: {stable_id}, stall: {stall_id})"
        )

        segments = manager.process_audio_file(
            local_path, stable_id=stable_id, stall_id=stall_id
        )

        # Build payload with grouped consecutive segments
        payload = build_ls_payload(segments, max_gap=max_gap)

        if payload:
            # DEBUG: Log payload details for anomalies
            anom_payloads = [p for p in payload if "anomaly" in p.get("from_name", "")]
            if anom_payloads:
                # Format for logging ONLY, do not mutate payload!
                log_msgs = []
                for p in anom_payloads:
                    s = p["value"]["start"]
                    e = p["value"]["end"]
                    log_msgs.append(
                        f"{int(s // 60)}:{int(s % 60):02d}-{int(e // 60)}:{int(e % 60):02d} (score {p['score']})"
                    )

                logger.info(
                    f"Anomaly Payloads to be sent ({len(anom_payloads)}): {log_msgs}"
                )

            # Determine overall prediction score
            # Force high score if anomalies are present to ensure they show up in UI
            # regardless of threshold
            if anom_payloads:
                prediction_score = 1.0
            else:
                prediction_score = max((p["score"] for p in payload), default=0.0)

            ls.predictions.create(
                task=task_id,
                result=payload,
                score=prediction_score,
                model_version=model_version,
            )

        logger.info(
            f"Prediction complete for task {task_id}: {len(segments)} segments -> {len(payload)} merged events"
        )

    except Exception as e:
        logger.error(f"Prediction failed for task {task_id}: {e}", exc_info=True)
