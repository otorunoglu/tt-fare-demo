"""
Generic Label Studio helper utilities.

This module contains functions that work with any Label Studio project,
independent of domain-specific concepts.

For SmartStable-specific task filtering (by stable_id, stall_id, etc.),
see: ml_backend.smartstable_core.task_filtering
"""

import json
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger("LabelStudioHelper")


# ─── Task Purpose Filtering ───────────────────────────────────────────────────
# Label Studio tasks can be marked as "test" or "train" via a Choices annotation.
# Test tasks should be excluded from training but used for model evaluation.


def is_test_task(task) -> bool:
    """
    Check if a task is marked as a test task (excluded from training).

    Looks for a 'task_purpose' choice annotation with alias/value 'test'.
    If no task_purpose is set, defaults to training (returns False).

    Args:
            task: A Label Studio task object (SDK object or dict)

    Returns:
            True if task is marked for testing only, False for training
    """
    annotations = []

    # Handle SDK task objects
    if hasattr(task, "annotations") and task.annotations:
        annotations = task.annotations
    # Handle dict format
    elif isinstance(task, dict) and "annotations" in task:
        annotations = task.get("annotations", [])

    for ann in annotations:
        # Get the result array from the annotation
        results = []
        if isinstance(ann, dict):
            results = ann.get("result", [])
        elif hasattr(ann, "result"):
            results = ann.result or []

        for result in results:
            # Check if this is a task_purpose choice
            from_name = (
                result.get("from_name", "")
                if isinstance(result, dict)
                else getattr(result, "from_name", "")
            )
            if from_name == "task_purpose":
                # Get the choice value
                value = (
                    result.get("value", {})
                    if isinstance(result, dict)
                    else getattr(result, "value", {})
                )
                choices = value.get("choices", []) if isinstance(value, dict) else []

                # Check if 'test' is selected (could be alias or display value)
                for choice in choices:
                    if choice.lower() in ("test", "test task (warning!)"):
                        return True

    return False


def filter_training_tasks(tasks: List, exclude_test: bool = True) -> List:
    """
    Filter a list of tasks to exclude test tasks.

    Args:
            tasks: List of Label Studio tasks
            exclude_test: If True (default), exclude tasks marked as test

    Returns:
            Filtered list of tasks suitable for training
    """
    if not exclude_test:
        return tasks

    training_tasks = []
    test_count = 0

    for task in tasks:
        if is_test_task(task):
            test_count += 1
        else:
            training_tasks.append(task)

    if test_count > 0:
        logger.info(
            f"Filtered out {test_count} test tasks, {len(training_tasks)} tasks remaining for training"
        )

    return training_tasks


def get_test_tasks(tasks: List) -> List:
    """
    Get only the test tasks from a list of tasks.

    Args:
            tasks: List of Label Studio tasks

    Returns:
            List of tasks marked as test (for evaluation purposes)
    """
    test_tasks = []

    for task in tasks:
        if is_test_task(task):
            test_tasks.append(task)

    if test_tasks:
        logger.info(f"Found {len(test_tasks)} test tasks out of {len(tasks)} total")
    else:
        logger.info(f"No test tasks found in {len(tasks)} tasks")

    return test_tasks


def partition_tasks_by_purpose(tasks: List) -> Dict[str, List]:
    """
    Partition tasks into training and test sets based on task_purpose annotation.

    Args:
            tasks: List of Label Studio tasks

    Returns:
            Dict with 'training' and 'test' keys, each containing a list of tasks
    """
    training_tasks = []
    test_tasks = []

    for task in tasks:
        if is_test_task(task):
            test_tasks.append(task)
        else:
            training_tasks.append(task)

    logger.info(
        f"Partitioned {len(tasks)} tasks: {len(training_tasks)} training, {len(test_tasks)} test"
    )

    return {"training": training_tasks, "test": test_tasks}


# ─── Generic Task Listing ─────────────────────────────────────────────────────


def list_tasks_by_project(
    project_id: int | str,
    ls_client=None,
    return_summary=False,
    only_annotated=False,
    exclude_test_tasks=False,
) -> List[Dict[str, Any]]:
    """
    Fetch all tasks from Label Studio for a given project.

    Args:
            project_id: The Label Studio project ID
            ls_client: Optional Label Studio client instance
            return_summary: If True, return minimal summary info instead of full task objects
            only_annotated: If True, only return tasks with at least one annotation (much faster for training)
            exclude_test_tasks: If True, exclude tasks marked with task_purpose='test' (for training)

    Returns a list of JSON-serializable dicts.
    """
    # Lazy import of LS client if not provided
    if ls_client is None:
        from ml_backend.ls_client import get_ls_client

        ls_client = get_ls_client()

    # Build query to filter for annotated tasks only
    query_str = None
    if only_annotated:
        query = {
            "filters": {
                "conjunction": "and",
                "items": [
                    {
                        "filter": "filter:tasks:total_annotations",
                        "operator": "greater",
                        "type": "Number",
                        "value": "0",
                    }
                ],
            }
        }
        query_str = json.dumps(query)

    tasks = list(ls_client.tasks.list(project=project_id, query=query_str))

    # Filter out test tasks if requested
    if exclude_test_tasks:
        tasks = filter_training_tasks(tasks, exclude_test=True)

    result: List[Dict[str, Any]] = []
    for t in tasks:
        data = t.data if hasattr(t, "data") else {}
        if return_summary:
            result.append(
                {
                    "id": getattr(t, "id", None),
                    "audio": data.get("audio"),
                    "recorded_date": data.get("recorded_date"),
                    "recorded_time": data.get("recorded_time"),
                }
            )
        else:
            result.append(t)
    return result


# ─── URL/Path Conversion ──────────────────────────────────────────────────────


def ls_url_to_local_path(ls_url):
    """
    Convert Label Studio URL to local file path.
    Handle different URL formats that Label Studio might send.
    """
    if "?d=" in ls_url:
        # Format: "/data/local-files/?d=path/to/file.wav"
        real_path = ls_url.split("?d=")[-1]
    elif ls_url.startswith("/data/"):
        # Already a path, might need adjustment
        real_path = ls_url
    else:
        # Direct path
        real_path = ls_url

    # Ensure absolute path for Label Studio context
    if not real_path.startswith("/"):
        real_path = "/" + real_path

    logger.debug(f"Converted URL '{ls_url}' to path '{real_path}'")
    return real_path


def parse_ls_task_annotations(task, verbose=False) -> List[Dict]:
    """Parse annotations from a Label Studio task object."""
    annotations = []
    if not hasattr(task, "annotations"):
        logger.warning(f"Task {task.id} has no annotations")
        return []

    for ann in task.annotations:
        # Handle dict or object access
        if isinstance(ann, dict):
            results = ann.get("result")
            ann_id = ann.get("id", "unknown")
        else:
            results = getattr(ann, "result", None)
            ann_id = getattr(ann, "id", "unknown")

        if results is None:
            logger.warning(f"Annotation {ann_id} in task {task.id} has no result")
            continue

        for res in results:
            if res.get("type") == "labels":
                val = res.get("value", {})
                labels = val.get("labels", [])
                # logger.info(f"Annotation {ann_id} has labels: {labels}")
                if labels and "start" in val and "end" in val:
                    annotations.append(
                        {
                            "start": float(val["start"]),
                            "end": float(val["end"]),
                            "label": labels[0],  # Assume single label
                        }
                    )

    if annotations and verbose:
        found_labels = [a["label"] for a in annotations]
        logger.debug(f"Task {task.id} has annotations: {set(found_labels)}")
    else:
        logger.debug(f"Task {task.id} has no annotations")

    return annotations
