"""
SmartStable-specific task filtering and metadata extraction utilities.

This module contains functions that depend on the SmartStable filename format
and domain-specific concepts like stable_id, stall_id, horse_id, etc.

Filename format: stable01_stall06_horse06_20250807_190013_mic.flac
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

logger = logging.getLogger("SmartStableTaskFiltering")


# ─── Filename Parsing Utilities ───────────────────────────────────────────────


def extract_raw_data_from_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Extract raw metadata from SmartStable filename without applying mappings.

    Expected filename format: stable01_stall06_horse06_20250807_190013_mic.flac

    Returns:
        Dict with stable_raw, stall_raw, horse_raw, recorded_date, recorded_time
        or None if filename doesn't match expected format.
    """
    # Clean the filename
    name = filename.split("/")[-1]  # Remove path if present
    parts = (
        name.replace(".flac", "")
        .replace(".mp4", "")
        .replace("_mic", "")
        .replace("_cam", "")
        .split("_")
    )

    if len(parts) < 5:
        logger.warning(
            f"Filename {filename} does not contain enough parts (expected 5+, got {len(parts)})"
        )
        return None

    return {
        "stable_raw": parts[0],  # stable01
        "stall_raw": parts[1],  # stall06
        "horse_raw": parts[2],  # horse06
        "recorded_date": parts[3],  # 20250807
        "recorded_time": parts[4],  # 190013
    }


def format_date(date_str: str) -> str:
    """Format YYYYMMDD to YYYY-MM-DD."""
    if len(date_str) == 8:
        return f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return date_str


def format_time(time_str: str) -> str:
    """Format HHMMSS to HH:MM:SS."""
    if len(time_str) == 6:
        return f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
    return time_str


def _extract_time_from_filename(filename: str) -> Optional[str]:
    """Extract time from SmartStable filename format."""
    try:
        name = filename.split("/")[-1]
        parts = name.replace(".flac", "").replace("_mic", "").split("_")
        if len(parts) > 4:
            t = parts[4]
            if len(t) == 6:
                return f"{t[0:2]}:{t[2:4]}:{t[4:6]}"
            return t
    except Exception:
        pass
    return None


def extract_and_map_metadata_from_filename(
    filename: str, config: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, str]]:
    """
    Extract metadata from SmartStable filename and apply human-readable mappings.

    Args:
        filename: The audio filename (e.g., stable01_stall06_horse06_20250807_190013_mic.flac)
        config: Optional config dict with stable_mappings and stable_configs.
                If not provided, raw values are used.

    Returns:
        Dict with filename, stable, stall, horse, recorded_date, recorded_time, recorded_date_time
        or None if parsing fails.
    """
    raw_data = extract_raw_data_from_filename(filename)
    if not raw_data:
        return None

    # Start with raw values
    stable = raw_data["stable_raw"]
    stall = raw_data["stall_raw"]
    horse = raw_data["horse_raw"]

    # Apply mappings if config is provided
    if config:
        stable = config.get("stable_mappings", {}).get(raw_data["stable_raw"], stable)

        stable_config = config.get("stable_configs", {}).get(raw_data["stable_raw"], {})
        stall = stable_config.get("stall_mappings", {}).get(
            raw_data["stall_raw"], stall
        )
        horse = stable_config.get("horse_mappings", {}).get(
            raw_data["horse_raw"], horse
        )

    # Format date and time
    formatted_date = format_date(raw_data["recorded_date"])
    formatted_time = format_time(raw_data["recorded_time"])

    return {
        "filename": filename.split("/")[-1],
        "stable": stable,
        "stall": stall,
        "horse": horse,
        "recorded_date": formatted_date,
        "recorded_time": formatted_time,
        "recorded_date_time": f"{formatted_date} {formatted_time}",
    }


def _parse_time(t: Optional[str]):
    """Parse time string in various formats."""
    if not t:
        return None
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(t, fmt).time()
        except Exception:
            continue
    return None


def check_if_task_in_time_window(
    metadata: Dict[str, str],
    time_window: tuple[str, str],
    date_window: tuple[str, str],
    stable_name: str = "none",
) -> bool:
    """
    Check if a SmartStable task's metadata falls within specified time and date windows.

    Args:
        metadata: Dict with 'stable', 'recorded_date' (YYYY-MM-DD), 'recorded_time' (HH:MM:SS)
        time_window: (time_from, time_until) in HH:MM format
        date_window: (date_from, date_until) in YYYY-MM-DD format
        stable_name: If not "none", filter by this stable name

    Returns:
        True if task is within the window, False otherwise.
    """
    # Check stable filter
    if stable_name != "none" and metadata.get("stable") != stable_name:
        return False

    try:
        # Parse dates
        date_from_obj = datetime.strptime(date_window[0], "%Y-%m-%d").date()
        date_until_obj = datetime.strptime(date_window[1], "%Y-%m-%d").date()
        recorded_date_obj = datetime.strptime(
            metadata["recorded_date"], "%Y-%m-%d"
        ).date()

        # Parse times
        time_from_obj = datetime.strptime(time_window[0], "%H:%M").time()
        time_until_obj = datetime.strptime(time_window[1], "%H:%M").time()

        recorded_time = metadata.get("recorded_time", "")
        recorded_time_obj = _parse_time(recorded_time)
        if not recorded_time_obj:
            logger.warning(f"Invalid time format: {recorded_time}")
            return False

        # Check date range
        if recorded_date_obj < date_from_obj or recorded_date_obj > date_until_obj:
            return False

        # Check time range (only boundary days are time-constrained)
        if recorded_date_obj == date_from_obj and recorded_time_obj < time_from_obj:
            return False
        if recorded_date_obj == date_until_obj and recorded_time_obj > time_until_obj:
            return False

        return True

    except Exception as e:
        logger.warning(f"Error checking time window: {e}")
        return False


# ─── SmartStable-Specific Task Listing ────────────────────────────────────────


def list_tasks_by_recording_period(
    project_id: int | str,
    stable_id: Optional[str],
    date_from: str,
    date_until: str,
    stall_id: Optional[str] = None,
    time_from: Optional[str] = None,
    time_until: Optional[str] = None,
    only_annotated: bool = False,
    exclude_test_tasks: bool = False,
    return_summary: bool = True,
    max_days: int = 3600,
    max_tasks: int = 50000,
    ls_client=None,
) -> List[Dict[str, Any]]:
    """
    Fetch SmartStable tasks from Label Studio filtered by recording date range
    encoded in the filename (YYYYMMDD). Optionally filter by stable_id, stall_id,
    and by time-of-day on the client side.

    This function is SmartStable-specific because it:
    - Expects filenames in format: stable01_stall06_horse06_20250807_190013_mic.flac
    - Filters by stable_id and stall_id which are domain concepts
    - Parses dates/times from the SmartStable filename format

    Args:
        project_id: Label Studio project ID
        stable_id: Filter by stable (e.g., 'stable01')
        date_from: Start date in YYYY-MM-DD format
        date_until: End date in YYYY-MM-DD format
        stall_id: Optional stall filter (e.g., 'stall06')
        time_from: Optional time filter start (HH:MM or HH:MM:SS)
        time_until: Optional time filter end (HH:MM or HH:MM:SS)
        only_annotated: If True, only return annotated tasks
        exclude_test_tasks: If True, exclude tasks marked with task_purpose='test'
        return_summary: If True, return minimal summary instead of full task objects
        max_days: Maximum allowed date range
        max_tasks: Maximum tasks to return
        ls_client: Optional Label Studio client

    Returns:
        List of task dicts: {id, audio, recorded_date, recorded_time} or full task objects
    """
    # Import here to avoid circular imports
    from ml_backend.util.label_studio_helper import filter_training_tasks

    # Lazy import of LS client if not provided
    if ls_client is None:
        from ml_backend.ls_client import get_ls_client

        ls_client = get_ls_client()

    try:
        df = datetime.strptime(date_from, "%Y-%m-%d").date()
        du = datetime.strptime(date_until, "%Y-%m-%d").date()
    except Exception:
        raise ValueError("date_from/date_until must be in format YYYY-MM-DD")

    days = (du - df).days + 1
    if days <= 0:
        return []
    if days > max_days:
        raise ValueError(f"Date range too large ({days} days > {max_days})")

    # Build date regex for SmartStable filename format (YYYYMMDD)
    tokens = [(df + timedelta(days=i)).strftime("%Y%m%d") for i in range(days)]
    regex_pattern = "|".join(tokens)

    # Build Label Studio query filters
    items = []
    if stable_id:
        items.append(
            {
                "filter": "filter:tasks:data__audio",
                "operator": "contains",
                "type": "String",
                "value": f"{stable_id}_",
            }
        )
    if stall_id:
        # Match whole token surrounded by underscores to avoid partials
        items.append(
            {
                "filter": "filter:tasks:data__audio",
                "operator": "contains",
                "type": "String",
                "value": f"_{stall_id}_",
            }
        )
    if only_annotated:
        items.append(
            {
                "filter": "filter:tasks:total_annotations",
                "operator": "greater",
                "type": "Number",
                "value": "0",
            }
        )
    items.append(
        {
            "filter": "filter:tasks:data__audio",
            "operator": "regex",
            "type": "String",
            "value": f"({regex_pattern})",
        }
    )

    query = {"filters": {"conjunction": "and", "items": items}}
    query_str = json.dumps(query)

    logger.info(f"Label Studio query: {query_str}")

    tasks = list(ls_client.tasks.list(project=project_id, query=query_str))

    # Optional client-side time filtering (SmartStable specific)
    tf = _parse_time(time_from) if time_from else None
    tu = _parse_time(time_until) if time_until else None
    wraps_midnight = False
    df_date = df
    du_date = du

    if tf and tu:
        filtered = []
        wraps_midnight = tf > tu  # e.g. 18:30 -> 09:30 next day
        for t in tasks:
            data = t.data if hasattr(t, "data") else {}
            rec_time = data.get("recorded_time")
            if not rec_time:
                audio_url = data.get("audio", "")
                rec_time = _extract_time_from_filename(audio_url or "")
            rt = _parse_time(rec_time) if rec_time else None
            rec_date_str = data.get("recorded_date")
            try:
                rec_date = (
                    datetime.strptime(rec_date_str, "%Y-%m-%d").date()
                    if rec_date_str
                    else None
                )
            except Exception:
                rec_date = None
            if not (rt and rec_date):
                continue

            if not wraps_midnight:
                if tf <= rt <= tu:
                    filtered.append(t)
            else:
                # Wrap-around across midnight over potentially multiple days
                if rec_date == df_date:
                    if rt >= tf:
                        filtered.append(t)
                elif rec_date == du_date:
                    if rt <= tu:
                        filtered.append(t)
                else:
                    # Intermediate date
                    if (rt <= tu) or (rt >= tf):
                        filtered.append(t)
        tasks = filtered

    if len(tasks) > max_tasks:
        tasks = tasks[:max_tasks]

    # Filter out test tasks if requested
    if exclude_test_tasks:
        tasks = filter_training_tasks(tasks, exclude_test=True)

    # Return minimal summaries or full objects
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

    if wraps_midnight and (du_date - df_date).days > 1:
        by_date = {}
        for r in result:
            if isinstance(r, dict):
                d = r.get("recorded_date")
            else:
                raw_data = r.data if hasattr(r, "data") else {}
                d = raw_data.get("recorded_date") if isinstance(raw_data, dict) else None
            if d:
                by_date.setdefault(d, 0)
                by_date[d] += 1
        logger.debug("[time_filter] wrap window summary: %s", by_date)

    # Collect unique stables and stalls for logging
    found_stables = set()
    found_stalls = set()
    for t in tasks:
        d = t.data if hasattr(t, "data") else {}
        s = d.get("stable") or d.get("stable_id")
        st = d.get("stall") or d.get("stall_id")

        if (s is None or st is None) and d.get("audio"):
            # Fallback to extraction from filename
            raw = extract_raw_data_from_filename(d["audio"])
            if raw:
                if s is None:
                    s = raw.get("stable_raw")
                if st is None:
                    st = raw.get("stall_raw")
        found_stables.add(s)
        found_stalls.add(st)

    logger.info(f"[time_filter] Found {len(result)} tasks")
    logger.info(f"[time_filter] different stables: {found_stables}")
    logger.info(f"[time_filter] different stalls: {found_stalls}")
    return result
