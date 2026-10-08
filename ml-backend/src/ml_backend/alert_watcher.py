"""
Background watcher that turns incoming edge-device alert clips into Label Studio
tasks.

Alert clips are pushed from the Raspberry Pis (see edge-monitor
scripts/push_alert_file.sh) into the shared audio volume under:

    <ALERTS_DIR>/<stable>/<stall>/<YYYYMMDD>_<HHMMSS>_<TYPE>_<label>.wav

e.g.  /label-studio/data/audio/alerts/stable03/pi138/20260630_065154_ALERT_scream.wav

The watcher runs inside the ml-backend process (started from the FastAPI startup
hook). It reuses the existing LS client, the task_config.toml stable/stall name
mappings, and the same task_data shape as the /tasks/add-single endpoint, so an
alert task looks like any other recording task.

Design notes
------------
* Decoupled from how files arrive: it watches the filesystem, so rsync/scp/manual
  copies all work.
* Fast restarts: processed clips are remembered in a small local state file
  (ALERT_WATCHER_STATE_FILE), so a normal restart does NO Label Studio queries —
  it just skips clips already in that set. Only genuinely new clips do any work.
* Idempotent without a full task listing: for a clip not in the local state, a
  single targeted Data Manager query ("is there a task whose data.audio == this
  URL?") guards against duplicates even if the state file is lost.
* Arrival-order safe: dedup is by clip identity, not file mtime — important
  because rsync -a preserves the original recording time, so late-pushed older
  clips would defeat a timestamp watermark.
* Partial-write safe: rsync renames into place atomically; we additionally wait
  for the file size to stabilise before creating a task.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger("AlertWatcher")

# Container path where the shared audio volume mounts the alert tree.
DEFAULT_ALERTS_DIR = "/label-studio/data/audio/alerts"
# `d=` prefix Label Studio uses to serve the alert tree (matches the working
# audio tasks created elsewhere: /data/local-files/?d=label-studio/data/audio/...).
DEFAULT_URL_PREFIX = "label-studio/data/audio/alerts"

# Recognised warning types embedded in the filename by the edge recorder.
KNOWN_WARNING_TYPES = {"ALERT", "CLUSTER", "LOUDNESS"}

# Local state file remembering which clips already have tasks (kept out of the
# audio volume that Label Studio serves). Defaults under the metamodel/log dir.
DEFAULT_STATE_FILE = os.path.join(
    os.getenv("SMARTSTABLE_METAMODEL_DIR", "/smart-stable-logs"),
    "alert_watcher_state.json",
)


def _enabled() -> bool:
    return os.getenv("ALERT_WATCHER_ENABLED", "true").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _fmt_date(raw: str) -> str:
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}" if len(raw) == 8 and raw.isdigit() else "unknown"


def _fmt_time(raw: str) -> str:
    return f"{raw[0:2]}:{raw[2:4]}:{raw[4:6]}" if len(raw) == 6 and raw.isdigit() else "unknown"


def parse_clip(path: Path, alerts_root: Path) -> dict:
    """
    Derive task metadata from an alert clip path. stable/stall come from the two
    directory levels under the alerts root; date/time/type/label from the filename
    (<date>_<time>_<TYPE>_<label>.wav). Unknown pieces fall back to "unknown"/"".
    """
    rel = path.relative_to(alerts_root)
    parts = rel.parts
    if len(parts) >= 3:
        stable_id, stall_id = parts[0], parts[1]
    elif len(parts) == 2:
        stable_id, stall_id = parts[0], ""
    else:
        stable_id, stall_id = "", ""

    tokens = path.stem.split("_")
    date_raw = tokens[0] if len(tokens) > 0 else ""
    time_raw = tokens[1] if len(tokens) > 1 else ""
    warning_type = tokens[2] if len(tokens) > 2 and tokens[2] in KNOWN_WARNING_TYPES else ""
    label = "_".join(tokens[3:]) if len(tokens) > 3 else ""

    return {
        "stable_id": stable_id,
        "stall_id": stall_id,
        "filename": path.name,
        "recorded_date": _fmt_date(date_raw),
        "recorded_time": _fmt_time(time_raw),
        "warning_type": warning_type,
        "label": label,
    }


def build_task_data(meta: dict, url_prefix: str, config: dict) -> dict:
    """Build the LS task data dict, mirroring the /tasks/add-single shape."""
    stable_id, stall_id = meta["stable_id"], meta["stall_id"]

    stable_name = config.get("stable_mappings", {}).get(stable_id, stable_id)
    stall_name = (
        config.get("stable_configs", {})
        .get(stable_id, {})
        .get("stall_mappings", {})
        .get(stall_id, stall_id)
    )

    # Build the served path, skipping an empty stall segment.
    segments = [url_prefix, stable_id] + ([stall_id] if stall_id else []) + [meta["filename"]]
    audio_url = "/data/local-files/?d=" + "/".join(s.strip("/") for s in segments if s)

    return {
        "audio": audio_url,
        "video": "",
        "stable": stable_name,
        "stall": stall_name,
        "horse": "unknown",
        "recorded_date": meta["recorded_date"],
        "recorded_time": meta["recorded_time"],
        "recorded_date_time": f"{meta['recorded_date']} {meta['recorded_time']}",
        "report": "",
        "source": "alert",
        "known_event": meta["label"] or meta["warning_type"].lower() or "alert",
        "alert_type": meta["warning_type"],
        "manual_notes": "",
        "weather": "unknown",
        "events": "",
        "grid_video": "",
        "diet_type": "",
        "protocol_step": "",
    }


def _load_processed(state_file: Path) -> Set[str]:
    """Load the set of already-tasked clip keys (relative paths). Missing/corrupt → empty."""
    try:
        with open(state_file) as f:
            data = json.load(f)
        return set(data.get("processed", []))
    except FileNotFoundError:
        return set()
    except Exception as e:
        logger.warning("Could not read alert watcher state %s: %s", state_file, e)
        return set()


def _save_processed(state_file: Path, processed: Set[str]) -> None:
    """Persist the processed-clip set atomically. Failures are non-fatal."""
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({"processed": sorted(processed)}, f)
        os.replace(tmp, state_file)
    except Exception as e:
        logger.warning("Could not persist alert watcher state %s: %s", state_file, e)


def _task_exists(ls, project_id: int, audio_url: str) -> bool:
    """
    Targeted dedup: ask Label Studio whether a task in this project already has
    data.audio == audio_url, via a single Data Manager filter query (no full list).
    We re-check the returned task's audio so a misinterpreted filter can't cause a
    false positive that would suppress a real task.
    """
    query = json.dumps(
        {
            "filters": {
                "conjunction": "and",
                "items": [
                    {
                        "filter": "filter:tasks:data.audio",
                        "operator": "equal",
                        "type": "String",
                        "value": audio_url,
                    }
                ],
            }
        }
    )
    try:
        results = ls.tasks.list(project=project_id, query=query, page_size=10)
        for task in results:
            data = getattr(task, "data", None) or {}
            if data.get("audio") == audio_url:
                return True
            break  # only need to inspect the first page's first hit
    except Exception as e:
        # On query failure, fall through to creation; the local state set still
        # prevents duplicates within and across normal runs.
        logger.warning("Dedup query failed for %s: %s", audio_url, e)
    return False


async def _wait_until_stable(path: Path, tries: int = 5, interval: float = 1.0) -> bool:
    """Return True once the file size stops changing (guards against partial writes)."""
    last = -1
    for _ in range(tries):
        try:
            size = path.stat().st_size
        except OSError:
            return False
        if size > 0 and size == last:
            return True
        last = size
        await asyncio.sleep(interval)
    return last > 0


async def _create_task(
    path: Path,
    alerts_root: Path,
    url_prefix: str,
    ls,
    project_id: int,
    processed: Set[str],
    state_file: Path,
) -> None:
    key = path.relative_to(alerts_root).as_posix()
    if key in processed:
        return  # fast path: already tasked — no import, config load, or LS call

    from ml_backend.routes.smartstable.tasks import load_config

    meta = parse_clip(path, alerts_root)
    task_data = build_task_data(meta, url_prefix, load_config(suppress_log=True))
    audio_url = task_data["audio"]

    # Not in local state — confirm against LS with a single targeted query so a
    # lost/rebuilt state file can't create duplicates.
    if _task_exists(ls, project_id, audio_url):
        processed.add(key)
        _save_processed(state_file, processed)
        return

    if not await _wait_until_stable(path):
        logger.warning("Skipping unstable/empty file: %s", path)
        return

    # Reserve before the network call so a burst can't double-create; release on
    # failure so a later scan/event can retry.
    processed.add(key)
    try:
        created = ls.tasks.create(project=project_id, data=task_data)
        _save_processed(state_file, processed)
        logger.info(
            "Created alert task %s in project %s for %s (stable=%s stall=%s)",
            getattr(created, "id", "?"),
            project_id,
            path.name,
            task_data["stable"],
            task_data["stall"],
        )
    except Exception as e:
        processed.discard(key)
        logger.error("Failed to create alert task for %s: %s", path, e, exc_info=True)


async def run_alert_watcher(alerts_dir: Optional[str] = None) -> None:
    """
    Long-running task: scan for existing clips, then watch for new ones and create
    a Label Studio task per clip. Safe to launch fire-and-forget from startup.
    """
    if not _enabled():
        logger.info("Alert watcher disabled (ALERT_WATCHER_ENABLED).")
        return

    try:
        from watchfiles import Change, awatch
    except ImportError:
        logger.error("watchfiles not installed — alert watcher cannot start.")
        return

    from ml_backend.ls_client import get_ls_client
    from ml_backend.routes.smartstable.tasks import get_or_default_project

    alerts_root = Path(alerts_dir or os.getenv("ALERTS_DIR", DEFAULT_ALERTS_DIR))
    url_prefix = os.getenv("ALERT_AUDIO_URL_PREFIX", DEFAULT_URL_PREFIX)
    alerts_root.mkdir(parents=True, exist_ok=True)

    ls = get_ls_client()
    project_env = os.getenv("ALERT_PROJECT_ID")
    try:
        if project_env:
            project_id = int(project_env)
        else:
            project_id = get_or_default_project(ls)
            logger.warning(
                "ALERT_PROJECT_ID not set — defaulting to the first project (id=%s). "
                "If alert tasks show up in the wrong project, set ALERT_PROJECT_ID "
                "to the intended project id.",
                project_id,
            )
    except Exception as e:
        logger.error("Alert watcher could not resolve a project: %s", e)
        return

    state_file = Path(os.getenv("ALERT_WATCHER_STATE_FILE", DEFAULT_STATE_FILE))
    processed = _load_processed(state_file)

    # Catch-up: clips already in the local state are skipped without any LS call;
    # only clips new since the last run trigger a targeted dedup query + create.
    existing = sorted(p for p in alerts_root.rglob("*.wav") if p.is_file())
    new_count = sum(1 for p in existing if p.relative_to(alerts_root).as_posix() not in processed)
    logger.info(
        "Alert watcher: %d clip(s) on disk, %d already processed, %d to check (state=%s)",
        len(existing), len(processed), new_count, state_file,
    )
    for p in existing:
        await _create_task(p, alerts_root, url_prefix, ls, project_id, processed, state_file)

    logger.info("Alert watcher now watching %s (project %s)", alerts_root, project_id)
    async for changes in awatch(alerts_root):
        for change, raw in changes:
            if change == Change.deleted:
                continue
            path = Path(raw)
            if path.suffix.lower() == ".wav":
                await _create_task(path, alerts_root, url_prefix, ls, project_id, processed, state_file)
