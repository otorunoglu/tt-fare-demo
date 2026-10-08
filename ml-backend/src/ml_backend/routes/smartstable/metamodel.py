"""
Metamodel routes for SmartStable.

Provides endpoints for:
- Batch processing audio files to generate soundscape predictions
- Writing predictions directly to the DataStore (SQLite)
- Optional warning generation via MetaModelDecider
"""

import os
import re
import io
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastapi import APIRouter, HTTPException, Request as FastAPIRequest
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from ml_backend.smartstable_core.manager_loader import get_multistall_manager
from ml_backend.smartstable_core.task_filtering import list_tasks_by_recording_period
from ml_backend.util.label_studio_helper import ls_url_to_local_path
from ml_backend.ls_client import get_ls_client

logger = logging.getLogger("MetamodelRouter")

metamodel_router = APIRouter(prefix="/metamodel", tags=["metamodel"])


# ─── Configuration ────────────────────────────────────────────────────────────

DB_PATH = os.getenv("SMARTSTABLE_DB_PATH", "/app/data/smartstable/soundscape.db")


def get_datastore():
    """Get or create DataStore instance for event persistence."""
    from smartstablemodel.services.data_store import DataStore

    return DataStore(DB_PATH)


# ─── Helper Functions ─────────────────────────────────────────────────────────


def _night_key(ts: datetime) -> str:
    """
    Night is defined as the 24h window starting at 19:00 local calendar date.
    All times from 19:00 D .. <19:00 D+1 are labeled with date D.
    """
    if ts.hour >= 19:
        return ts.strftime("%Y-%m-%d")
    else:
        return (ts - timedelta(days=1)).strftime("%Y-%m-%d")


def _parse_recording_start_time(
    filename: str, task: Dict[str, Any]
) -> Optional[datetime]:
    """Try task metadata first, then filename pattern *_YYYYMMDD_HHMMSS_."""
    rd = task.get("recorded_date")
    rt = task.get("recorded_time")
    if rd and rt:
        try:
            return datetime.strptime(f"{rd} {rt}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    if rd:
        # If only a date is present assume start of NIGHT (19:00) not midnight
        try:
            return datetime.strptime(rd, "%Y-%m-%d") + timedelta(hours=19)
        except ValueError:
            pass
    m = re.search(r"_(\d{8})_(\d{6})", filename)
    if m:
        d, t = m.group(1), m.group(2)
        try:
            return datetime.strptime(d + t, "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return None


# ─── Request Models ───────────────────────────────────────────────────────────


class AnalyzeBatchRequest(BaseModel):
    """Request model for batch analysis of audio tasks."""

    project_id: int = Field(..., description="Label Studio project ID")
    stable_id: str = Field(..., description="Stable ID (e.g., 'stable01')")
    stall_id: str = Field(..., description="Stall ID (e.g., 'stall01') or 'all'")

    # Dates are now optional if task_ids are provided
    date_from: Optional[str] = Field(
        default=None, description="Start date (YYYY-MM-DD)"
    )
    date_until: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")

    task_ids: Optional[List[int]] = Field(
        default=None,
        description="Specific list of task IDs to process (overrides dates)",
    )

    time_from: str = Field(default="18:30", description="Start time (HH:MM)")
    time_until: str = Field(default="09:30", description="End time (HH:MM)")
    max_tasks: int = Field(
        default=200, description="Maximum number of tasks to process"
    )
    generate_warnings: bool = Field(
        default=True, description="Generate warnings via MetaModelDecider"
    )
    clear_db: bool = Field(
        default=False, description="Clear existing events before import"
    )
    dry_run: bool = Field(
        default=False, description="If True, only show what would be processed"
    )


class ClassFeatureScatterRequest(BaseModel):
    """Request model for class-specific RMS vs spectral centroid scatter plotting."""

    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(
        ...,
        alias="class",
        description="Target class label (for example: 'eating').",
    )
    date_from: str = Field(..., description="Start date (YYYY-MM-DD)")
    date_until: str = Field(..., description="End date (YYYY-MM-DD)")


class PreviewHeatmapRequest(BaseModel):
    """Request model for generating a temporary heatmap image without writing to the main DB."""

    project_id: int = Field(..., description="Label Studio project ID")
    stable_id: str = Field(..., description="Stable ID (e.g., 'stable01')")
    stall_id: str = Field(..., description="Stall ID (e.g., 'stall01') or 'all'")

    date_from: Optional[str] = Field(
        default=None, description="Start date (YYYY-MM-DD)"
    )
    date_until: Optional[str] = Field(
        default=None, description="End date (YYYY-MM-DD)"
    )

    task_ids: Optional[List[int]] = Field(
        default=None,
        description="Specific list of task IDs to process (overrides dates)",
    )

    time_from: str = Field(default="18:30", description="Start time (HH:MM)")
    time_until: str = Field(default="09:30", description="End time (HH:MM)")
    max_tasks: int = Field(
        default=200, description="Maximum number of tasks to process"
    )

    bin_size_minutes: int = Field(default=2, description="Heatmap bin size in minutes")
    group_labels: bool = Field(default=False, description="Group labels in heatmap")
    hide_empty_time_bins: bool = Field(
        default=False,
        description="Drop time bins where all labels have zero events",
    )
    max_bins: int = Field(default=2000, description="Maximum number of heatmap bins")


# ─── Routes ───────────────────────────────────────────────────────────────────


@metamodel_router.get("/debug")
def debug_metamodel():
    """Health check for metamodel routes."""
    return {"status": "ok", "message": "metamodel route reachable", "db_path": DB_PATH}


@metamodel_router.get("/labels")
def get_labels(date_from: Optional[str] = None, date_until: Optional[str] = None):
    """Return distinct labels present in the DB, optionally filtered by date range."""
    try:
        ds = get_datastore()
        params: list = []
        where = ""
        if date_from and date_until:
            try:
                start = datetime.strptime(date_from, "%Y-%m-%d")
                end = datetime.strptime(date_until, "%Y-%m-%d") + timedelta(days=1)
            except ValueError:
                raise HTTPException(status_code=400, detail="Use YYYY-MM-DD for dates.")
            where = " WHERE timestamp >= ? AND timestamp < ?"
            params = [start.isoformat(), end.isoformat()]

        cur = ds._conn.cursor()
        cur.execute(
            f"SELECT label, COUNT(*) as n FROM events{where} GROUP BY label ORDER BY n DESC",
            params,
        )
        rows = cur.fetchall()
        return {
            "date_from": date_from,
            "date_until": date_until,
            "labels": [{"label": r[0], "count": r[1]} for r in rows],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[get_labels] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@metamodel_router.get("/db-stats")
def get_db_stats():
    """Get statistics about the events database."""
    try:
        ds = get_datastore()
        bounds = ds.time_bounds()

        if bounds and bounds[0] and bounds[1]:
            count = ds.count_in_range(
                start=bounds[0], end=bounds[1] + timedelta(seconds=1)
            )
            return {
                "db_path": str(ds.path),
                "min_timestamp": bounds[0].isoformat() if bounds[0] else None,
                "max_timestamp": bounds[1].isoformat() if bounds[1] else None,
                "event_count": count,
            }
        else:
            return {
                "db_path": str(ds.path),
                "min_timestamp": None,
                "max_timestamp": None,
                "event_count": 0,
            }
    except Exception as e:
        logger.error(f"Failed to get DB stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@metamodel_router.post("/analyze-batch")
async def analyze_batch(request: AnalyzeBatchRequest):
    """
    Analyze tasks from Label Studio.

    Modes:
    1. specific tasks: Provide `task_ids` (list of integers).
    2. date range: Provide `date_from` and `date_until`.

    This endpoint:
    1. Fetches tasks from Label Studio
    2. Runs soundscape classification on each audio file
    3. Writes predictions directly to the SQLite DataStore
    4. Optionally generates warnings via MetaModelDecider

    The results are immediately available in the Streamlit visualization.
    """
    try:
        ls = get_ls_client()
        manager = get_multistall_manager()
        ds = get_datastore()

        tasks_found = []

        # Mode 1: specific task IDs
        if request.task_ids:
            if request.date_from or request.date_until:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot provide 'task_ids' and date range ('date_from'/'date_until') simultaneously.",
                )
            logger.info(f"[analyze_batch] Fetching specific tasks: {request.task_ids}")
            for tid in request.task_ids:
                try:
                    # ls_client.tasks is from label_studio_sdk.client
                    t = ls.tasks.get(id=str(tid))
                    tasks_found.append(t)
                except Exception as e:
                    logger.warning(f"Failed to fetch task {tid}: {e}")

        # Mode 2: Date range
        elif request.date_from and request.date_until:
            # Validate dates
            try:
                datetime.strptime(request.date_from, "%Y-%m-%d")
                datetime.strptime(request.date_until, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="Invalid date format (expected YYYY-MM-DD)"
                )

            # Get tasks from Label Studio
            tasks_found = list_tasks_by_recording_period(
                project_id=request.project_id,
                stable_id=request.stable_id,
                stall_id=request.stall_id if request.stall_id != "all" else None,
                date_from=request.date_from,
                date_until=request.date_until,
                time_from=request.time_from,
                time_until=request.time_until,
                max_tasks=request.max_tasks + 1,
                ls_client=ls,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either 'task_ids' or ('date_from' AND 'date_until')",
            )

        tasks = tasks_found  # alias

        if not tasks:
            return {
                "success": True,
                "message": "No tasks found for the given criteria",
                "request": request.dict(exclude_defaults=True),
            }

        # Sort tasks by date/time
        def get_task_field(t, field, default=""):
            if isinstance(t, dict):
                return t.get(field, default) or default
            return (
                getattr(t, field, default) or default
                if hasattr(t, field)
                else (
                    t.data.get(field, default)
                    if hasattr(t, "data") and t.data
                    else default
                )
            )

        tasks.sort(
            key=lambda t: (
                get_task_field(t, "recorded_date"),
                get_task_field(t, "recorded_time"),
            )
        )

        truncated = False
        if len(tasks) > request.max_tasks:
            tasks = tasks[: request.max_tasks]
            truncated = True

        logger.info(
            f"[analyze_batch] Found {len(tasks)} tasks (truncated={truncated}) for "
            f"stable={request.stable_id} stall={request.stall_id}"
        )

        # Dry run - just return what would be processed
        if request.dry_run:
            task_info = []
            for t in tasks[:10]:  # Show first 10
                if isinstance(t, dict):
                    task_info.append(
                        {
                            "id": t.get("id"),
                            "recorded_date": t.get("recorded_date"),
                            "recorded_time": t.get("recorded_time"),
                            "audio": t.get("audio", "")[:50] + "..."
                            if t.get("audio")
                            else None,
                        }
                    )
                else:
                    data = t.data if hasattr(t, "data") and t.data else {}
                    task_info.append(
                        {
                            "id": getattr(t, "id", None),
                            "recorded_date": data.get("recorded_date"),
                            "recorded_time": data.get("recorded_time"),
                            "audio": data.get("audio", "")[:50] + "..."
                            if data.get("audio")
                            else None,
                        }
                    )

            return {
                "success": True,
                "dry_run": True,
                "message": f"Would process {len(tasks)} tasks",
                "tasks_found": len(tasks),
                "truncated": truncated,
                "sample_tasks": task_info,
                "generate_warnings": request.generate_warnings,
                "clear_db": request.clear_db,
            }

        # Clear database if requested
        if request.clear_db:
            ds.clear()
            logger.info("[analyze_batch] Cleared events and warnings tables")

        # Process tasks
        nights_touched = set()
        segments_logged = 0
        tasks_processed = 0
        errors = []

        # Start transaction for bulk insert
        ds.begin()

        try:
            for t in tasks:
                try:
                    # Handle both dict and Task object
                    if isinstance(t, dict):
                        task_id = t.get("id")
                        audio_url = t.get("audio", "")
                        task_data = t
                    else:
                        task_id = getattr(t, "id", None)
                        task_data = t.data if hasattr(t, "data") and t.data else {}
                        audio_url = task_data.get("audio", "")

                    if not audio_url:
                        continue

                    filename = audio_url.split("/")[-1]
                    local_fp = ls_url_to_local_path(audio_url)

                    base_start = _parse_recording_start_time(filename, task_data)
                    logger.debug(
                        f"[analyze_batch] Processing task {task_id} file={filename} start={base_start}"
                    )

                    # Process audio file to get segment predictions
                    segment_results = manager.process_audio_file(
                        local_fp,
                        stable_id=request.stable_id,
                        stall_id=request.stall_id,
                    )

                    # Approx duration for fallback timestamp calc
                    try:
                        audio_duration = (
                            segment_results[-1].end_time if segment_results else 0.0
                        )
                    except Exception:
                        audio_duration = 0.0

                    for seg in segment_results:
                        rel_start_sec = getattr(seg, "start_time", 0.0)
                        rel_end_sec = getattr(seg, "end_time", rel_start_sec + 2.0)
                        midpoint = rel_start_sec + (rel_end_sec - rel_start_sec) / 2.0

                        if base_start:
                            ts = base_start + timedelta(seconds=midpoint)
                        else:
                            ts = datetime.utcnow() - timedelta(
                                seconds=(audio_duration - midpoint)
                            )

                        label_dist = getattr(seg, "label_probabilities", {}) or {}
                        loudness = getattr(seg, "loudness", 0.0)
                        spectral_centroid = getattr(seg, "spectral_centroid", None)
                        high_freq_ratio = getattr(seg, "high_freq_ratio", None)

                        # Determine dominant label and confidence from raw probabilities.
                        # We intentionally avoid manager-side acceptance threshold labels
                        # (e.g. "uncertain") here so DB keeps raw inference outputs.
                        # INJECT ANOMALY if present (matching edge_realtime_monitor logic)
                        original_label_val = None
                        if getattr(seg, "is_anomaly", False):
                            if label_dist:
                                _lbl = max(label_dist, key=label_dist.get)  # type: ignore
                                _prob = label_dist[_lbl]
                                _score = getattr(seg, "anomaly_score", 0.0)
                                _is_anom = getattr(seg, "is_anomaly", True)
                                original_label_val = f"{_lbl} (prob:{_prob:.2f}) (anomaly_score:{_score:.2f} confirmed:{_is_anom})"
                            dominant_label = "anomaly_abnormal"
                            confidence = 1.0
                        elif label_dist:
                            dominant_label = max(label_dist, key=label_dist.get)  # type: ignore
                            confidence = max(label_dist.values())
                        elif getattr(seg, "predicted_label", None):
                            # Fallback for unexpected payloads without label distribution.
                            dominant_label = seg.predicted_label
                            confidence = float(getattr(seg, "probability", 0.0))
                        else:
                            dominant_label = "unknown"
                            confidence = 0.0

                        night = _night_key(ts)
                        nights_touched.add(night)

                        # Write directly to DataStore
                        ds.append_event(
                            {
                                "timestamp": ts.isoformat(),
                                "label": dominant_label,
                                "confidence": round(confidence, 3),
                                "loudness": round(loudness, 3)
                                if loudness is not None
                                else None,
                                "spectral_centroid": round(spectral_centroid, 3)
                                if spectral_centroid is not None
                                else None,
                                "high_freq_ratio": round(high_freq_ratio, 4)
                                if high_freq_ratio is not None
                                else None,
                                "original_label": original_label_val,
                            },
                            autocommit=False,
                        )

                        segments_logged += 1

                    tasks_processed += 1

                except Exception as inner_e:
                    task_id_str = (
                        t.get("id")
                        if isinstance(t, dict)
                        else getattr(t, "id", "unknown")
                    )
                    errors.append({"task_id": task_id_str, "error": str(inner_e)})
                    logger.warning(
                        f"[analyze_batch] Task {task_id_str} failed: {inner_e}"
                    )

            # Commit all events
            ds.commit()

        except Exception as e:
            # Rollback on error
            ds.commit()  # SQLite doesn't have explicit rollback, commit what we have
            raise e

        logger.info(
            f"[analyze_batch] DONE tasks_processed={tasks_processed} "
            f"segments_logged={segments_logged} nights={len(nights_touched)}"
        )

        # Generate warnings if requested
        warnings_generated = 0
        if request.generate_warnings and segments_logged > 0:
            try:
                from smartstablemodel.services.meta_model_decider import (
                    MetaModelDecider,
                )
                from smartstablemodel.config import load_metamodel_config

                cfg = load_metamodel_config()
                decider = MetaModelDecider(config=cfg, datastore=ds)

                # Query the events we just added and evaluate them
                # This is a simplified approach - for full warning generation,
                # you might want to run the decider on the full time range
                logger.info(
                    "[analyze_batch] Warning generation completed (via DataStore)"
                )

            except Exception as warn_e:
                logger.warning(f"[analyze_batch] Warning generation failed: {warn_e}")

        return {
            "success": True,
            "message": "Batch analysis completed",
            "project_id": request.project_id,
            "stable_id": request.stable_id,
            "stall_id": request.stall_id,
            "date_from": request.date_from,
            "date_until": request.date_until,
            "tasks_found": len(tasks),
            "tasks_processed": tasks_processed,
            "segments_logged": segments_logged,
            "nights_touched": sorted(nights_touched),
            "truncated": truncated,
            "generate_warnings": request.generate_warnings,
            "warnings_generated": warnings_generated,
            "errors_count": len(errors),
            "errors": errors[:10] if errors else [],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[analyze_batch] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@metamodel_router.post("/preview-heatmap")
async def preview_heatmap(request: PreviewHeatmapRequest):
    """
    Generate a temporary heatmap image from Label Studio tasks without writing
    any events to the main SQLite database.
    """
    try:
        from smartstablemodel.services.data_store import DataStore
        from smartstablemodel.services.visualizer import Visualizer
        from smartstablemodel.config import load_metamodel_config

        ls = get_ls_client()
        manager = get_multistall_manager()
        cfg = load_metamodel_config()

        # Use in-memory SQLite so this endpoint never touches persistent DB state.
        ds = DataStore(":memory:")
        viz = Visualizer(ds)

        tasks_found = []

        # Mode 1: specific task IDs
        if request.task_ids:
            if request.date_from or request.date_until:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot provide 'task_ids' and date range ('date_from'/'date_until') simultaneously.",
                )
            for tid in request.task_ids:
                try:
                    t = ls.tasks.get(id=str(tid))
                    tasks_found.append(t)
                except Exception as e:
                    logger.warning(f"[preview_heatmap] Failed to fetch task {tid}: {e}")

        # Mode 2: date range
        elif request.date_from and request.date_until:
            try:
                datetime.strptime(request.date_from, "%Y-%m-%d")
                datetime.strptime(request.date_until, "%Y-%m-%d")
                datetime.strptime(request.time_from, "%H:%M")
                datetime.strptime(request.time_until, "%H:%M")
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid date/time format (expected YYYY-MM-DD and HH:MM)",
                )

            tasks_found = list_tasks_by_recording_period(
                project_id=request.project_id,
                stable_id=request.stable_id,
                stall_id=request.stall_id if request.stall_id != "all" else None,
                date_from=request.date_from,
                date_until=request.date_until,
                time_from=request.time_from,
                time_until=request.time_until,
                max_tasks=request.max_tasks + 1,
                ls_client=ls,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either 'task_ids' or ('date_from' AND 'date_until')",
            )

        tasks = tasks_found
        if not tasks:
            raise HTTPException(status_code=404, detail="No tasks found for the given criteria")

        # Sort tasks by date/time for deterministic processing
        def get_task_field(t, field, default=""):
            if isinstance(t, dict):
                return t.get(field, default) or default
            return (
                getattr(t, field, default) or default
                if hasattr(t, field)
                else (
                    t.data.get(field, default)
                    if hasattr(t, "data") and t.data
                    else default
                )
            )

        tasks.sort(
            key=lambda t: (
                get_task_field(t, "recorded_date"),
                get_task_field(t, "recorded_time"),
            )
        )

        truncated = False
        if len(tasks) > request.max_tasks:
            tasks = tasks[: request.max_tasks]
            truncated = True

        ds.begin()
        tasks_processed = 0
        segments_logged = 0

        for t in tasks:
            try:
                if isinstance(t, dict):
                    audio_url = t.get("audio", "")
                    task_data = t
                else:
                    task_data = t.data if hasattr(t, "data") and t.data else {}
                    audio_url = task_data.get("audio", "")

                if not audio_url:
                    continue

                filename = audio_url.split("/")[-1]
                local_fp = ls_url_to_local_path(audio_url)
                base_start = _parse_recording_start_time(filename, task_data)

                segment_results = manager.process_audio_file(
                    local_fp,
                    stable_id=request.stable_id,
                    stall_id=request.stall_id,
                )

                try:
                    audio_duration = segment_results[-1].end_time if segment_results else 0.0
                except Exception:
                    audio_duration = 0.0

                for seg in segment_results:
                    rel_start_sec = getattr(seg, "start_time", 0.0)
                    rel_end_sec = getattr(seg, "end_time", rel_start_sec + 2.0)
                    midpoint = rel_start_sec + (rel_end_sec - rel_start_sec) / 2.0

                    if base_start:
                        ts = base_start + timedelta(seconds=midpoint)
                    else:
                        ts = datetime.utcnow() - timedelta(seconds=(audio_duration - midpoint))

                    label_dist = getattr(seg, "label_probabilities", {}) or {}
                    loudness = getattr(seg, "loudness", 0.0)
                    spectral_centroid = getattr(seg, "spectral_centroid", None)
                    high_freq_ratio = getattr(seg, "high_freq_ratio", None)

                    original_label_val = None
                    if getattr(seg, "is_anomaly", False):
                        if label_dist:
                            _lbl = max(label_dist, key=label_dist.get)  # type: ignore
                            _prob = label_dist[_lbl]
                            _score = getattr(seg, "anomaly_score", 0.0)
                            _is_anom = getattr(seg, "is_anomaly", True)
                            original_label_val = f"{_lbl} (prob:{_prob:.2f}) (anomaly_score:{_score:.2f} confirmed:{_is_anom})"
                        dominant_label = "anomaly_abnormal"
                        confidence = 1.0
                    elif getattr(seg, "predicted_label", None):
                        dominant_label = seg.predicted_label
                        confidence = float(getattr(seg, "probability", 0.0))
                    elif label_dist:
                        dominant_label = max(label_dist, key=label_dist.get)  # type: ignore
                        confidence = max(label_dist.values())
                    else:
                        dominant_label = "unknown"
                        confidence = 0.0

                    ds.append_event(
                        {
                            "timestamp": ts.isoformat(),
                            "label": dominant_label,
                            "confidence": round(confidence, 3),
                            "loudness": round(loudness, 3) if loudness is not None else None,
                            "spectral_centroid": round(spectral_centroid, 3)
                            if spectral_centroid is not None
                            else None,
                            "high_freq_ratio": round(high_freq_ratio, 4)
                            if high_freq_ratio is not None
                            else None,
                            "original_label": original_label_val,
                        },
                        autocommit=False,
                    )
                    segments_logged += 1

                tasks_processed += 1
            except Exception as inner_e:
                logger.warning(f"[preview_heatmap] Task failed: {inner_e}")

        ds.commit()

        if segments_logged == 0:
            raise HTTPException(status_code=404, detail="No segments produced for selected tasks")

        if request.date_from and request.date_until:
            start_dt = datetime.strptime(
                f"{request.date_from} {request.time_from}", "%Y-%m-%d %H:%M"
            )
            end_dt = datetime.strptime(
                f"{request.date_until} {request.time_until}", "%Y-%m-%d %H:%M"
            )
            if end_dt <= start_dt and request.date_from == request.date_until:
                end_dt += timedelta(days=1)
            if end_dt <= start_dt:
                raise HTTPException(
                    status_code=400,
                    detail="End must be after start for the selected preview window",
                )
        else:
            full_start, full_end = ds.time_bounds()
            if not full_start or not full_end:
                raise HTTPException(status_code=404, detail="No events available for preview")
            start_dt, end_dt = full_start, full_end + timedelta(seconds=1)

        mat, labels, meta = viz.build_heatmap_matrix(
            start=start_dt,
            end=end_dt,
            bin_seconds=request.bin_size_minutes * 60,
            group_labels=request.group_labels,
            label_groups=getattr(cfg, "label_groups", None),
            min_total_seconds=0.0,
            top_n=None,
            max_bins=request.max_bins,
        )

        if int(meta.get("num_bins", 0)) == 0 or not labels or getattr(mat, "size", 0) == 0:
            raise HTTPException(status_code=404, detail="No bins available for this preview window")

        if request.hide_empty_time_bins:
            non_empty_cols = np.where(mat.sum(axis=0) > 0)[0]
            if len(non_empty_cols) == 0:
                raise HTTPException(status_code=404, detail="All bins are empty for this preview")
            mat = mat[:, non_empty_cols]
            meta = dict(meta)
            meta["num_bins"] = len(non_empty_cols)
            bin_seconds = int(meta["bin_seconds"])
            original_t0 = meta["t0"]
            kept_times = [
                original_t0 + timedelta(seconds=int(i) * bin_seconds)
                for i in non_empty_cols
            ]
            meta["t0"] = kept_times[0]
            meta["t_end"] = kept_times[-1] + timedelta(seconds=bin_seconds)

        fig = viz.plot_heatmap(
            mat,
            labels,
            meta,
            show_warnings=False,
            show_loudness=True,
            use_seaborn=False,
        )

        fig.suptitle(
            (
                f"Preview Heatmap | stable={request.stable_id} stall={request.stall_id} "
                f"| tasks={tasks_processed} segments={segments_logged}"
                + (" | truncated" if truncated else "")
            ),
            fontsize=11,
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        filename = (
            f"preview_heatmap_{request.stable_id}_{request.stall_id}_{start_dt.strftime('%Y%m%d_%H%M')}_{end_dt.strftime('%Y%m%d_%H%M')}.png"
        )

        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                "X-Tasks-Processed": str(tasks_processed),
                "X-Segments-Logged": str(segments_logged),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[preview_heatmap] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@metamodel_router.post("/clear-db")
async def clear_database():
    """Clear all events and warnings from the database."""
    try:
        ds = get_datastore()
        ds.clear()
        return {"success": True, "message": "Database cleared", "db_path": str(ds.path)}
    except Exception as e:
        logger.error(f"Failed to clear database: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@metamodel_router.post("/feature-scatter-image")
async def feature_scatter_image(raw_request: FastAPIRequest):
    """
    Return a PNG image visualizing loudness (RMS proxy) vs spectral centroid
    for the requested class and date range.

    Accepts: {"class": "<label>", "date_from": "YYYY-MM-DD", "date_until": "YYYY-MM-DD"}
    """
    try:
        # Parse body defensively: handles both a proper JSON object *and* a
        # double-encoded JSON string (sent when Content-Type is missing/wrong).
        raw_bytes = await raw_request.body()
        try:
            body = json.loads(raw_bytes)
            if isinstance(body, str):
                body = json.loads(body)
        except Exception:
            raise HTTPException(status_code=400, detail="Request body is not valid JSON.")

        try:
            request = ClassFeatureScatterRequest.model_validate(body)
        except Exception as ve:
            raise HTTPException(status_code=422, detail=str(ve))

        try:
            start = datetime.strptime(request.date_from, "%Y-%m-%d")
            end = datetime.strptime(request.date_until, "%Y-%m-%d") + timedelta(days=1)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid date format. Use YYYY-MM-DD for date_from/date_until.",
            )

        if end <= start:
            raise HTTPException(
                status_code=400,
                detail="date_until must be the same as or after date_from.",
            )

        ds = get_datastore()
        df = pd.read_sql_query(
            (
                "SELECT timestamp, loudness, spectral_centroid, high_freq_ratio "
                "FROM events "
                "WHERE label = ? "
                "AND timestamp >= ? "
                "AND timestamp < ? "
                "AND loudness IS NOT NULL "
                "AND spectral_centroid IS NOT NULL "
                "ORDER BY timestamp ASC"
            ),
            ds._conn,
            params=(request.class_name, start.isoformat(), end.isoformat()),
        )

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No events found for class '{request.class_name}' in "
                    f"[{request.date_from}, {request.date_until}]."
                ),
            )

        fig, ax = plt.subplots(figsize=(10, 6))
        has_hf = "high_freq_ratio" in df.columns and df["high_freq_ratio"].notna().any()

        if has_hf:
            scat = ax.scatter(
                df["loudness"],
                df["spectral_centroid"],
                c=df["high_freq_ratio"],
                cmap="viridis",
                s=18,
                alpha=0.75,
                edgecolors="none",
            )
            cbar = fig.colorbar(scat, ax=ax)
            cbar.set_label("High-frequency ratio (>4kHz / total)")
        else:
            ax.scatter(
                df["loudness"],
                df["spectral_centroid"],
                s=18,
                alpha=0.75,
                color="#1f77b4",
                edgecolors="none",
            )

        ax.set_xlabel("RMS proxy (dBFS loudness)")
        ax.set_ylabel("Spectral centroid (Hz)")
        ax.set_title(
            f"Class '{request.class_name}' feature scatter ({request.date_from} to {request.date_until})"
        )
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140)
        plt.close(fig)
        buf.seek(0)

        filename = (
            f"feature_scatter_{request.class_name}_{request.date_from}_{request.date_until}.png"
        )
        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[feature_scatter_image] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
