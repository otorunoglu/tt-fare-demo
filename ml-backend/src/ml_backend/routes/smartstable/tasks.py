"""
Task management routes for Label Studio.

Provides endpoints for:
- Updating task metadata from filenames
- Adding new tasks from audio/video files
- Backup and restore functionality
- Project listing and sync operations
- Grid video creation
"""

import os
import re
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from fastapi import APIRouter, Query, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
import uuid
from yt_dlp import YoutubeDL

from ml_backend.smartstable_core.task_filtering import (
    extract_raw_data_from_filename,
    extract_and_map_metadata_from_filename,
    check_if_task_in_time_window,
    list_tasks_by_recording_period,
)
from ml_backend.util.label_studio_helper import ls_url_to_local_path
from ml_backend.ls_client import get_ls_client

logger = logging.getLogger("TasksRouter")

tasks_router = APIRouter(prefix="/tasks", tags=["tasks"])


# ─── Configuration ────────────────────────────────────────────────────────────


def load_config(suppress_log: bool = False) -> Dict[str, Any]:
    """Load configuration mappings from task_config.toml"""
    config_path = os.getenv("TASKS_CONFIG_PATH")
    if not config_path:
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "config",
            "smartstable",
            "task_config.toml",
        )
        if not suppress_log:
            logger.warning(
                f"TASKS_CONFIG_PATH not set, using default path: {config_path}"
            )
    if not suppress_log:
        logger.info(f"Loading config from {config_path}")
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        logger.warning(
            f"Config file not found at {config_path}. Using default mappings."
        )
        return {
            "stable_mappings": {"stable01": "rauhalahti stable"},
            "stable_configs": {
                "stable01": {
                    "stall_mappings": {
                        "stall01": "Stall 01",
                        "stall02": "Stall 02",
                        "stall06": "Stall 06",
                    },
                    "horse_mappings": {
                        "horse01": "Horse 01",
                        "horse02": "Horse 02",
                        "horse06": "Horse 06",
                    },
                }
            },
        }


def get_or_default_project(ls, project_id: Optional[int] = None) -> int:
    """Get project_id or use first available project."""
    if project_id:
        return project_id
    try:
        projects = list(ls.projects.list())
        if projects:
            return projects[0].id
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"No project_id provided and failed to list projects: {str(e)}",
        )
    raise HTTPException(
        status_code=400, detail="No project_id provided and no projects found"
    )


# ─── Request/Response Models ──────────────────────────────────────────────────


class UpdateTasksRequest(BaseModel):
    project_id: Optional[int] = None
    dry_run: bool = True


class UpdateTaskFieldsRequest(BaseModel):
    project_id: Optional[int] = None
    task_id: Optional[int] = None
    stable_id: Optional[str] = "stable01"
    stall_id: Optional[str] = "all"
    date_from: Optional[str] = None
    time_from: Optional[str] = "00:00"
    date_until: Optional[str] = None
    time_until: Optional[str] = "23:59"
    update: Dict[str, Any]
    dry_run: bool = True


class AddTasksRequest(BaseModel):
    project_id: Optional[int] = None
    audio_directory: str = "/label-studio/data/audio/"
    video_directory: str = "/label-studio/data/video/"
    dry_run: bool = True
    extra_metadata: Dict[str, Any] = {}


class AddTasksByConfigRequest(BaseModel):
    project_id: Optional[int] = None
    audio_directory: str = "/label-studio/data/audio/"
    video_directory: str = "/label-studio/data/video/"
    dry_run: bool = True
    stable_id: str = "stable01"
    stall_id: str = "all"
    date_from: str = "2000-01-01"
    time_from: str = "00:00"
    date_until: str = "2099-12-31"
    time_until: str = "23:59"
    horse: Optional[str] = None
    report: str = ""
    extra_metadata: Dict[str, Any] = {}


class AddFromSourceRequest(BaseModel):
    project_id: Optional[int] = None
    audio_directory: str = "/data/audio"
    video_directory: str = "/data/video"
    audio_file_name: str
    video_file_name: str = "none"
    source: str = "unknown"
    horse: str = "unknown"
    recorded_date: str = "unknown"
    recorded_time: str = "unknown"
    report: str = ""
    extra_metadata: Dict[str, Any] = {}


class BackupRequest(BaseModel):
    project_id: Optional[int] = None
    backup_filename: Optional[str] = None
    backup_directory: str = "/label-studio/data/label_studio_backups"
    include_annotations: bool = True
    include_predictions: bool = True


class RestoreRequest(BaseModel):
    project_id: Optional[int] = None
    backup_path: Optional[str] = None
    backup_directory: str = "/label-studio/data/label_studio_backups"
    backup_filename: Optional[str] = None
    dry_run: bool = True
    restore_mode: str = "skip_existing"  # "skip_existing", "overwrite", "create_new"
    restore_annotations: bool = True
    restore_predictions: bool = True


class AddSingleTaskRequest(BaseModel):
    project_id: Optional[int] = None
    dry_run: bool = False
    recorded_date: str = "unknown"
    recorded_time: str = "unknown"
    audio_directory: str = "/label-studio/data/audio/"
    video_directory: str = "/label-studio/data/video/"
    audio_file_name: str
    video_file_name: str = ""

    source: str = "recording"
    horse: str = "unknown"
    report: str = ""
    extra_metadata: Dict[str, Any] = {}


# ─── Routes ───────────────────────────────────────────────────────────────────


@tasks_router.get("/debug")
def debug_tasks():
    """Health check for task routes."""
    return {"status": "ok", "message": "task route reachable"}


@tasks_router.get("/get-task-source")
def get_task_source(
    project_id: int = Query(..., description="Label Studio project number"),
    task_number: int = Query(..., description="Label Studio task number (task ID)"),
):
    """
    Return the full Label Studio task JSON (including annotations/predictions)
    for a given project and task number.
    """
    ls = get_ls_client()

    try:
        task = ls.tasks.get(id=str(task_number))
    except Exception as e:
        logger.error(
            f"Failed to fetch task {task_number} from Label Studio: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_number} not found",
        )

    task_payload = jsonable_encoder(task)

    # Validate that the task belongs to the requested project.
    # Depending on LS SDK version this may live at top-level `project`.
    task_project = task_payload.get("project")
    if task_project is not None and int(task_project) != int(project_id):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Task {task_number} exists but belongs to project {task_project}, "
                f"not project {project_id}"
            ),
        )

    return task_payload


@tasks_router.post("/sync-metadata")
def sync_task_metadata(request: UpdateTasksRequest):
    """
    Update Label Studio tasks with metadata extracted from filenames.

    Extracts stable, stall, horse, date, time from audio filenames and
    updates the task data with human-readable mappings from config.
    """
    ls = get_ls_client()

    try:
        project_id = get_or_default_project(ls, request.project_id)
        config = load_config(suppress_log=True)

        logger.info(
            f"Updating tasks for project {project_id} (dry_run: {request.dry_run})"
        )

        # Get all tasks
        tasks = list(ls.tasks.list(project=project_id))
        logger.info(f"Found {len(tasks)} tasks")

        updates_needed = []
        errors = []

        for task in tasks:
            try:
                audio_url = task.data.get("audio", "") if hasattr(task, "data") else ""
                if not audio_url:
                    continue

                filename = audio_url.split("/")[-1]
                if not filename.endswith(".flac"):
                    continue

                # Extract and map metadata
                new_metadata = extract_and_map_metadata_from_filename(filename, config)
                if not new_metadata:
                    continue

                # Check if update is needed
                current_data = task.data if hasattr(task, "data") else {}
                needs_update = False
                changes = []

                for key, new_value in new_metadata.items():
                    current_value = current_data.get(key)
                    if current_value != new_value:
                        needs_update = True
                        changes.append(f"{key}: '{current_value}' -> '{new_value}'")

                if needs_update:
                    updates_needed.append(
                        {
                            "task_id": task.id,
                            "filename": filename,
                            "changes": changes,
                            "new_data": {**current_data, **new_metadata},
                        }
                    )

            except Exception as e:
                errors.append(f"Task {getattr(task, 'id', 'unknown')}: {str(e)}")

        summary = {
            "project_id": project_id,
            "total_tasks": len(tasks),
            "tasks_needing_updates": len(updates_needed),
            "errors_encountered": len(errors),
            "dry_run": request.dry_run,
            "sample_updates": updates_needed[:3] if updates_needed else [],
            "errors": errors[:5] if errors else [],
        }

        if request.dry_run:
            summary["message"] = (
                "DRY RUN - No changes were made. Set dry_run=false to apply updates."
            )
            return summary

        if not updates_needed:
            summary["message"] = "No updates needed!"
            return summary

        # Apply updates
        logger.info(f"Applying {len(updates_needed)} updates...")
        success_count = 0
        update_errors = []

        for update in updates_needed:
            try:
                ls.tasks.update(id=update["task_id"], data=update["new_data"])
                success_count += 1
            except Exception as e:
                update_errors.append(
                    f"Failed to update task {update['task_id']}: {str(e)}"
                )

        summary.update(
            {
                "message": f"Update complete: {success_count}/{len(updates_needed)} tasks updated successfully",
                "successful_updates": success_count,
                "failed_updates": len(update_errors),
                "update_errors": update_errors,
            }
        )

        return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task update failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@tasks_router.post("/add")
def add_new_tasks(request: AddTasksRequest):
    """
    Add new tasks for audio files that don't exist in Label Studio yet.

    Scans the audio directory for .flac files not already in the project
    and creates tasks with metadata extracted from filenames.
    """
    ls = get_ls_client()

    try:
        project_id = get_or_default_project(ls, request.project_id)
        config = load_config(suppress_log=True)

        logger.info(
            f"Adding new tasks for project {project_id} from {request.audio_directory}"
        )

        # Get existing task filenames
        existing_tasks = list(ls.tasks.list(project=project_id))
        existing_filenames = set()
        for task in existing_tasks:
            audio_url = task.data.get("audio", "") if hasattr(task, "data") else ""
            if audio_url:
                existing_filenames.add(audio_url.split("/")[-1])

        logger.info(f"Found {len(existing_filenames)} existing tasks")

        if not os.path.exists(request.audio_directory):
            raise HTTPException(
                status_code=400,
                detail=f"Audio directory {request.audio_directory} does not exist",
            )

        # Scan for new audio files
        new_tasks = []
        errors = []
        orphan_files = []

        general_config = config.get("stable_configs", {})

        for filename in os.listdir(request.audio_directory):
            if not filename.endswith(".flac") or filename in existing_filenames:
                continue

            # Check if stable has auto_import disabled
            stable_id = filename.split("_")[0] if "_" in filename else None
            if stable_id and not general_config.get(stable_id, {}).get(
                "auto_import_tasks", True
            ):
                logger.debug(f"Skipping {filename} - stable set to manual import mode")
                continue

            try:
                metadata = extract_and_map_metadata_from_filename(filename, config)
                if not metadata:
                    errors.append(f"Failed to extract metadata from {filename}")
                    continue

                # Check for video file
                video_filename = filename.replace(".flac", ".mp4").replace(
                    "_mic", "_cam"
                )
                video_path = os.path.join(request.video_directory, video_filename)

                if os.path.exists(video_path):
                    video_url = (
                        f"/data/local-files/?d=label-studio/data/video/{video_filename}"
                    )
                else:
                    video_url = ""
                    orphan_files.append(filename)

                task_data = {
                    "audio": f"/data/local-files/?d=label-studio/data/audio/{filename}",
                    "video": video_url,
                    "stable": metadata["stable"],
                    "stall": metadata["stall"],
                    "horse": metadata["horse"],
                    "recorded_date": metadata["recorded_date"],
                    "recorded_time": metadata["recorded_time"],
                    "recorded_date_time": metadata["recorded_date_time"],
                    "known_event": "unknown",
                    "report": "",
                    "manual_notes": "",
                    "weather": "unknown",
                    "events": "",
                    "grid_video": "",
                }

                if request.extra_metadata:
                    task_data.update(request.extra_metadata)

                new_tasks.append({"filename": filename, "task_data": task_data})

            except Exception as e:
                errors.append(f"Error processing {filename}: {str(e)}")

        summary = {
            "project_id": project_id,
            "audio_directory": request.audio_directory,
            "existing_tasks": len(existing_filenames),
            "new_audio_files_found": len(new_tasks) + len(errors),
            "new_tasks_to_create": len(new_tasks),
            "orphan_files": len(orphan_files),
            "errors_encountered": len(errors),
            "dry_run": request.dry_run,
            "sample_new_tasks": new_tasks[:3] if new_tasks else [],
            "sample_orphan_files": orphan_files[:5] if orphan_files else [],
            "errors": errors[:5] if errors else [],
        }

        if request.dry_run:
            summary["message"] = (
                "DRY RUN - No tasks were created. Set dry_run=false to create tasks."
            )
            return summary

        if not new_tasks:
            summary["message"] = "No new tasks to create!"
            return summary

        # Create tasks
        logger.info(f"Creating {len(new_tasks)} new tasks...")
        created_count = 0
        creation_errors = []

        for task_info in new_tasks:
            try:
                ls.tasks.create(project=project_id, data=task_info["task_data"])
                created_count += 1
            except Exception as e:
                creation_errors.append(
                    f"Failed to create task for {task_info['filename']}: {str(e)}"
                )

        summary.update(
            {
                "message": f"Task creation complete: {created_count}/{len(new_tasks)} tasks created successfully",
                "successful_creations": created_count,
                "failed_creations": len(creation_errors),
                "creation_errors": creation_errors,
            }
        )

        return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add new tasks failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@tasks_router.post("/add-by-config")
def add_new_tasks_by_config(request: AddTasksByConfigRequest):
    """
    Add new tasks filtered by date/time window and stable/stall config.

    Useful for importing specific date ranges or stalls only.
    """
    ls = get_ls_client()

    try:
        project_id = get_or_default_project(ls, request.project_id)
        config = load_config(suppress_log=True)

        stable_name = config.get("stable_mappings", {}).get(
            request.stable_id, request.stable_id
        )

        # Get existing filenames
        existing_tasks = list(ls.tasks.list(project=project_id))
        existing_filenames = set()
        for task in existing_tasks:
            audio_url = task.data.get("audio", "") if hasattr(task, "data") else ""
            if audio_url:
                existing_filenames.add(audio_url.split("/")[-1])

        logger.info(f"Found {len(existing_filenames)} existing tasks")

        # Filter and prepare new tasks
        new_tasks = []
        errors = []
        orphan_files = []

        if not os.path.exists(request.audio_directory):
            raise HTTPException(
                status_code=400,
                detail=f"Audio directory {request.audio_directory} does not exist",
            )

        for audio_filename in os.listdir(request.audio_directory):
            if (
                not audio_filename.endswith(".flac")
                or audio_filename in existing_filenames
            ):
                continue

            try:
                metadata = extract_and_map_metadata_from_filename(
                    audio_filename, config
                )
                if not metadata:
                    errors.append(f"Failed to extract metadata from {audio_filename}")
                    continue

                # Check time window
                if not check_if_task_in_time_window(
                    metadata,
                    (request.time_from, request.time_until),
                    (request.date_from, request.date_until),
                    stable_name,
                ):
                    continue

                # Check stall filter
                if request.stall_id != "all":
                    raw_data = extract_raw_data_from_filename(audio_filename)
                    if raw_data and raw_data["stall_raw"] != request.stall_id:
                        continue

                # Check for video
                video_filename = audio_filename.replace(".flac", ".mp4").replace(
                    "_mic", "_cam"
                )
                video_path = os.path.join(request.video_directory, video_filename)

                if os.path.exists(video_path):
                    video_url = (
                        f"/data/local-files/?d=label-studio/data/video/{video_filename}"
                    )
                else:
                    video_url = ""
                    orphan_files.append(audio_filename)

                task_data = {
                    "audio": f"/data/local-files/?d=label-studio/data/audio/{audio_filename}",
                    "video": video_url,
                    "stable": metadata["stable"],
                    "stall": metadata["stall"],
                    "horse": request.horse or metadata["horse"],
                    "recorded_date": metadata["recorded_date"],
                    "recorded_time": metadata["recorded_time"],
                    "recorded_date_time": metadata["recorded_date_time"],
                    "report": request.report,
                    "known_event": "unknown",
                    "manual_notes": "",
                    "weather": "unknown",
                    "events": "",
                    "grid_video": "",
                    "diet_type": "",
                    "protocol_step": "",
                }

                if request.extra_metadata:
                    task_data.update(request.extra_metadata)

                new_tasks.append({"filename": audio_filename, "task_data": task_data})

            except Exception as e:
                errors.append(f"Error processing {audio_filename}: {str(e)}")

        summary = {
            "project_id": project_id,
            "filter": {
                "stable_id": request.stable_id,
                "stall_id": request.stall_id,
                "date_range": f"{request.date_from} to {request.date_until}",
                "time_range": f"{request.time_from} to {request.time_until}",
                "horse_override": request.horse,
                "report": request.report[:50] + "..."
                if len(request.report) > 50
                else request.report,
            },
            "existing_tasks": len(existing_filenames),
            "new_tasks_to_create": len(new_tasks),
            "orphan_files_count": len(orphan_files),
            "errors_encountered": len(errors),
            "dry_run": request.dry_run,
            "sample_new_tasks": new_tasks[:3] if new_tasks else [],
            "sample_orphan_files": orphan_files[:5] if orphan_files else [],
            "errors": errors[:5] if errors else [],
        }

        if request.dry_run:
            summary["message"] = (
                "DRY RUN - No changes were made. Set dry_run=false to apply updates."
            )
            return summary

        # Create tasks
        created_count = 0
        creation_errors = []

        for task_info in new_tasks:
            try:
                ls.tasks.create(project=project_id, data=task_info["task_data"])
                created_count += 1
            except Exception as e:
                creation_errors.append(
                    f"Failed to create task for {task_info['filename']}: {str(e)}"
                )

        summary.update(
            {
                "message": f"Task creation complete: {created_count}/{len(new_tasks)} tasks created successfully",
                "successful_creations": created_count,
                "creation_errors": creation_errors,
            }
        )

        return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add new tasks: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@tasks_router.post("/add-from-source")
def add_from_other_source(request: AddFromSourceRequest):
    """
    Add a task from an external audio source (not following standard naming).
    """
    ls = get_ls_client()

    try:
        project_id = get_or_default_project(ls, request.project_id)

        task_data = {
            "audio": f"/data/local-files/?d=label-studio{request.audio_directory}/{request.audio_file_name}",
            "video": f"/data/local-files/?d=label-studio{request.video_directory}/{request.video_file_name}"
            if request.video_file_name != "none"
            else "",
            "stable": f"audio source: {request.source}",
            "stall": "",
            "horse": request.horse,
            "recorded_date": request.recorded_date,
            "recorded_time": request.recorded_time,
            "recorded_date_time": f"{request.recorded_date} {request.recorded_time}",
            "report": request.report,
            "known_event": "unknown",
            "manual_notes": "",
            "events": "",
            "grid_video": "",
        }

        if request.extra_metadata:
            task_data.update(request.extra_metadata)

        created_task = ls.tasks.create(project=project_id, data=task_data)

        return {
            "message": "Task created successfully",
            "task_id": created_task.id,
            "audio_file": request.audio_file_name,
        }

    except Exception as e:
        logger.error(f"Failed to add task: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class AddYoutubeTaskRequest(BaseModel):
    youtube_url: str
    audio_directory: str
    video_directory: str
    project_id: Optional[int] = None
    dry_run: bool = False
    recorded_date: str = "unknown"
    recorded_time: str = "unknown"
    horse: str = "unknown"
    report: str = ""
    sample_rate: Optional[int] = (
        None  # None = keep native (e.g. 32000 for PANNs-ready wav)
    )
    mono: bool = False  # True = downmix to 1 channel
    extra_metadata: Optional[Dict[str, Any]] = None


@tasks_router.post("/add-youtube")
def add_youtube_task(request: AddYoutubeTaskRequest):
    """
    Download a YouTube video, write the video as <uuid>.mp4 and the extracted
    audio as <uuid>.wav into the given directories, and create a Label Studio task.
    'source' is set to 'youtube:<uuid>' and that same uuid is the file basename.
    """
    ls = get_ls_client()
    try:
        project_id = get_or_default_project(ls, request.project_id)

        audio_dir = request.audio_directory.rstrip("/")
        video_dir = request.video_directory.rstrip("/")

        # Metadata only, no download. Also covers the dry_run path.
        with YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True}) as ydl:
            info = ydl.extract_info(request.youtube_url, download=False)
        if info.get("_type") == "playlist":
            raise HTTPException(
                status_code=400, detail="Pass a single video URL, not a playlist."
            )

        video_uuid = str(uuid.uuid4())
        source = f"youtube:{video_uuid}"

        audio_file_name = f"{video_uuid}.wav"
        video_file_name = f"{video_uuid}.mp4"
        audio_url = f"/data/local-files/?d={audio_dir}/{audio_file_name}"
        video_url = f"/data/local-files/?d={video_dir}/{video_file_name}"

        task_data = {
            "audio": audio_url,
            "video": video_url,
            "stable": f"source: {source}",
            "stall": "",
            "horse": request.horse,
            "recorded_date": request.recorded_date,
            "recorded_time": request.recorded_time,
            "recorded_date_time": f"{request.recorded_date} {request.recorded_time}",
            "report": request.report,
            "source": source,
            "known_event": "unknown",
            "manual_notes": "",
            "weather": "unknown",
            "events": "",
            "grid_video": "",
            "diet_type": "",
            "protocol_step": "",
            # provenance so you can trace the task back to the original video
            "youtube_url": info.get("webpage_url", request.youtube_url),
            "youtube_id": info.get("id", ""),
            "youtube_title": info.get("title", ""),
        }
        if request.extra_metadata:
            task_data.update(request.extra_metadata)

        if request.dry_run:
            return {
                "message": "Dry run successful. Nothing downloaded; task not created.",
                "project_id": project_id,
                "source": source,
                "audio_file": audio_file_name,
                "video_file": video_file_name,
                "task_data": task_data,
            }

        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        video_path = os.path.join(video_dir, video_file_name)
        audio_path = os.path.join(audio_dir, audio_file_name)

        # Download merged mp4. Prefers native mp4/m4a so the merge yields a real mp4.
        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "merge_output_format": "mp4",
            "outtmpl": os.path.join(video_dir, f"{video_uuid}.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([request.youtube_url])

        if not os.path.exists(video_path):
            raise HTTPException(
                status_code=500,
                detail=f"Expected {video_path} after download. The source likely had no "
                f"mp4-native stream; adjust the yt-dlp format selector.",
            )

        # Extract audio to wav.
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le"]
        if request.sample_rate:
            ffmpeg_cmd += ["-ar", str(request.sample_rate)]
        if request.mono:
            ffmpeg_cmd += ["-ac", "1"]
        ffmpeg_cmd.append(audio_path)
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise HTTPException(
                status_code=500,
                detail=f"ffmpeg failed: {e.stderr.decode(errors='ignore')}",
            )

        created_task = ls.tasks.create(project=project_id, data=task_data)
        return {
            "message": "Video downloaded and task created successfully",
            "task_id": created_task.id,
            "project_id": project_id,
            "source": source,
            "audio_file": audio_file_name,
            "video_file": video_file_name,
            "task_data": task_data,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add youtube task: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@tasks_router.post("/add-single")
def add_single_task(request: AddSingleTaskRequest):
    """
    Add a single task manually, usually for a specific recording.
    """
    ls = get_ls_client()

    try:
        project_id = get_or_default_project(ls, request.project_id)

        # Construct task data
        # Note: input directories typically strictly follow conventions, but we allow flexibility here.
        # Ensure directories end with /
        audio_dir = request.audio_directory.rstrip("/")
        video_dir = request.video_directory.rstrip("/")

        audio_url = f"/data/local-files/?d={audio_dir}/{request.audio_file_name}"

        video_url = ""
        if request.video_file_name:
            video_url = f"/data/local-files/?d={video_dir}/{request.video_file_name}"

        task_data = {
            "audio": audio_url,
            "video": video_url,
            "stable": f"source: {request.source}",  # Using 'source' as stable identifier/metadata
            "stall": "",
            "horse": request.horse,
            "recorded_date": request.recorded_date,
            "recorded_time": request.recorded_time,
            "recorded_date_time": f"{request.recorded_date} {request.recorded_time}",
            "report": request.report,
            "source": request.source,
            "known_event": "unknown",
            "manual_notes": "",
            "weather": "unknown",
            "events": "",
            "grid_video": "",
            "diet_type": "",
            "protocol_step": "",
        }

        # Merge extra metadata if provided
        if request.extra_metadata:
            task_data.update(request.extra_metadata)

        if request.dry_run:
            return {
                "message": "Dry run successful. Task would be created with the following data.",
                "project_id": project_id,
                "task_data": task_data,
            }

        # Check for existing task with same audio URL to avoid duplicates (optional but safe)
        # This can be slow if there are many tasks, effectively we rely on users not spamming
        # ... skipping duplicate check for now to keep it fast and simple,
        # relying on LS potentially handling it or user discretion.

        created_task = ls.tasks.create(project=project_id, data=task_data)

        return {
            "message": "Task created successfully",
            "task_id": created_task.id,
            "project_id": project_id,
            "audio_file": request.audio_file_name,
            "task_data": task_data,
        }

    except Exception as e:
        logger.error(f"Failed to add single task: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@tasks_router.post("/update-fields")
def update_task_fields(request: UpdateTaskFieldsRequest):
    """
    Update specific fields of Label Studio tasks.

    Can update a single task by ID or a batch of tasks based on filters.
    Safe update: merges new fields into existing task data.
    """
    ls = get_ls_client()

    try:
        project_id = get_or_default_project(ls, request.project_id)

        # Determine tasks to update
        tasks_to_process = []

        if request.task_id:
            # Single task mode
            try:
                task = ls.tasks.get(request.task_id)
                if not task:
                    raise HTTPException(
                        status_code=404, detail=f"Task {request.task_id} not found"
                    )
                tasks_to_process.append(task)
            except Exception as e:
                # Check if it's a 404 from SDK or other error
                if "404" in str(e):
                    raise HTTPException(
                        status_code=404, detail=f"Task {request.task_id} not found"
                    )
                raise e
        else:
            # Filter mode
            if not request.date_from or not request.date_until:
                raise HTTPException(
                    status_code=400,
                    detail="date_from and date_until are required for batch updates",
                )

            stall_filter = request.stall_id if request.stall_id != "all" else None

            # Fetch tasks using SmartStable filtering logic
            tasks_to_process = list_tasks_by_recording_period(
                project_id=project_id,
                stable_id=request.stable_id,
                stall_id=stall_filter,
                date_from=request.date_from,
                date_until=request.date_until,
                time_from=request.time_from,
                time_until=request.time_until,
                return_summary=False,  # We need full task objects to preserve data
                ls_client=ls,
            )

        logger.info(
            f"Found {len(tasks_to_process)} tasks to update (dry_run={request.dry_run})"
        )

        updates_needed = []
        errors = []

        for task in tasks_to_process:
            try:
                current_data = task.data if hasattr(task, "data") else {}

                # Calculate new data state
                new_data = current_data.copy()
                has_changes = False
                changes_log = {}

                for key, value in request.update.items():
                    # Only update if value is different
                    if current_data.get(key) != value:
                        new_data[key] = value
                        has_changes = True
                        changes_log[key] = {"from": current_data.get(key), "to": value}

                if has_changes:
                    updates_needed.append(
                        {
                            "task_id": task.id,
                            "changes": changes_log,
                            "new_data": new_data,
                        }
                    )

            except Exception as e:
                errors.append(
                    f"Error processing task {getattr(task, 'id', 'unknown')}: {str(e)}"
                )

        summary = {
            "project_id": project_id,
            "total_tasks_found": len(tasks_to_process),
            "tasks_needing_updates": len(updates_needed),
            "errors_processing": len(errors),
            "dry_run": request.dry_run,
            "sample_updates": updates_needed[:3] if updates_needed else [],
            "errors": errors[:5] if errors else [],
        }

        if request.dry_run:
            summary["message"] = "DRY RUN - No changes made."
            return summary

        if not updates_needed:
            summary["message"] = (
                "No updates needed (all tasks already have target values)."
            )
            return summary

        # Apply updates
        success_count = 0
        update_errors = []

        for update in updates_needed:
            try:
                # IMPORTANT: We send the FULL data object to ensure we don't lose fields
                # if the LS API treats this as a replacement of the 'data' JSON column.
                ls.tasks.update(id=update["task_id"], data=update["new_data"])
                success_count += 1
            except Exception as e:
                update_errors.append(
                    f"Failed to update task {update['task_id']}: {str(e)}"
                )

        summary.update(
            {
                "message": f"Update complete: {success_count}/{len(updates_needed)} tasks updated.",
                "successful_updates": success_count,
                "failed_updates": len(update_errors),
                "update_errors": update_errors,
            }
        )

        return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update fields failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@tasks_router.post("/sync")
def sync_tasks(request: UpdateTasksRequest):
    """
    Complete sync: update existing task metadata AND add new tasks.
    """
    try:
        # First update existing
        update_result = sync_task_metadata(request)

        # Then add new
        add_request = AddTasksRequest(
            project_id=request.project_id, dry_run=request.dry_run
        )
        add_result = add_new_tasks(add_request)

        return {
            "message": "Complete sync finished",
            "update_summary": update_result,
            "add_new_summary": add_result,
            "total_operations": {
                "existing_tasks_updated": update_result.get("successful_updates", 0),
                "new_tasks_created": add_result.get("successful_creations", 0),
                "total_errors": update_result.get("errors_encountered", 0)
                + add_result.get("errors_encountered", 0),
            },
        }

    except Exception as e:
        logger.error(f"Sync tasks failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@tasks_router.get("/projects")
def list_projects():
    """List all available Label Studio projects."""
    ls = get_ls_client()

    try:
        projects = list(ls.projects.list())
        project_list = []

        for project in projects:
            project_list.append(
                {
                    "id": project.id,
                    "title": project.title,
                    "description": getattr(project, "description", ""),
                    "total_tasks": getattr(project, "task_number", 0),
                    "total_annotations": getattr(
                        project, "total_annotations_number", 0
                    ),
                    "created_at": str(getattr(project, "created_at", "")),
                    "updated_at": str(getattr(project, "updated_at", "")),
                }
            )

        return {"projects": project_list, "count": len(project_list)}

    except Exception as e:
        logger.error(f"Failed to list projects: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@tasks_router.post("/backup")
def create_backup(request: BackupRequest):
    """Create a backup of all tasks from a project."""
    ls = get_ls_client()

    try:
        project_id = get_or_default_project(ls, request.project_id)

        # Generate backup filename if not provided
        if not request.backup_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"project_{project_id}_backup_{timestamp}.json"
        else:
            backup_filename = request.backup_filename

        os.makedirs(request.backup_directory, exist_ok=True)
        backup_path = os.path.join(request.backup_directory, backup_filename)

        logger.info(f"Creating backup for project {project_id} to {backup_path}")

        tasks = list(ls.tasks.list(project=project_id))

        backup_data = {
            "metadata": {
                "project_id": project_id,
                "backup_created": datetime.now().isoformat(),
                "total_tasks": len(tasks),
                "include_annotations": request.include_annotations,
                "include_predictions": request.include_predictions,
            },
            "tasks": [],
        }

        processed_count = 0
        errors = []

        for task in tasks:
            try:
                task_backup = {
                    "id": task.id,
                    "data": task.data if hasattr(task, "data") else {},
                    "meta": getattr(task, "meta", {}),
                    "created_at": str(getattr(task, "created_at", "")),
                    "updated_at": str(getattr(task, "updated_at", "")),
                }

                if request.include_annotations and hasattr(task, "annotations"):
                    task_backup["annotations"] = []
                    for ann in getattr(task, "annotations", []):
                        task_backup["annotations"].append(
                            {
                                "id": getattr(ann, "id", None),
                                "result": getattr(ann, "result", []),
                                "created_at": str(getattr(ann, "created_at", "")),
                                "updated_at": str(getattr(ann, "updated_at", "")),
                                "created_by": getattr(ann, "created_by", None),
                                "completed_by": getattr(ann, "completed_by", None),
                                "was_cancelled": getattr(ann, "was_cancelled", False),
                            }
                        )

                if request.include_predictions and hasattr(task, "predictions"):
                    task_backup["predictions"] = []
                    for pred in getattr(task, "predictions", []):
                        task_backup["predictions"].append(
                            {
                                "id": getattr(pred, "id", None),
                                "result": getattr(pred, "result", []),
                                "score": getattr(pred, "score", None),
                                "created_at": str(getattr(pred, "created_at", "")),
                                "model_version": getattr(pred, "model_version", ""),
                            }
                        )

                backup_data["tasks"].append(task_backup)
                processed_count += 1

            except Exception as e:
                errors.append(f"Task {getattr(task, 'id', 'unknown')}: {str(e)}")

        with open(backup_path, "w") as f:
            json.dump(backup_data, f, indent=2, default=str)

        return {
            "message": "Backup created successfully",
            "project_id": project_id,
            "backup_path": backup_path,
            "backup_filename": backup_filename,
            "total_tasks": len(tasks),
            "successfully_backed_up": processed_count,
            "errors_encountered": len(errors),
            "backup_size_bytes": os.path.getsize(backup_path),
            "errors": errors[:5] if errors else [],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backup creation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@tasks_router.post("/restore")
def restore_from_backup(request: RestoreRequest):
    """Restore tasks from a backup file."""
    ls = get_ls_client()

    try:
        # Determine backup path
        if request.backup_path:
            backup_path = request.backup_path
        elif request.backup_filename:
            backup_path = os.path.join(
                request.backup_directory, request.backup_filename
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Either backup_path or backup_filename must be provided",
            )

        if not os.path.exists(backup_path):
            raise HTTPException(
                status_code=400, detail=f"Backup file not found: {backup_path}"
            )

        project_id = get_or_default_project(ls, request.project_id)

        # Load backup
        with open(backup_path, "r") as f:
            backup_data = json.load(f)

        backup_tasks = backup_data.get("tasks", [])
        logger.info(f"Loaded backup with {len(backup_tasks)} tasks")

        # Get existing task IDs
        existing_tasks = list(ls.tasks.list(project=project_id))
        existing_ids = {task.id for task in existing_tasks}

        summary = {
            "project_id": project_id,
            "backup_path": backup_path,
            "backup_tasks": len(backup_tasks),
            "existing_tasks": len(existing_ids),
            "restore_mode": request.restore_mode,
            "dry_run": request.dry_run,
        }

        if request.dry_run:
            # Count what would be restored
            to_restore = sum(
                1
                for t in backup_tasks
                if t.get("id") not in existing_ids
                or request.restore_mode != "skip_existing"
            )
            summary["message"] = f"DRY RUN - Would restore {to_restore} tasks"
            return summary

        # Restore tasks
        restored_count = 0
        skipped_count = 0
        errors = []

        for task_backup in backup_tasks:
            task_id = task_backup.get("id")

            if request.restore_mode == "skip_existing" and task_id in existing_ids:
                skipped_count += 1
                continue

            try:
                task_data = task_backup.get("data", {})

                if request.restore_mode == "overwrite" and task_id in existing_ids:
                    ls.tasks.update(id=task_id, data=task_data)
                else:
                    ls.tasks.create(project=project_id, data=task_data)

                restored_count += 1

            except Exception as e:
                errors.append(f"Task {task_id}: {str(e)}")

        summary.update(
            {
                "message": f"Restore complete: {restored_count} restored, {skipped_count} skipped",
                "restored": restored_count,
                "skipped": skipped_count,
                "errors": errors[:10] if errors else [],
            }
        )

        return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Restoration failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@tasks_router.get("/backups")
def list_backups(
    backup_directory: str = Query(default="/label-studio/data/label_studio_backups"),
):
    """List available backup files."""
    try:
        if not os.path.exists(backup_directory):
            return {
                "backup_directory": backup_directory,
                "backups": [],
                "count": 0,
                "message": "Backup directory does not exist",
            }

        backup_files = []

        for filename in os.listdir(backup_directory):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(backup_directory, filename)
            try:
                stat = os.stat(filepath)

                with open(filepath, "r") as f:
                    backup_data = json.load(f)
                    metadata = backup_data.get("metadata", {})

                backup_files.append(
                    {
                        "filename": filename,
                        "filepath": filepath,
                        "size_bytes": stat.st_size,
                        "size_mb": round(stat.st_size / 1024 / 1024, 2),
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "project_id": metadata.get("project_id"),
                        "total_tasks": metadata.get("total_tasks", 0),
                        "backup_created": metadata.get("backup_created"),
                        "include_annotations": metadata.get(
                            "include_annotations", False
                        ),
                        "include_predictions": metadata.get(
                            "include_predictions", False
                        ),
                    }
                )

            except Exception as e:
                backup_files.append(
                    {
                        "filename": filename,
                        "filepath": filepath,
                        "error": f"Could not read backup metadata: {str(e)}",
                    }
                )

        # Sort by creation time (newest first)
        backup_files.sort(key=lambda x: x.get("created", ""), reverse=True)

        return {
            "backup_directory": backup_directory,
            "backups": backup_files,
            "count": len(backup_files),
        }

    except Exception as e:
        logger.error(f"List backups failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@tasks_router.get("/debug-structure")
def debug_task_structure(project_id: Optional[int] = Query(default=None)):
    """Debug the structure of existing tasks to understand required fields."""
    ls = get_ls_client()

    try:
        project_id = get_or_default_project(ls, project_id)

        tasks = list(ls.tasks.list(project=project_id))[:3]

        if not tasks:
            return {"message": "No existing tasks found", "project_id": project_id}

        task_samples = []
        for task in tasks:
            task_samples.append(
                {
                    "task_id": task.id,
                    "data_keys": list(task.data.keys())
                    if hasattr(task, "data")
                    else [],
                    "data_sample": task.data if hasattr(task, "data") else {},
                    "meta": getattr(task, "meta", {}),
                    "has_annotations": hasattr(task, "annotations"),
                    "has_predictions": hasattr(task, "predictions"),
                }
            )

        # Get project label config
        try:
            project_info = ls.projects.get(id=project_id)
            label_config = getattr(project_info, "label_config", "")
        except Exception:
            label_config = "Could not retrieve label config"

        return {
            "project_id": project_id,
            "total_tasks_sampled": len(task_samples),
            "task_samples": task_samples,
            "label_config_preview": label_config[:500]
            if isinstance(label_config, str)
            else str(label_config)[:500],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Debug task structure failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# Grid Video Endpoints
# =====================


class GridVideoRequest(BaseModel):
    """Request model for creating a grid video from a single task."""

    layout: str = Field(
        default="3x3", description="Grid layout (3x3, 2x2, 2x4, 1x7, 1x2, 2x1)"
    )
    quality: str = Field(
        default="medium", description="Video quality (low, medium, high)"
    )
    output_path: Optional[str] = Field(
        default=None, description="Custom output path for the video"
    )
    use_cuda: bool = Field(
        default=True, description="Try to use CUDA hardware acceleration"
    )


class GridVideoConfigRequest(BaseModel):
    """Request model for batch grid video creation by config."""

    project_id: int = Field(..., description="Label Studio project ID")
    stable_id: str = Field(
        ..., description="Stable ID (e.g., 'stable01', 'stable02', 'stable03')"
    )
    date_from: str = Field(..., description="Start date (YYYY-MM-DD)")
    date_until: str = Field(..., description="End date (YYYY-MM-DD)")
    time_from: str = Field(default="00:00", description="Start time (HH:MM)")
    time_until: str = Field(default="23:59", description="End time (HH:MM)")
    layout: str = Field(
        default="2x2", description="Grid layout (3x3, 2x2, 2x4, 1x7, 1x2, 2x1)"
    )
    quality: str = Field(default="low", description="Video quality (low, medium, high)")
    time_threshold: int = Field(
        default=180, description="Time threshold in SECONDS for grouping videos"
    )
    dry_run: bool = Field(
        default=False, description="If True, only show what would be created"
    )
    use_cuda: bool = Field(
        default=True, description="Try to use CUDA hardware acceleration"
    )
    overwrite_existing: bool = Field(
        default=False, description="Overwrite existing grid videos"
    )


def get_video_info(video_path: str) -> Dict:
    """Get video information using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception as e:
        logger.warning(f"Could not get video info for {video_path}: {e}")
    return {}


def get_quality_settings(quality: str) -> Dict:
    """Get FFmpeg quality settings based on quality level."""
    settings = {
        "low": {
            "crf": "28",
            "preset": "ultrafast",
            "scale_factor": 0.5,
            "bitrate": "1M",
        },
        "medium": {
            "crf": "23",
            "preset": "medium",
            "scale_factor": 0.75,
            "bitrate": "2M",
        },
        "high": {"crf": "18", "preset": "slow", "scale_factor": 1.0, "bitrate": "4M"},
    }
    return settings.get(quality, settings["medium"])


def get_grid_layout_params(layout: str, num_videos: int) -> Dict:
    """Get grid layout parameters."""
    layouts = {
        "3x3": {"cols": 3, "rows": 3, "max_videos": 9},
        "2x2": {"cols": 2, "rows": 2, "max_videos": 4},
        "2x4": {"cols": 4, "rows": 2, "max_videos": 8},
        "1x7": {"cols": 7, "rows": 1, "max_videos": 7},
        "1x2": {"cols": 2, "rows": 1, "max_videos": 2},
        "2x1": {"cols": 1, "rows": 2, "max_videos": 2},
    }

    layout_params = layouts.get(layout, layouts["3x3"])
    actual_videos = min(num_videos, layout_params["max_videos"])

    return {**layout_params, "actual_videos": actual_videos}


def build_ffmpeg_grid_command(
    input_files: List[str],
    output_path: str,
    layout: str,
    quality: str,
    use_cuda: bool = True,
) -> Tuple[List[str], bool]:
    """
    Build FFmpeg command for creating a grid video.
    Returns (command, using_cuda).
    """
    quality_settings = get_quality_settings(quality)
    layout_params = get_grid_layout_params(layout, len(input_files))

    cols = layout_params["cols"]
    rows = layout_params["rows"]
    actual_videos = layout_params["actual_videos"]

    # Get video dimensions from first file
    video_info = get_video_info(input_files[0])
    try:
        stream = next(
            (
                s
                for s in video_info.get("streams", [])
                if s.get("codec_type") == "video"
            ),
            None,
        )
        if stream:
            src_width = int(stream.get("width", 1920))
            src_height = int(stream.get("height", 1080))
        else:
            src_width, src_height = 1920, 1080
    except:
        src_width, src_height = 1920, 1080

    # Calculate cell dimensions
    scale_factor = quality_settings["scale_factor"]
    cell_width = int(src_width * scale_factor)
    cell_height = int(src_height * scale_factor)

    # Ensure dimensions are even (required by many codecs)
    cell_width = cell_width - (cell_width % 2)
    cell_height = cell_height - (cell_height % 2)

    output_width = cell_width * cols
    output_height = cell_height * rows

    # Build input arguments
    cmd = ["ffmpeg", "-y"]  # -y to overwrite output

    # Add CUDA hardware acceleration if requested
    using_cuda = False
    if use_cuda:
        # Check if CUDA is available
        try:
            check_cmd = ["ffmpeg", "-hwaccels"]
            result = subprocess.run(
                check_cmd, capture_output=True, text=True, timeout=5
            )
            if "cuda" in result.stdout.lower():
                cmd.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
                using_cuda = True
        except:
            pass

    # Add input files
    for input_file in input_files[:actual_videos]:
        cmd.extend(["-i", input_file])

    # Build filter complex for grid
    filter_parts = []

    # Scale each input
    for i in range(actual_videos):
        if using_cuda:
            filter_parts.append(
                f"[{i}:v]scale_cuda={cell_width}:{cell_height}:format=yuv420p[v{i}]"
            )
        else:
            filter_parts.append(
                f"[{i}:v]scale={cell_width}:{cell_height}:force_original_aspect_ratio=decrease,pad={cell_width}:{cell_height}:(ow-iw)/2:(oh-ih)/2[v{i}]"
            )

    # Build xstack layout
    if actual_videos > 1:
        input_labels = "".join([f"[v{i}]" for i in range(actual_videos)])

        # Calculate positions for each cell
        positions = []
        for i in range(actual_videos):
            row = i // cols
            col = i % cols
            x = col * cell_width
            y = row * cell_height
            positions.append(f"{x}_{y}")

        layout_str = "|".join(positions)
        filter_parts.append(
            f"{input_labels}xstack=inputs={actual_videos}:layout={layout_str}[vout]"
        )
    else:
        filter_parts.append("[v0]null[vout]")

    filter_complex = ";".join(filter_parts)
    cmd.extend(["-filter_complex", filter_complex])

    # Add output options
    cmd.extend(["-map", "[vout]"])

    if using_cuda:
        cmd.extend(
            [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",  # NVENC preset
                "-b:v",
                quality_settings["bitrate"],
            ]
        )
    else:
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                quality_settings["preset"],
                "-crf",
                quality_settings["crf"],
            ]
        )

    cmd.extend(["-pix_fmt", "yuv420p", "-movflags", "+faststart", output_path])

    return cmd, using_cuda


def create_grid_video_from_task_id(
    task_id: int,
    project_id: int,
    video_directory: str = "/label-studio/data/video/",
    output_directory: str = "/label-studio/data/grid_videos/",
    layout: str = "2x2",
    quality: str = "low",
    use_cuda: bool = True,
    time_threshold: int = 180,
    overwrite_existing: bool = False,
    output_filename_override: Optional[str] = None,
    all_video_files: Optional[List[str]] = None,
) -> Dict:
    """
    Create a grid video combining all stall videos for a 30-minute window.

    Finds all videos from the same stable within time_threshold seconds of the
    main video and combines them into a grid layout.

    Args:
        task_id: The task to get the main video from
        project_id: Label Studio project ID
        video_directory: Where to find source videos
        output_directory: Where to save grid videos (flat directory)
        layout: Grid layout (2x2, 3x3, 2x4, etc.)
        quality: Video quality (low, medium, high)
        use_cuda: Whether to try CUDA acceleration
        time_threshold: Max seconds difference to include videos (default 180s = 3min)
        overwrite_existing: Whether to overwrite existing grid videos
        output_filename_override: Override the output filename
        all_video_files: Pre-listed video files to avoid repeated os.listdir

    Returns a dict with:
    - success: bool
    - output_path: str (if successful)
    - grid_video_url: str (Label Studio URL format)
    - error: str (if failed)
    """
    import time

    start_time = time.time()

    ls = get_ls_client()

    try:
        # Get task
        task = ls.tasks.get(id=str(task_id))
        task_data: Dict[str, Any] = (
            task.data if hasattr(task, "data") and task.data else {}
        )

        # Get main video URL from task
        main_video_url = task_data.get("video", "")
        if not main_video_url:
            return {
                "success": False,
                "error": "No main video URL found in task data",
                "task_id": task_id,
            }

        # Extract filename from URL
        main_video_filename = main_video_url.split("/")[-1]
        main_video_path = os.path.join(video_directory, main_video_filename)

        logger.info(f"Main video: {main_video_filename}")

        if not os.path.exists(main_video_path):
            return {
                "success": False,
                "error": f"Main video file not found: {main_video_path}",
                "task_id": task_id,
            }

        # Parse stable and timestamp from filename
        # Example: stable01_stall05_horse05_20250817_003014_cam.mp4
        try:
            parts = main_video_filename.split("_")
            main_stable_prefix = parts[0]  # e.g., 'stable01'
            main_video_date = parts[3]  # e.g., '20250817'
            main_video_time = parts[4]  # e.g., '003014'
            main_video_timestamp = datetime.strptime(
                f"{main_video_date}_{main_video_time}", "%Y%m%d_%H%M%S"
            )
        except (IndexError, ValueError) as e:
            return {
                "success": False,
                "error": f"Could not parse timestamp from filename {main_video_filename}: {e}",
                "task_id": task_id,
            }

        # Get list of all video files
        if all_video_files is None:
            all_video_files = os.listdir(video_directory)

        # Find videos from the same stable within time threshold
        video_paths = {main_video_filename: main_video_path}

        stall_video_pattern = re.compile(
            r"stable\d{2}_stall\d{2}_horse\d{2}_\d{8}_\d{6}_cam\.mp4"
        )
        stall_video_files = [
            f
            for f in all_video_files
            if f.startswith(main_stable_prefix + "_") and stall_video_pattern.match(f)
        ]

        for video_file in stall_video_files:
            if video_file == main_video_filename:
                continue  # Already added
            try:
                parts = video_file.split("_")
                video_date = parts[3]
                video_time = parts[4]
                video_timestamp = datetime.strptime(
                    f"{video_date}_{video_time}", "%Y%m%d_%H%M%S"
                )

                if (
                    abs((video_timestamp - main_video_timestamp).total_seconds())
                    <= time_threshold
                ):
                    logger.info(f"Found matching stall video: {video_file}")
                    video_paths[video_file] = os.path.join(video_directory, video_file)
            except (IndexError, ValueError):
                continue  # Skip files with invalid format

        logger.info(f"Found {len(video_paths)} videos for grid")

        if len(video_paths) < 2:
            return {
                "success": False,
                "error": f"Not enough videos found. Need at least 2, found {len(video_paths)}",
                "task_id": task_id,
                "available_videos": list(video_paths.keys()),
            }

        # Generate output filename
        if output_filename_override:
            output_filename = output_filename_override
        else:
            # Format: stable01_20250827_210003_grid_2by2.mp4
            layout_str = layout.replace("x", "by")
            output_filename = f"{main_stable_prefix}_{main_video_date}_{main_video_time}_grid_{layout_str}.mp4"

        # Output directory is flat (no subfolders)
        os.makedirs(output_directory, exist_ok=True)
        output_path = os.path.join(output_directory, output_filename)
        grid_video_url = (
            f"/data/local-files/?d=label-studio/data/grid_videos/{output_filename}"
        )

        # Check if file already exists
        if os.path.exists(output_path) and not overwrite_existing:
            logger.info(f"Grid video already exists: {output_path}")
            return {
                "success": True,
                "output_path": output_path,
                "grid_video_url": grid_video_url,
                "already_existed": True,
                "task_id": task_id,
                "input_videos": len(video_paths),
            }

        # Build and run FFmpeg command
        video_path_list = list(video_paths.values())
        cmd, using_cuda = build_ffmpeg_grid_command(
            video_path_list, output_path, layout, quality, use_cuda
        )

        logger.info(
            f"Creating grid video with {len(video_path_list)} inputs -> {output_path}"
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout for long videos
            )

            if result.returncode != 0:
                # If CUDA failed, try CPU fallback
                if using_cuda and use_cuda:
                    logger.warning("CUDA encoding failed, falling back to CPU")
                    cmd, using_cuda = build_ffmpeg_grid_command(
                        video_path_list, output_path, layout, quality, use_cuda=False
                    )
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=1800
                    )

                if result.returncode != 0:
                    stderr = result.stderr
                    error_msg = stderr[-1000:] if len(stderr) > 1000 else stderr
                    return {
                        "success": False,
                        "error": f"FFmpeg failed: {error_msg}",
                        "task_id": task_id,
                    }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "FFmpeg timeout (>30 minutes)",
                "task_id": task_id,
            }

        duration = time.time() - start_time

        # Verify output file exists
        if not os.path.exists(output_path):
            return {
                "success": False,
                "error": "Output file was not created",
                "task_id": task_id,
            }

        return {
            "success": True,
            "output_path": output_path,
            "grid_video_url": grid_video_url,
            "output_filename": output_filename,
            "duration": round(duration, 2),
            "task_id": task_id,
            "input_videos": len(video_path_list),
            "video_files": list(video_paths.keys()),
            "layout": layout,
            "quality": quality,
            "used_cuda": using_cuda,
        }

    except Exception as e:
        logger.error(
            f"Error creating grid video for task {task_id}: {e}", exc_info=True
        )
        return {"success": False, "error": str(e), "task_id": task_id}


def generate_canonical_grid_filename(
    stable_id: str, date_compact: str, time_compact: str, layout: str
) -> str:
    """
    Generate a canonical filename for a grid video.

    Format: stable01_20250827_210003_grid_2by2.mp4

    Args:
        stable_id: e.g., 'stable01'
        date_compact: e.g., '20250827' (YYYYMMDD)
        time_compact: e.g., '210003' (HHMMSS)
        layout: e.g., '2x2' or '2by2'
    """
    layout_str = layout.replace("x", "by")
    return f"{stable_id}_{date_compact}_{time_compact}_grid_{layout_str}.mp4"


@tasks_router.post("/grid-video/{task_id}")
async def create_grid_video_endpoint(
    task_id: int,
    request: GridVideoRequest,
    project_id: int = Query(..., description="Label Studio project ID"),
):
    """
    Create a grid video from a single task's videos.

    The task should contain multiple video files that will be combined into a grid layout.
    """
    result = create_grid_video_from_task_id(
        task_id=task_id,
        project_id=project_id,
        layout=request.layout,
        quality=request.quality,
        use_cuda=request.use_cuda,
    )

    if not result["success"]:
        raise HTTPException(
            status_code=500, detail=result.get("error", "Failed to create grid video")
        )

    return result


@tasks_router.post("/grid-video-by-config")
async def create_grid_videos_by_config(request: GridVideoConfigRequest):
    """
    Create grid videos for multiple tasks based on configuration.

    This endpoint:
    1. Finds tasks for the specified stable within the date/time range
    2. Groups them by time proximity (using time_threshold in seconds)
    3. For each group, creates ONE grid video by finding all stall videos in the video directory
    4. Updates all tasks in the group with the grid_video URL
    """
    from ml_backend.smartstable_core.task_filtering import (
        list_tasks_by_recording_period,
    )

    ls = get_ls_client()

    video_directory = "/label-studio/data/video/"
    output_directory = "/label-studio/data/grid_videos/"

    try:
        # Get tasks for this stable in date range
        tasks_meta = list_tasks_by_recording_period(
            project_id=request.project_id,
            stable_id=request.stable_id,
            date_from=request.date_from,
            date_until=request.date_until,
            time_from=request.time_from,
            time_until=request.time_until,
            return_summary=False,
            ls_client=ls,
        )

        if not tasks_meta:
            return {
                "success": True,
                "message": "No tasks found for window",
                "project_id": request.project_id,
                "stable_id": request.stable_id,
                "date_from": request.date_from,
                "date_until": request.date_until,
            }

        # Get list of video files once
        try:
            all_video_files = os.listdir(video_directory)
        except FileNotFoundError:
            raise HTTPException(
                status_code=400, detail=f"Video directory not found: {video_directory}"
            )

        processed = []
        skipped = []
        failures = []

        # Build candidate list with parsed timestamps
        candidates = []
        for tm in tasks_meta:
            try:
                # Handle both dict and Task object forms
                if isinstance(tm, dict):
                    task_id_val = tm.get("id")
                    audio_url = tm.get("audio") or ""
                    rd = tm.get("recorded_date")
                    rt = tm.get("recorded_time", "00:00:00")
                else:
                    # It's a Task object
                    task_id_val = getattr(tm, "id", None)
                    task_data = tm.data if hasattr(tm, "data") and tm.data else {}
                    audio_url = (
                        task_data.get("audio", "")
                        if isinstance(task_data, dict)
                        else ""
                    )
                    rd = (
                        task_data.get("recorded_date")
                        if isinstance(task_data, dict)
                        else None
                    )
                    rt = (
                        task_data.get("recorded_time", "00:00:00")
                        if isinstance(task_data, dict)
                        else "00:00:00"
                    )

                if not audio_url or not task_id_val:
                    skipped.append(
                        {
                            "task_id": task_id_val or "unknown",
                            "reason": "missing_audio_or_id",
                        }
                    )
                    continue

                audio_filename = audio_url.split("/")[-1]

                # Parse datetime from recorded_date/recorded_time
                if not rd:
                    skipped.append(
                        {"task_id": task_id_val, "reason": "missing_recorded_date"}
                    )
                    continue

                try:
                    dt = datetime.strptime(f"{rd} {rt}", "%Y-%m-%d %H:%M:%S")
                except Exception:
                    try:
                        dt = datetime.strptime(rd, "%Y-%m-%d")
                    except Exception:
                        skipped.append(
                            {"task_id": task_id_val, "reason": "bad_datetime"}
                        )
                        continue

                candidates.append(
                    {
                        "task_id": task_id_val,
                        "audio_filename": audio_filename,
                        "dt": dt,
                        "task_data": tm,
                    }
                )
            except Exception as e:
                task_id_for_error = (
                    getattr(tm, "id", None)
                    if hasattr(tm, "id")
                    else (
                        tm.get("id", "unknown") if isinstance(tm, dict) else "unknown"
                    )
                )
                failures.append({"task_id": task_id_for_error, "error": str(e)})

        # Sort by time
        candidates.sort(key=lambda c: c["dt"])

        # Group by time_threshold windows (in seconds)
        assigned = set()
        groups = []
        for idx, c in enumerate(candidates):
            task_id_val = c.get("task_id")
            if task_id_val in assigned:
                continue
            anchor_dt = c["dt"]
            group = {
                "anchor": c,
                "members": [c],
            }
            assigned.add(task_id_val)

            # Pull in all subsequent tasks within threshold of anchor
            for j in range(idx + 1, len(candidates)):
                cj = candidates[j]
                tj_id = cj.get("task_id")
                if tj_id in assigned:
                    continue
                if (
                    abs((cj["dt"] - anchor_dt).total_seconds())
                    <= request.time_threshold
                ):
                    group["members"].append(cj)
                    assigned.add(tj_id)
                else:
                    break
            groups.append(group)

        logger.info(f"Found {len(groups)} task groups for grid video creation")

        # Dry run - just return what would be created
        if request.dry_run:
            group_info = []
            for i, group in enumerate(groups):
                anchor = group["anchor"]
                members = group["members"]
                anchor_dt = anchor["dt"]
                date_compact = anchor_dt.strftime("%Y%m%d")
                time_compact = anchor_dt.strftime("%H%M%S")

                filename = generate_canonical_grid_filename(
                    stable_id=request.stable_id,
                    date_compact=date_compact,
                    time_compact=time_compact,
                    layout=request.layout,
                )

                group_info.append(
                    {
                        "group_index": i,
                        "num_tasks": len(members),
                        "task_ids": [m["task_id"] for m in members],
                        "anchor_time": anchor_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "proposed_filename": filename,
                    }
                )

            return {
                "success": True,
                "dry_run": True,
                "message": "DRY RUN - No grid videos created",
                "num_groups": len(groups),
                "total_candidates": len(candidates),
                "skipped": skipped[:20],
                "groups": group_info,
            }

        # Process groups: create grid video once per group, then link all members
        for group in groups:
            anchor = group["anchor"]
            members = group["members"]
            anchor_dt = anchor["dt"]
            date_compact = anchor_dt.strftime("%Y%m%d")
            time_compact = anchor_dt.strftime("%H%M%S")

            canonical_grid_filename = generate_canonical_grid_filename(
                stable_id=request.stable_id,
                date_compact=date_compact,
                time_compact=time_compact,
                layout=request.layout,
            )
            canonical_grid_path = os.path.join(
                output_directory, canonical_grid_filename
            )
            grid_url = f"/data/local-files/?d=label-studio/data/grid_videos/{canonical_grid_filename}"

            # Find a creator task that has a valid main video file
            creator = None
            for m in members:
                vf = (
                    m["audio_filename"].replace(".flac", ".mp4").replace("_mic", "_cam")
                )
                if vf in all_video_files:
                    creator = m
                    break

            if not creator:
                for m in members:
                    skipped.append(
                        {
                            "task_id": m.get("task_id", "unknown"),
                            "reason": "no_main_video_for_group",
                        }
                    )
                continue

            # If file exists and not overwriting, just link all members
            if os.path.exists(canonical_grid_path) and not request.overwrite_existing:
                for m in members:
                    try:
                        task_obj = ls.tasks.get(m["task_id"])
                        current_url = (
                            task_obj.data.get("grid_video")
                            if hasattr(task_obj, "data")
                            else None
                        )
                        if current_url != grid_url:
                            new_data = (
                                dict(task_obj.data) if hasattr(task_obj, "data") else {}
                            )
                            new_data["grid_video"] = grid_url
                            ls.tasks.update(id=m["task_id"], data=new_data)
                        processed.append(
                            {
                                "task_id": m.get("task_id", "unknown"),
                                "action": "linked_existing",
                                "grid": canonical_grid_filename,
                            }
                        )
                    except Exception as e:
                        failures.append(
                            {
                                "task_id": m.get("task_id", "unknown"),
                                "error": f"link_failed: {str(e)}",
                            }
                        )
                continue

            # Create grid video using the creator task
            result = create_grid_video_from_task_id(
                task_id=creator["task_id"],
                project_id=request.project_id,
                video_directory=video_directory,
                output_directory=output_directory,
                layout=request.layout,
                quality=request.quality,
                use_cuda=request.use_cuda,
                time_threshold=request.time_threshold,
                overwrite_existing=request.overwrite_existing,
                output_filename_override=canonical_grid_filename,
                all_video_files=all_video_files,
            )

            if not result.get("success"):
                failures.append(
                    {
                        "task_id": creator["task_id"],
                        "error": result.get("error", "unknown error"),
                    }
                )
                continue

            # Link all members to the grid video
            for m in members:
                try:
                    task_obj = ls.tasks.get(m["task_id"])
                    new_data = dict(task_obj.data)
                    new_data["grid_video"] = grid_url
                    ls.tasks.update(id=m["task_id"], data=new_data)
                    processed.append(
                        {
                            "task_id": m["task_id"],
                            "action": "created_and_linked",
                            "grid": canonical_grid_filename,
                        }
                    )
                except Exception as e:
                    failures.append(
                        {
                            "task_id": m.get("task_id", "unknown"),
                            "error": f"update_failed: {str(e)}",
                        }
                    )

        return {
            "success": True,
            "message": "Grid video batch completed",
            "project_id": request.project_id,
            "stable_id": request.stable_id,
            "layout": request.layout,
            "quality": request.quality,
            "time_threshold_seconds": request.time_threshold,
            "overwrite_existing": request.overwrite_existing,
            "date_range": f"{request.date_from} to {request.date_until}",
            "processed_count": len(processed),
            "skipped_count": len(skipped),
            "failed_count": len(failures),
            "processed": processed[:50],
            "skipped": skipped[:50],
            "failures": failures[:50],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Grid video batch creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
