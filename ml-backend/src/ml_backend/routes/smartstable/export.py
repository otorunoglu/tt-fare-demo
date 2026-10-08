from datetime import date, datetime, time, timedelta
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time as pytime
import zipfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.background import BackgroundTask

from ml_backend.smartstable_core.mlflow_support import (
    download_test_pool_manifest,
    get_champion_tflite_bundle,
    get_model_version_info,
    get_production_model_info,
)

logger = logging.getLogger(__name__)

export_router = APIRouter()

DB_PATH = os.getenv("SMARTSTABLE_DB_PATH", "/app/data/smartstable/soundscape.db")
MAX_BATCH_DAYS = 366
RCLONE_TPSLIMIT = os.getenv("RCLONE_TPSLIMIT", "2")
RCLONE_TPSLIMIT_BURST = os.getenv("RCLONE_TPSLIMIT_BURST", "2")
RCLONE_RETRY_ATTEMPTS = int(os.getenv("RCLONE_RETRY_ATTEMPTS", "5"))
RCLONE_RETRY_BASE_DELAY_SECONDS = float(
    os.getenv("RCLONE_RETRY_BASE_DELAY_SECONDS", "2")
)
COVERAGE_MIN_VIDEO_BYTES = int(os.getenv("SMARTSTABLE_COVERAGE_MIN_VIDEO_BYTES", "1000000"))
COVERAGE_MIN_AUDIO_BYTES = int(os.getenv("SMARTSTABLE_COVERAGE_MIN_AUDIO_BYTES", "128000"))
COVERAGE_SHORT_SAMPLE_LIMIT = int(os.getenv("SMARTSTABLE_COVERAGE_SHORT_SAMPLE_LIMIT", "5"))


class BatchHeatmapRequest(BaseModel):
    date_from: date = Field(..., description="Start date (YYYY-MM-DD)")
    date_until: date = Field(..., description="End date (YYYY-MM-DD)")
    bin_size: int = Field(default=2, ge=1, le=60, description="Bin size in minutes")
    cap_min_per_bin: float = Field(
        default=1.0, gt=0.0, description="Cap heatmap cell values in minutes"
    )
    confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Threshold under which the prediction is ignored",
    )
    start_time: str = Field(default="19:00", description="Start time in HH:MM")
    end_time: str = Field(default="09:00", description="End time in HH:MM")
    group_labels: bool = Field(default=False)
    hide_label: list[str] = Field(default_factory=list)
    hide_empty_bins: bool = Field(default=True)

    @field_validator("start_time", "end_time")
    @classmethod
    def validate_hhmm(cls, value: str) -> str:
        _parse_hhmm(value)
        return value

    @model_validator(mode="after")
    def validate_date_range(self) -> "BatchHeatmapRequest":
        if self.date_until < self.date_from:
            raise ValueError("date_until must be on or after date_from")
        if (self.date_until - self.date_from).days + 1 > MAX_BATCH_DAYS:
            raise ValueError(f"Date range too large; maximum is {MAX_BATCH_DAYS} days")
        return self


class BatchLabelSumRequest(BaseModel):
    date_from: date = Field(..., description="Start date (YYYY-MM-DD)")
    date_until: date = Field(..., description="End date (YYYY-MM-DD)")
    start_time: str = Field(default="19:00", description="Start time in HH:MM")
    end_time: str = Field(default="09:00", description="End time in HH:MM")
    confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Threshold under which the prediction is ignored",
    )
    format_as_xlsx: bool = Field(
        default=False,
        description="If true, return XLSX when possible; otherwise return CSV",
    )
    sum_labels: list[str] = Field(..., description="List of labels to sum")

    @field_validator("start_time", "end_time")
    @classmethod
    def validate_hhmm(cls, value: str) -> str:
        _parse_hhmm(value)
        return value

    @field_validator("sum_labels")
    @classmethod
    def validate_sum_labels(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("sum_labels must contain at least one label")
        return [label.strip() for label in value if label.strip()]

    @model_validator(mode="after")
    def validate_date_range(self) -> "BatchLabelSumRequest":
        if self.date_until < self.date_from:
            raise ValueError("date_until must be on or after date_from")
        if (self.date_until - self.date_from).days + 1 > MAX_BATCH_DAYS:
            raise ValueError(f"Date range too large; maximum is {MAX_BATCH_DAYS} days")
        return self


class LabelDurationInfo(BaseModel):
    seconds: float = Field(..., description="Total duration in seconds")
    display: str = Field(..., description="Human-readable duration (e.g., 2h 20m 4s)")


class NightlyLabelSumItem(BaseModel):
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    time_window: str = Field(..., description="Time window (e.g., 19:00-09:00)")
    label_sums: dict[str, LabelDurationInfo] = Field(
        ..., description="Per-label total durations"
    )


class TestSetBundleRequest(BaseModel):
    model_version: str = Field(..., description="Model version, e.g. 'v29' or '29'")
    model_architecture: str = Field(
        default="panns", description="Model architecture ('panns' or 'yamnet')"
    )
    snippet_duration_seconds: float = Field(
        default=4.0,
        gt=0.1,
        le=30.0,
        description="Duration of each extracted snippet in seconds",
    )

    @field_validator("model_version")
    @classmethod
    def validate_model_version(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("model_version must not be empty")
        return cleaned

    @field_validator("model_architecture")
    @classmethod
    def validate_model_architecture(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in {"panns", "yamnet"}:
            raise ValueError("model_architecture must be 'panns' or 'yamnet'")
        return cleaned


class PublishChampionOnnxRequest(BaseModel):
    model_architecture: str = Field(
        default="panns", description="Model architecture ('panns' or 'yamnet')"
    )
    remote: str | None = Field(
        default=None,
        description="rclone remote:path target (defaults to RCLONE_WEIGHTS_REMOTE or model_weights:weights)",
    )
    target_name: str | None = Field(
        default=None,
        description="Optional target filename on remote (defaults to generated zip name)",
    )
    publish_latest_alias: bool = Field(
        default=True,
        description="Also upload a latest alias file for easy edge syncing",
    )

    @field_validator("model_architecture")
    @classmethod
    def validate_model_architecture(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in {"panns", "yamnet"}:
            raise ValueError("model_architecture must be 'panns' or 'yamnet'")
        return cleaned


class VideoAndAudioExportRequest(BaseModel):
    project_id: int = Field(..., description="Label Studio project ID")
    date_from: date = Field(..., description="Start date (YYYY-MM-DD)")
    date_until: date = Field(..., description="End date (YYYY-MM-DD)")
    stable_id: str = Field(..., description="Stable ID in filename, e.g. stable03")
    stall_id: str = Field(..., description="Stall ID in filename, e.g. stall03")
    max_tasks: int = Field(
        default=10000,
        ge=1,
        le=100000,
        description="Maximum number of tasks/files to process",
    )

    @model_validator(mode="after")
    def validate_date_range(self) -> "VideoAndAudioExportRequest":
        if self.date_until < self.date_from:
            raise ValueError("date_until must be on or after date_from")
        if (self.date_until - self.date_from).days + 1 > MAX_BATCH_DAYS:
            raise ValueError(f"Date range too large; maximum is {MAX_BATCH_DAYS} days")
        return self


def _parse_hhmm(value: str) -> time:
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:
        raise ValueError(f"Invalid time '{value}', expected HH:MM") from exc


def _format_seconds(total_seconds: float) -> str:
    """
    Convert total seconds into a human-readable duration string (e.g., 2h 20m 4s).
    """
    total_sec = int(total_seconds)
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    seconds = total_sec % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")
    
    return " ".join(parts)


def _format_hhmmss(total_seconds: float) -> str:
    total_sec = max(0, int(total_seconds))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    seconds = total_sec % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _daily_window(day: date, start_clock: time, end_clock: time) -> tuple[datetime, datetime]:
    start_dt = datetime.combine(day, start_clock)
    end_dt = datetime.combine(day, end_clock)
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)
    return start_dt, end_dt


def _normalize_model_version(model_version: str) -> str:
    cleaned = model_version.strip()
    return cleaned[1:] if cleaned.lower().startswith("v") else cleaned


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())


def _parse_manifest_sample_id(sample_id: str) -> tuple[str, float] | None:
    if "#" not in sample_id:
        return None

    filename, offset_str = sample_id.rsplit("#", 1)
    try:
        return filename, float(offset_str)
    except ValueError:
        return None


def _scan_audio_files() -> dict[str, str]:
    search_dirs_raw = os.getenv(
        "SMARTSTABLE_AUDIO_SEARCH_DIRS", "/label-studio/data/audio,/label-studio/data"
    )
    search_dirs = [p.strip() for p in search_dirs_raw.split(",") if p.strip()]

    filename_to_path: dict[str, str] = {}
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue

        for root, _, files in os.walk(search_dir):
            for filename in files:
                lower = filename.lower()
                if lower.endswith((".wav", ".flac", ".mp3", ".ogg", ".m4a")):
                    filename_to_path[filename] = os.path.join(root, filename)

    return filename_to_path


def _probe_media_duration(media_path: str) -> float | None:
    """Return media duration in seconds using ffprobe, or None if unavailable."""
    if not media_path or not os.path.exists(media_path):
        return None

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                media_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None

        raw_value = result.stdout.strip()
        return float(raw_value) if raw_value else None
    except Exception:
        return None


def _get_file_size(media_path: str) -> int | None:
    if not media_path or not os.path.exists(media_path):
        return None
    try:
        return os.path.getsize(media_path)
    except OSError:
        return None


def _is_drive_rate_limit(stderr: str, stdout: str) -> bool:
    combined = f"{stderr}\n{stdout}".lower()
    return (
        "ratelimitexceeded" in combined
        or "rate_limit_exceeded" in combined
        or "quota exceeded" in combined
        or "user rate limit exceeded" in combined
    )


def _rclone_copyto_with_retry(source_path: str, destination_path: str) -> None:
    """
    Copy a file with rclone and retry on rate-limit errors using exponential backoff.
    """
    attempts = max(1, RCLONE_RETRY_ATTEMPTS)
    delay = max(0.1, RCLONE_RETRY_BASE_DELAY_SECONDS)

    for attempt in range(1, attempts + 1):
        cmd = [
            "rclone",
            "copyto",
            source_path,
            destination_path,
            "--tpslimit",
            RCLONE_TPSLIMIT,
            "--tpslimit-burst",
            RCLONE_TPSLIMIT_BURST,
            "--retries",
            "1",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return

        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        if _is_drive_rate_limit(stderr, stdout):
            if attempt >= attempts:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "Google Drive API rate limit exceeded during upload. "
                        f"Tried {attempts} time(s). Last error: {stderr or stdout}"
                    ),
                )

            sleep_seconds = delay * (2 ** (attempt - 1))
            logger.warning(
                "rclone rate-limited uploading %s (attempt %s/%s). Retrying in %.1fs",
                destination_path,
                attempt,
                attempts,
                sleep_seconds,
            )
            pytime.sleep(sleep_seconds)
            continue

        raise HTTPException(
            status_code=500,
            detail=f"rclone upload failed: {stderr or stdout}",
        )


def _extract_snippet_to_wav(
    source_path: str,
    offset_seconds: float,
    duration_seconds: float,
    output_path: str,
) -> tuple[bool, str | None, float]:
    try:
        import soundfile as sf

        with sf.SoundFile(source_path) as snd:
            samplerate = int(snd.samplerate)
            total_frames = int(snd.frames)

            start_frame = max(0, int(offset_seconds * samplerate))
            num_frames = max(1, int(duration_seconds * samplerate))

            if start_frame >= total_frames:
                return (
                    False,
                    "offset is outside source audio",
                    0.0,
                )

            snd.seek(start_frame)
            audio = snd.read(frames=num_frames, dtype="float32", always_2d=False)

            if audio is None or getattr(audio, "size", 0) == 0:
                return False, "empty snippet", 0.0

            actual_duration = float(len(audio) / samplerate)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio, samplerate)

            return True, None, actual_duration
    except Exception as e:
        return False, str(e), 0.0


def _to_half_hour_slot(ts: datetime) -> datetime:
    minute = 30 if ts.minute >= 30 else 0
    return ts.replace(minute=minute, second=0, microsecond=0)


def _night_slots_for_day(day: date) -> list[datetime]:
    slots: list[datetime] = []
    current = datetime.combine(day, time(hour=19, minute=0))
    end = datetime.combine(day + timedelta(days=1), time(hour=9, minute=0))
    while current < end:
        slots.append(current)
        current += timedelta(minutes=30)
    return slots


@export_router.get("/champion/info")
async def get_champion_info():
    """
    Get information about the current champion model (version, run_id, etc.)
    """
    try:
        info = get_production_model_info()
        if info is None:
            raise HTTPException(status_code=404, detail="No champion model found")
        return info
    except Exception as e:
        logger.error(f"Error in get_champion_info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@export_router.get("/champion/tflite")
async def export_champion_tflite():
    """
    Download the champion model in TFLite format bundled with its label mapping.
    """
    try:
        zip_path = get_champion_tflite_bundle()
        if not zip_path or not os.path.exists(zip_path):
            raise HTTPException(
                status_code=404,
                detail="Champion model bundle could not be created or found",
            )

        filename = os.path.basename(zip_path)
        return FileResponse(
            path=zip_path, filename=filename, media_type="application/zip"
        )
    except Exception as e:
        logger.error(f"Error in export_champion_tflite: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@export_router.get("/champion/onnx")
async def export_champion_onnx():
    """
    Download the champion model in ONNX format bundled with its label mapping.
    """
    try:
        from ml_backend.smartstable_core.mlflow_support import get_champion_onnx_bundle

        zip_path = get_champion_onnx_bundle()
        if not zip_path or not os.path.exists(zip_path):
            raise HTTPException(
                status_code=404,
                detail="Champion ONNX model bundle could not be created or found",
            )

        filename = os.path.basename(zip_path)
        return FileResponse(
            path=zip_path, filename=filename, media_type="application/zip"
        )
    except Exception as e:
        logger.error(f"Error in export_champion_onnx: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@export_router.get("/champion/onnx/cache")
async def export_champion_onnx_to_cache():
    """
    Build the champion ONNX bundle and save it to the model cache folder.

    The resulting file is written as `stable_campion_onnx.zip`.
    """
    try:
        from ml_backend.smartstable_core.mlflow_support import get_champion_onnx_bundle

        zip_path = get_champion_onnx_bundle()
        if not zip_path or not os.path.exists(zip_path):
            raise HTTPException(
                status_code=404,
                detail="Champion ONNX model bundle could not be created or found",
            )

        model_cache_dir = os.getenv("SMARTSTABLE_MODELS_DIR", "./data/model_cache")
        os.makedirs(model_cache_dir, exist_ok=True)

        target_filename = "stable_campion_onnx.zip"
        target_path = os.path.join(model_cache_dir, target_filename)
        shutil.copy2(zip_path, target_path)

        return {
            "status": "ok",
            "source_bundle": os.path.basename(zip_path),
            "saved_filename": target_filename,
            "saved_path": os.path.abspath(target_path),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in export_champion_onnx_to_cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@export_router.post("/champion/onnx/publish")
async def publish_champion_onnx(request: PublishChampionOnnxRequest):
    """
    Build the champion ONNX bundle and upload the resulting ZIP to rclone remote storage.
    """
    try:
        from ml_backend.smartstable_core.mlflow_support import get_champion_onnx_bundle

        zip_path = get_champion_onnx_bundle(model_architecture=request.model_architecture)
        if not zip_path or not os.path.exists(zip_path):
            raise HTTPException(
                status_code=404,
                detail="Champion ONNX model bundle could not be created or found",
            )

        remote = (request.remote or os.getenv("RCLONE_WEIGHTS_REMOTE") or "model_weights:weights").strip()
        if not remote:
            raise HTTPException(status_code=400, detail="Remote target must not be empty")

        generated_name = os.path.basename(zip_path)
        target_name = os.path.basename(request.target_name) if request.target_name else generated_name
        remote_path = f"{remote.rstrip('/')}/{target_name}"

        logger.info("Uploading champion ONNX bundle via rclone to %s", remote_path)
        _rclone_copyto_with_retry(zip_path, remote_path)

        latest_alias_path = None
        if request.publish_latest_alias:
            latest_alias_name = os.getenv("RCLONE_LATEST_ONNX_NAME", "latest_onnx.zip").strip() or "latest_onnx.zip"
            latest_alias_name = os.path.basename(latest_alias_name)
            latest_alias_path = f"{remote.rstrip('/')}/{latest_alias_name}"
            logger.info("Uploading latest ONNX alias via rclone to %s", latest_alias_path)
            _rclone_copyto_with_retry(zip_path, latest_alias_path)

        return {
            "status": "ok",
            "model_architecture": request.model_architecture,
            "generated_bundle": generated_name,
            "uploaded_path": remote_path,
            "latest_alias_path": latest_alias_path,
        }
    except HTTPException:
        raise
    except FileNotFoundError as e:
        if str(e).find("rclone") != -1:
            raise HTTPException(status_code=500, detail="rclone executable not found") from e
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error in publish_champion_onnx: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@export_router.post("/export/heatmap/batch")
async def export_batch_heatmaps(request: BatchHeatmapRequest):
    """
    Batch render one heatmap image per day and return them in a ZIP archive.
    """
    try:
        from smartstablemodel.config import load_metamodel_config
        from smartstablemodel.services.data_store import DataStore
        from smartstablemodel.services.visualizer import Visualizer

        start_clock = _parse_hhmm(request.start_time)
        end_clock = _parse_hhmm(request.end_time)

        ds = DataStore(DB_PATH)
        viz = Visualizer(ds)
        cfg = load_metamodel_config()
        configured_groups = getattr(cfg, "label_groups", None)

        hide_set = {label.strip() for label in request.hide_label if label.strip()}

        tmp_dir = tempfile.mkdtemp(prefix="batch_heatmap_")
        zip_filename = (
            f"heatmaps_{request.date_from.isoformat()}_{request.date_until.isoformat()}.zip"
        )
        zip_path = os.path.join(tmp_dir, zip_filename)

        # ── Pass 1: collect raw per-night results ──────────────────────────────
        # We need all nightly matrices before rendering so we can establish a
        # stable global label order (sorted by total-seconds across ALL nights).
        nights: list[tuple[date, np.ndarray, list[str], dict]] = []
        global_label_totals: dict[str, float] = {}
        day = request.date_from

        while day <= request.date_until:
            window_start, window_end = _daily_window(day, start_clock, end_clock)
            mat_full, labels_full, meta = viz.build_heatmap_matrix(
                start=window_start,
                end=window_end,
                bin_seconds=request.bin_size * 60,
                group_labels=request.group_labels,
                confidence_threshold = request.confidence_threshold,
                reassign_low_confidence_to_background=False,
                label_groups=configured_groups,
                min_total_seconds=0.0,
                top_n=None,
                max_bins=2000,
            )

            if (
                int(meta.get("num_bins", 0)) == 0
                or not labels_full
                or getattr(mat_full, "size", 0) == 0
            ):
                day += timedelta(days=1)
                continue

            keep_idx = [
                idx for idx, label in enumerate(labels_full) if label not in hide_set
            ]
            if not keep_idx:
                day += timedelta(days=1)
                continue

            mat = mat_full[keep_idx, :] if len(keep_idx) < len(labels_full) else mat_full.copy()
            labels = [labels_full[idx] for idx in keep_idx]

            # Accumulate global totals for stable cross-night ordering
            for i, label in enumerate(labels):
                global_label_totals[label] = global_label_totals.get(label, 0.0) + float(mat[i].sum())

            nights.append((day, mat, labels, meta))
            day += timedelta(days=1)

        # Global label order: descending by total seconds across all nights
        global_labels = sorted(global_label_totals, key=lambda k: -global_label_totals[k])

        # ── Pass 2: reindex each night to global order and render ──────────────
        generated = 0
        skipped = 0

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for (day, mat, labels, meta) in nights:
                label_to_row = {lab: i for i, lab in enumerate(labels)}

                # Build matrix with global row order; missing labels stay zero
                reindexed = np.zeros((len(global_labels), mat.shape[1]), dtype=float)
                for gi, lab in enumerate(global_labels):
                    if lab in label_to_row:
                        reindexed[gi] = mat[label_to_row[lab]]

                # Drop rows that are all-zero in this night's data so the plot
                # isn't padded with empty rows, while keeping order stable for
                # labels that do appear.
                active_rows = [
                    gi for gi, lab in enumerate(global_labels) if lab in label_to_row
                ]
                mat_night = reindexed[active_rows, :]
                labels_night = [global_labels[gi] for gi in active_rows]

                if request.hide_empty_bins:
                    non_empty_cols = np.where(mat_night.sum(axis=0) > 0)[0]
                    if len(non_empty_cols) == 0:
                        skipped += 1
                        continue
                    mat_night = mat_night[:, non_empty_cols]
                    meta = dict(meta)
                    meta["num_bins"] = len(non_empty_cols)
                    if "x_labels" in meta and meta["x_labels"] is not None:
                        meta["x_labels"] = [meta["x_labels"][i] for i in non_empty_cols]
                    bin_seconds = int(meta["bin_seconds"])
                    original_t0 = meta["t0"]
                    kept_times = [
                        original_t0 + timedelta(seconds=int(i) * bin_seconds)
                        for i in non_empty_cols
                    ]
                    meta["t0"] = kept_times[0]
                    meta["t_end"] = kept_times[-1] + timedelta(seconds=bin_seconds)
                    meta["_kept_bin_times"] = kept_times

                fig = viz.plot_heatmap(
                    mat_night,
                    labels_night,
                    meta,
                    show_warnings=True,
                    show_loudness=True,
                    max_cell_seconds=float(request.cap_min_per_bin * 60.0),
                    use_seaborn=False,
                )
                image_name = f"heatmap_{day.isoformat()}.png"
                image_path = os.path.join(tmp_dir, image_name)
                fig.savefig(image_path, dpi=160, bbox_inches="tight")
                try:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                except Exception:
                    pass
                zf.write(image_path, arcname=image_name)
                generated += 1

        if generated == 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise HTTPException(
                status_code=404,
                detail="No heatmaps generated for the selected date range and filters",
            )

        logger.info(
            "Generated %d heatmaps (skipped %d) for %s to %s",
            generated,
            skipped,
            request.date_from,
            request.date_until,
        )

        return FileResponse(
            path=zip_path,
            filename=zip_filename,
            media_type="application/zip",
            background=BackgroundTask(shutil.rmtree, tmp_dir, ignore_errors=True),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in export_batch_heatmaps: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@export_router.post("/export/label_sum/batch")
async def export_label_sum_batch(request: BatchLabelSumRequest):
    """
    Batch compute total duration per label for each night cycle in the date range.
    Returns a file response with columns: date,eating,drinking,snoring.
    If format_as_xlsx=true, XLSX is attempted first and falls back to CSV.
    """
    try:
        from smartstablemodel.services.data_store import DataStore
        from smartstablemodel.services.visualizer import Visualizer

        start_clock = _parse_hhmm(request.start_time)
        end_clock = _parse_hhmm(request.end_time)

        ds = DataStore(DB_PATH)
        viz = Visualizer(ds)

        # Prepare the set of labels we're interested in
        target_labels = set(request.sum_labels)

        items: list[NightlyLabelSumItem] = []
        day = request.date_from

        while day <= request.date_until:
            window_start, window_end = _daily_window(day, start_clock, end_clock)

            # Build heatmap matrix to get per-label durations
            mat_full, labels_full, meta = viz.build_heatmap_matrix(
                start=window_start,
                end=window_end,
                bin_seconds=60,  # 1-minute bins for aggregation
                confidence_threshold = request.confidence_threshold,
                reassign_low_confidence_to_background=False,
                group_labels=False,
                label_groups=None,
                min_total_seconds=0.0,
                top_n=None,
                max_bins=2000,
            )

            # Initialize label sums for this night (even if zero)
            label_sums: dict[str, LabelDurationInfo] = {}
            for label in target_labels:
                label_sums[label] = LabelDurationInfo(seconds=0.0, display=_format_seconds(0.0))

            # Sum up the requested labels from the matrix
            if (
                int(meta.get("num_bins", 0)) > 0
                and labels_full
                and getattr(mat_full, "size", 0) > 0
            ):
                for i, label in enumerate(labels_full):
                    if label in target_labels:
                        total_seconds = float(mat_full[i].sum())
                        label_sums[label] = LabelDurationInfo(
                            seconds=total_seconds, display=_format_seconds(total_seconds)
                        )

            # Create the nightly item with a consistent time window format
            time_window = f"{request.start_time}-{request.end_time}"
            item = NightlyLabelSumItem(
                date=day.isoformat(),
                time_window=time_window,
                label_sums=label_sums,
            )
            items.append(item)

            day += timedelta(days=1)

        logger.info(
            "Computed label sums for %d nights from %s to %s",
            len(items),
            request.date_from,
            request.date_until,
        )

        rows: list[list[str]] = []
        for item in items:
            rows.append(
                [
                    item.date,
                    _format_hhmmss(
                        item.label_sums.get(
                            "eating", LabelDurationInfo(seconds=0.0, display="0s")
                        ).seconds
                    ),
                    _format_hhmmss(
                        item.label_sums.get(
                            "drinking", LabelDurationInfo(seconds=0.0, display="0s")
                        ).seconds
                    ),
                    _format_hhmmss(
                        item.label_sums.get(
                            "snoring", LabelDurationInfo(seconds=0.0, display="0s")
                        ).seconds
                    ),
                ]
            )

        tmp_dir = tempfile.mkdtemp(prefix="batch_label_sum_")
        base_name = (
            f"label_sum_{request.date_from.isoformat()}_{request.date_until.isoformat()}"
        )

        if request.format_as_xlsx:
            xlsx_path = os.path.join(tmp_dir, f"{base_name}.xlsx")
            try:
                import importlib

                openpyxl_module = importlib.import_module("openpyxl")
                workbook_class = getattr(openpyxl_module, "Workbook")
                workbook = workbook_class()
                sheet = workbook.active
                sheet.title = "label_sum"
                sheet.append(["date", "eating", "drinking", "snoring"])
                for row in rows:
                    sheet.append(row)
                workbook.save(xlsx_path)

                return FileResponse(
                    path=xlsx_path,
                    filename=os.path.basename(xlsx_path),
                    media_type=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    background=BackgroundTask(shutil.rmtree, tmp_dir, ignore_errors=True),
                )
            except Exception as xlsx_error:
                logger.warning(
                    "XLSX export requested but unavailable (%s). Falling back to CSV.",
                    xlsx_error,
                )

        csv_path = os.path.join(tmp_dir, f"{base_name}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["date", "eating", "drinking", "snoring"])
            csv_writer.writerows(rows)

        return FileResponse(
            path=csv_path,
            filename=os.path.basename(csv_path),
            media_type="text/csv",
            background=BackgroundTask(shutil.rmtree, tmp_dir, ignore_errors=True),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in export_label_sum_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@export_router.post("/export/testset/bundle")
async def export_testset_bundle(request: TestSetBundleRequest):
    """
    Bundle test-pool snippets and manifest for a selected model version.

    Request body example:
    {
      "model_version": "v29"
    }
    """
    try:
        version_info = get_model_version_info(
            request.model_version, model_architecture=request.model_architecture
        )
        if not version_info:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Model version '{request.model_version}' not found "
                    f"for architecture '{request.model_architecture}'"
                ),
            )

        run_id = version_info.get("run_id")
        if not run_id:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Model version '{request.model_version}' exists but has no run_id"
                ),
            )

        manifest = download_test_pool_manifest(run_id)
        if not manifest:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Could not download test_pool_manifest.json from MLflow artifacts "
                    f"for run '{run_id}'"
                ),
            )

        audio_index = _scan_audio_files()
        if not audio_index:
            raise HTTPException(
                status_code=404,
                detail="No local audio files found for snippet extraction",
            )

        normalized_version = _normalize_model_version(request.model_version)
        tmp_dir = tempfile.mkdtemp(prefix="testset_bundle_")
        zip_filename = (
            f"testset_v{normalized_version}_{request.model_architecture}.zip"
        )
        zip_path = os.path.join(tmp_dir, zip_filename)

        included_count = 0
        missing_entries: list[dict[str, str]] = []
        included_entries: list[dict[str, object]] = []

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("test_pool_manifest.json", json.dumps(manifest, indent=2))

            for label, samples in manifest.items():
                safe_label = _safe_name(label)
                for sample_id in samples:
                    parsed = _parse_manifest_sample_id(sample_id)
                    if parsed is None:
                        missing_entries.append(
                            {
                                "sample_id": sample_id,
                                "label": label,
                                "reason": "invalid sample_id format",
                            }
                        )
                        continue

                    source_filename, offset_seconds = parsed
                    source_basename = os.path.basename(source_filename)
                    source_path = audio_index.get(source_filename) or audio_index.get(
                        source_basename
                    )
                    if not source_path:
                        missing_entries.append(
                            {
                                "sample_id": sample_id,
                                "label": label,
                                "reason": "source audio file not found locally",
                            }
                        )
                        continue

                    sample_stem = _safe_name(sample_id)
                    clip_relpath = f"audio/{safe_label}/{sample_stem}.wav"
                    clip_abspath = os.path.join(tmp_dir, clip_relpath)

                    ok, error_msg, actual_duration = _extract_snippet_to_wav(
                        source_path=source_path,
                        offset_seconds=offset_seconds,
                        duration_seconds=request.snippet_duration_seconds,
                        output_path=clip_abspath,
                    )
                    if not ok:
                        missing_entries.append(
                            {
                                "sample_id": sample_id,
                                "label": label,
                                "reason": error_msg or "failed to extract snippet",
                            }
                        )
                        continue

                    zf.write(clip_abspath, arcname=clip_relpath)
                    included_count += 1
                    included_entries.append(
                        {
                            "sample_id": sample_id,
                            "label": label,
                            "source_file": source_basename,
                            "source_offset_seconds": offset_seconds,
                            "snippet_duration_seconds": actual_duration,
                            "bundle_path": clip_relpath,
                        }
                    )

            bundle_manifest = {
                "model": {
                    "architecture": request.model_architecture,
                    "requested_version": request.model_version,
                    "resolved_version": version_info.get("version"),
                    "run_id": run_id,
                },
                "summary": {
                    "labels": len(manifest),
                    "manifest_samples": sum(len(v) for v in manifest.values()),
                    "included_snippets": included_count,
                    "missing_or_failed": len(missing_entries),
                    "snippet_duration_seconds": request.snippet_duration_seconds,
                },
                "included": included_entries,
                "missing": missing_entries,
            }
            zf.writestr("bundle_manifest.json", json.dumps(bundle_manifest, indent=2))

        if included_count == 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise HTTPException(
                status_code=404,
                detail=(
                    "No snippets could be extracted for the requested model version. "
                    "Check that source audio files are mounted and match manifest sample IDs."
                ),
            )

        logger.info(
            "Created testset bundle for %s (%s): %d snippets included, %d missing",
            request.model_version,
            run_id,
            included_count,
            len(missing_entries),
        )

        return FileResponse(
            path=zip_path,
            filename=zip_filename,
            media_type="application/zip",
            background=BackgroundTask(shutil.rmtree, tmp_dir, ignore_errors=True),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in export_testset_bundle: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _build_video_audio_coverage(
    request: "VideoAndAudioExportRequest",
) -> tuple[
    list[datetime],                           # expected_slots
    set[datetime],                            # available_video_slots
    set[datetime],                            # available_audio_slots
    list[tuple[int | None, datetime, str, str, bool, bool]],  # task_entries
    list[dict[str, str]],                     # missing_videos
    list[dict[str, str]],                     # missing_audio_for_existing_video
    list[tuple[int | None, datetime, str, str]],  # to_process (video+audio both present)
    int,                                      # tasks_fetched
]:
    """Shared logic: fetch LS tasks, resolve paths, compute expected slots and gaps."""
    from ml_backend.smartstable_core.task_filtering import list_tasks_by_recording_period
    from ml_backend.util.label_studio_helper import ls_url_to_local_path

    tasks = list_tasks_by_recording_period(
        project_id=request.project_id,
        stable_id=request.stable_id,
        stall_id=request.stall_id,
        date_from=request.date_from.isoformat(),
        date_until=request.date_until.isoformat(),
        time_from="19:00",
        time_until="09:00",
        return_summary=False,
        max_tasks=request.max_tasks,
    )

    # One night window per start-date. date_from=Feb2 date_until=Feb3 → 1 night (Feb2 evening).
    day_span = (request.date_until - request.date_from).days
    night_count = max(1, day_span)
    expected_slots: list[datetime] = []
    for i in range(night_count):
        expected_slots.extend(_night_slots_for_day(request.date_from + timedelta(days=i)))

    task_entries: list[tuple[int | None, datetime, str, str, bool, bool]] = []
    available_video_slots: set[datetime] = set()
    available_audio_slots: set[datetime] = set()

    for task in tasks:
        if hasattr(task, "data"):
            data = getattr(task, "data") or {}
        elif isinstance(task, dict):
            data = task.get("data", {})
        else:
            data = {}
        task_id = getattr(task, "id", None)
        audio_url = data.get("audio")
        video_url = data.get("video")
        recorded_date = data.get("recorded_date")
        recorded_time = data.get("recorded_time")

        slot: datetime | None = None
        if isinstance(recorded_date, str) and isinstance(recorded_time, str):
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    slot = _to_half_hour_slot(
                        datetime.strptime(f"{recorded_date} {recorded_time}", fmt)
                    )
                    break
                except ValueError:
                    continue

        audio_path = ls_url_to_local_path(audio_url) if isinstance(audio_url, str) and audio_url else ""
        video_path = ls_url_to_local_path(video_url) if isinstance(video_url, str) and video_url else ""

        has_audio = bool(audio_path) and os.path.exists(audio_path)
        has_video = bool(video_path) and os.path.exists(video_path)

        if slot:
            if has_video:
                available_video_slots.add(slot)
            if has_audio:
                available_audio_slots.add(slot)
            task_entries.append((task_id, slot, audio_path, video_path, has_audio, has_video))

    missing_videos: list[dict[str, str]] = []
    for slot in expected_slots:
        if slot not in available_video_slots:
            missing_videos.append(
                {
                    "slot_date": slot.date().isoformat(),
                    "slot_time": slot.time().strftime("%H:%M:%S"),
                    "reason": "no video task/file found for expected slot",
                }
            )

    missing_audio_for_existing_video: list[dict[str, str]] = []
    to_process: list[tuple[int | None, datetime, str, str]] = []
    for task_id, slot, audio_path, video_path, has_audio, has_video in task_entries:
        if has_video and not has_audio:
            missing_audio_for_existing_video.append(
                {
                    "task_id": str(task_id or ""),
                    "slot_date": slot.date().isoformat(),
                    "slot_time": slot.time().strftime("%H:%M:%S"),
                    "video_file": os.path.basename(video_path),
                    "reason": "video exists but audio is missing/unresolvable",
                }
            )
        elif has_video and has_audio:
            to_process.append((task_id, slot, audio_path, video_path))

    return (
        expected_slots,
        available_video_slots,
        available_audio_slots,
        task_entries,
        missing_videos,
        missing_audio_for_existing_video,
        sorted(to_process, key=lambda x: x[1])[: request.max_tasks],
        len(tasks),
    )


@export_router.post("/export/video_and_audio/coverage")
async def export_video_and_audio_coverage(request: VideoAndAudioExportRequest):
    """
    Coverage report for the nightly video+audio export.

    Uses the same request body as /export/video_and_audio/ but performs no ffmpeg
    processing. Returns a per-night breakdown showing how many 30-minute slots
    are present/missing for both video and audio.
    """
    try:
        logger.info(
            "[video_and_audio_coverage] start project=%s stable=%s stall=%s range=%s..%s",
            request.project_id,
            request.stable_id,
            request.stall_id,
            request.date_from.isoformat(),
            request.date_until.isoformat(),
        )

        (
            expected_slots,
            available_video_slots,
            available_audio_slots,
            _task_entries,
            missing_videos,
            missing_audio_for_existing_video,
            to_process,
            tasks_fetched,
        ) = _build_video_audio_coverage(request)

        night_count = max(1, (request.date_until - request.date_from).days)
        nights: list[dict] = []
        suspicious_short_videos: list[dict[str, object]] = []
        suspicious_short_audios: list[dict[str, object]] = []

        run_start = datetime.now()
        total_nights = night_count

        for i in range(night_count):
            night_started = datetime.now()
            night_start = request.date_from + timedelta(days=i)
            night_slots = _night_slots_for_day(night_start)
            night_end = (night_start + timedelta(days=1)).isoformat()
            night_slots_set = set(night_slots)

            night_missing_v = [s for s in night_slots if s not in available_video_slots]
            night_missing_a = [
                s for s in night_slots if s in available_video_slots and s not in available_audio_slots
            ]

            night_pairs = [item for item in to_process if item[1] in night_slots_set]
            night_short_v = 0
            night_short_a = 0
            night_short_samples: list[dict[str, object]] = []

            for task_id, slot, audio_path, video_path in night_pairs:
                video_size = _get_file_size(video_path)
                audio_size = _get_file_size(audio_path)
                video_short = video_size is not None and video_size < COVERAGE_MIN_VIDEO_BYTES
                audio_short = audio_size is not None and audio_size < COVERAGE_MIN_AUDIO_BYTES

                if video_short:
                    night_short_v += 1
                    if len(suspicious_short_videos) < COVERAGE_SHORT_SAMPLE_LIMIT:
                        suspicious_short_videos.append(
                            {
                                "task_id": str(task_id or ""),
                                "slot_date": slot.date().isoformat(),
                                "slot_time": slot.time().strftime("%H:%M:%S"),
                                "video_file": os.path.basename(video_path),
                                "video_size_bytes": video_size,
                                "threshold_bytes": COVERAGE_MIN_VIDEO_BYTES,
                            }
                        )
                    if len(night_short_samples) < COVERAGE_SHORT_SAMPLE_LIMIT:
                        night_short_samples.append(
                            {
                                "kind": "video",
                                "task_id": str(task_id or ""),
                                "slot_time": slot.strftime("%Y-%m-%d %H:%M:%S"),
                                "file": os.path.basename(video_path),
                                "size_bytes": video_size,
                            }
                        )

                if audio_short:
                    night_short_a += 1
                    if len(suspicious_short_audios) < COVERAGE_SHORT_SAMPLE_LIMIT:
                        suspicious_short_audios.append(
                            {
                                "task_id": str(task_id or ""),
                                "slot_date": slot.date().isoformat(),
                                "slot_time": slot.time().strftime("%H:%M:%S"),
                                "audio_file": os.path.basename(audio_path),
                                "audio_size_bytes": audio_size,
                                "threshold_bytes": COVERAGE_MIN_AUDIO_BYTES,
                            }
                        )
                    if len(night_short_samples) < COVERAGE_SHORT_SAMPLE_LIMIT:
                        night_short_samples.append(
                            {
                                "kind": "audio",
                                "task_id": str(task_id or ""),
                                "slot_time": slot.strftime("%Y-%m-%d %H:%M:%S"),
                                "file": os.path.basename(audio_path),
                                "size_bytes": audio_size,
                            }
                        )

            night_elapsed = max(0.0, (datetime.now() - night_started).total_seconds())
            elapsed_total = max(0.0, (datetime.now() - run_start).total_seconds())
            avg_per_night = elapsed_total / (i + 1)
            eta_seconds = int(avg_per_night * max(0, total_nights - i - 1))

            logger.info(
                "[video_and_audio_coverage] night %d/%d start=%s expected=%d video_missing=%d audio_missing=%d short_video=%d short_audio=%d elapsed=%.1fs eta=%ss",
                i + 1,
                total_nights,
                night_start.isoformat(),
                len(night_slots),
                len(night_missing_v),
                len(night_missing_a),
                night_short_v,
                night_short_a,
                night_elapsed,
                eta_seconds,
            )

            nights.append(
                {
                    "night": f"{night_start.isoformat()} 19:00 → {night_end} 09:00",
                    "total_expected_slots": len(night_slots),
                    "slots_with_video": len(night_slots) - len(night_missing_v),
                    "slots_with_audio": len(night_slots) - len(night_missing_v) - len(night_missing_a),
                    "missing_video_count": len(night_missing_v),
                    "missing_audio_count": len(night_missing_a),
                    "suspicious_short_video_count": night_short_v,
                    "suspicious_short_audio_count": night_short_a,
                    "missing_video_sample": [
                        f"{s.date().isoformat()} {s.strftime('%H:%M')}" for s in night_missing_v[:3]
                    ],
                    "missing_audio_sample": [
                        f"{s.date().isoformat()} {s.strftime('%H:%M')}" for s in night_missing_a[:3]
                    ],
                    "suspicious_short_sample": night_short_samples,
                }
            )

        summary = {
            "expected_slots_total": len(expected_slots),
            "slots_with_video": len(available_video_slots & set(expected_slots)),
            "slots_with_audio": len(available_audio_slots & set(expected_slots)),
            "slots_ready_for_export": len(to_process),
            "missing_video_slots": len(missing_videos),
            "missing_audio_slots": len(missing_audio_for_existing_video),
            "suspicious_short_video_count": len(suspicious_short_videos),
            "suspicious_short_audio_count": len(suspicious_short_audios),
            "short_size_thresholds": {
                "video_bytes": COVERAGE_MIN_VIDEO_BYTES,
                "audio_bytes": COVERAGE_MIN_AUDIO_BYTES,
            },
        }

        logger.info(
            "[video_and_audio_coverage] done project=%s stable=%s stall=%s nights=%d slots=%d videos=%d audios=%d short_videos=%d short_audios=%d missing_videos=%d missing_audios=%d",
            request.project_id,
            request.stable_id,
            request.stall_id,
            total_nights,
            summary["expected_slots_total"],
            summary["slots_with_video"],
            summary["slots_with_audio"],
            summary["suspicious_short_video_count"],
            summary["suspicious_short_audio_count"],
            summary["missing_video_slots"],
            summary["missing_audio_slots"],
        )

        return {
            "project_id": request.project_id,
            "stable_id": request.stable_id,
            "stall_id": request.stall_id,
            "date_from": request.date_from.isoformat(),
            "date_until": request.date_until.isoformat(),
            "tasks_fetched": tasks_fetched,
            "summary": summary,
            "nights": nights,
            "suspicious_short_videos": suspicious_short_videos,
            "suspicious_short_audios": suspicious_short_audios,
            "missing_videos_report": missing_videos[:50],
            "missing_audio_report": missing_audio_for_existing_video[:50],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in export_video_and_audio_coverage: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@export_router.post("/export/video_and_audio/")
async def export_video_and_audio(request: VideoAndAudioExportRequest):
    """
    Export video files with attached audio based on existing Label Studio tasks.

    Uses SmartStable task filtering (date/stable/stall and nightly 19:00-09:00 window),
    resolves task-local paths, muxes video+audio using ffmpeg, and writes outputs to
    model_cache/export (inside SMARTSTABLE_MODELS_DIR/export).

    Returns JSON report including output paths and missing-video slot report.
    """
    try:
        audio_root = os.getenv("SMARTSTABLE_AUDIO_DIR", "/label-studio/data/audio")
        video_root = os.getenv("SMARTSTABLE_VIDEO_DIR", "/label-studio/data/video")
        ffmpeg_bin = shutil.which("ffmpeg")
        models_root = os.getenv("SMARTSTABLE_MODELS_DIR", "/app/smartstablemodel/models")
        export_root = os.path.join(models_root, "export")
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(
            export_root,
            f"video_and_audio_{request.stable_id}_{request.stall_id}_{run_stamp}",
        )

        if not ffmpeg_bin:
            raise HTTPException(
                status_code=500,
                detail="ffmpeg is required for video+audio export but was not found on PATH",
            )

        if not os.path.isdir(audio_root):
            logger.warning("Configured audio root does not exist: %s", audio_root)
        if not os.path.isdir(video_root):
            logger.warning("Configured video root does not exist: %s", video_root)

        os.makedirs(run_dir, exist_ok=True)

        (
            expected_slots,
            available_video_slots,
            available_audio_slots,
            _task_entries,
            missing_videos,
            missing_audio_for_existing_video,
            to_process,
            tasks_fetched,
        ) = _build_video_audio_coverage(request)

        exported_files: list[dict[str, str]] = []
        export_errors: list[dict[str, str]] = []
        total = len(to_process)
        start_ts = datetime.now()

        for idx, (task_id, slot, audio_path, video_path) in enumerate(to_process, start=1):

            output_name = (
                f"{request.stable_id}_{request.stall_id}_{slot.strftime('%Y%m%d_%H%M%S')}_av.mp4"
            )
            output_path = os.path.join(run_dir, output_name)

            elapsed_sec = max(0.0, (datetime.now() - start_ts).total_seconds())
            avg_sec = (elapsed_sec / (idx - 1)) if idx > 1 else 0.0
            remaining = max(0, total - idx + 1)
            eta_sec = int(avg_sec * remaining) if avg_sec > 0 else 0

            logger.info(
                "[video_and_audio] Processing %d/%d task_id=%s slot=%s eta=%ss",
                idx,
                total,
                task_id,
                slot.strftime("%Y-%m-%d %H:%M:%S"),
                eta_sec,
            )

            cmd = [
                ffmpeg_bin,
                "-y",
                "-i",
                video_path,
                "-i",
                audio_path,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                output_path,
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                elapsed_done = int((datetime.now() - start_ts).total_seconds())
                avg_done = (elapsed_done / idx) if idx > 0 else 0.0
                eta_done = int(avg_done * max(0, total - idx))

                logger.info(
                    "[video_and_audio] Done %d/%d task_id=%s output=%s elapsed=%ss eta=%ss",
                    idx,
                    total,
                    task_id,
                    output_name,
                    elapsed_done,
                    eta_done,
                )

                exported_files.append(
                    {
                        "task_id": str(task_id or ""),
                        "slot_date": slot.date().isoformat(),
                        "slot_time": slot.time().strftime("%H:%M:%S"),
                        "video_file": os.path.basename(video_path),
                        "audio_file": os.path.basename(audio_path),
                        "export_file": output_name,
                        "export_path": output_path,
                    }
                )
            except subprocess.CalledProcessError as mux_exc:
                elapsed_done = int((datetime.now() - start_ts).total_seconds())
                avg_done = (elapsed_done / idx) if idx > 0 else 0.0
                eta_done = int(avg_done * max(0, total - idx))

                logger.warning(
                    "[video_and_audio] Failed %d/%d task_id=%s video=%s audio=%s elapsed=%ss eta=%ss",
                    idx,
                    total,
                    task_id,
                    os.path.basename(video_path),
                    os.path.basename(audio_path),
                    elapsed_done,
                    eta_done,
                )

                export_errors.append(
                    {
                        "task_id": str(task_id or ""),
                        "slot_date": slot.date().isoformat(),
                        "slot_time": slot.time().strftime("%H:%M:%S"),
                        "video_file": os.path.basename(video_path),
                        "audio_file": os.path.basename(audio_path),
                        "reason": (mux_exc.stderr or mux_exc.stdout or str(mux_exc)).strip(),
                    }
                )

        report = {
            "project_id": request.project_id,
            "stable_id": request.stable_id,
            "stall_id": request.stall_id,
            "date_from": request.date_from.isoformat(),
            "date_until": request.date_until.isoformat(),
            "max_tasks": request.max_tasks,
            "export_directory": run_dir,
            "summary": {
                "expected_slots_19_to_09_every_30min": len(expected_slots),
                "slots_with_video": len([s for s in expected_slots if s in available_video_slots]),
                "slots_with_audio": len([s for s in expected_slots if s in available_audio_slots]),
                "slots_prepared_for_export": len(to_process),
                "videos_exported": len(exported_files),
                "video_export_failed": len(export_errors),
                "missing_videos": len(missing_videos),
                "missing_audio_for_existing_video": len(missing_audio_for_existing_video),
                "tasks_fetched": tasks_fetched,
            },
            "exported_files": exported_files,
            "missing_videos_report": missing_videos,
            "missing_audio_report": missing_audio_for_existing_video,
            "export_errors": export_errors,
        }

        report_path = os.path.join(run_dir, "report.json")
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2)

        if not exported_files:
            raise HTTPException(
                status_code=404,
                detail="No videos could be exported with attached audio for the requested filters",
            )

        logger.info(
            "Video+audio export finished for stable=%s stall=%s: exported=%d missing_videos=%d missing_audio=%d",
            request.stable_id,
            request.stall_id,
            len(exported_files),
            len(missing_videos),
            len(missing_audio_for_existing_video),
        )

        return report
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in export_video_and_audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
