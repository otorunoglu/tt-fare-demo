#!/usr/bin/env python3
"""
Edge (Raspberry Pi) realtime monitoring script.

Pipeline:
  Audio stream  --> frame buffer (e.g. 1s snippets) --> Model inference (label_distribution + loudness)
                  --> ContextualMetaModelEvaluator.evaluate(event)
                  --> STDOUT JSONL (events + warnings) & optional daily rolling log.

Key goals:
  - Low memory footprint
  - Graceful recovery on audio / model hiccups
  - Pluggable inference backend (replace DummyModel with real model)
  - Night rollover handled by evaluator
  - Optional dry-run mode reading from a prerecorded WAV/RAW file

Replace the DummyModel.infer() with the actual classifier.
"""

from __future__ import annotations
import calendar
import json
import logging
import math
import queue
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Iterable

import numpy as np

from smartstablemodel.config import load_metamodel_config
from smartstablemodel.services.data_store import DataStore

try:
    import sounddevice as sd  # lightweight & good on Pi if ALSA present

    HAS_SD = True
except ImportError:
    sd = None
    HAS_SD = False

# Local imports
from smartstablemodel import MetaModelDecider
from smartstablemodel.services import ModelUpdateService
from smartstablemodel.multi_stall_manager import (
    MultiStallDetectorManager,
    SR as MODEL_SR,
    WIN_SECONDS as MODEL_SEGMENT_DURATION,
    WIN_OVERLAP as MODEL_SEGMENT_OVERLAP,
)
from smartstablemodel.smartstable_types import MultiStallBatchResultReturn

# ─── Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("SoundscapeMonitor")
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)


############################################################
# RUNTIME CONSTANTS (minimal; adjust as needed)
############################################################
STABLE_ID = "stable01"
STALL_ID = "stall01"
SAMPLE_RATE = MODEL_SR  # ensure alignment with model
SEGMENT_DURATION = MODEL_SEGMENT_DURATION  # re-use central config (2.0s)
SEGMENT_OVERLAP = MODEL_SEGMENT_OVERLAP  # re-use central config (0.5)
HOP_SECONDS = SEGMENT_DURATION * (1 - SEGMENT_OVERLAP)  # should be 1.0
SNIPPET_SECONDS = HOP_SECONDS  # one snippet == hop for simple loop
ONE_EVENT_PER_SECOND = True  # emit exactly one inference each hop
BATCH_GROUP_SIZE = 1  # keep 1 for low latency edge
EXIT_AFTER_SECONDS: int | None = (
    30  # auto-exit for quick tests; set None for continuous
)
PRINT_EVENTS = False
PRINT_WARNINGS = True  # (warnings currently not separately emitted here)
SIMULATE_FILE: str | None = (
    "data/stable01_stall01_horse01_20250920_060019_mic.flac"  # path to .wav/.flac/.npy for simulation; None for mic
)
# SIMULATE_FILE: str | None = None

# Metamodel / evaluator parameters
LOUDNESS_WARMUP_SEC = 900
BURST_WINDOW_SEC = 120
BURST_THRESHOLD_SEC = 10

# Verbosity (keep lean: only top prediction summary)
VERBOSE = True
VERBOSE_TOP_N = 1
VERBOSE_SHOW_FULL = False

# Simulation debug (kept lightweight)
SIMULATION_DEBUG = True
SIMULATION_PROGRESS_SNIPPETS = 10
############################################################

# ---------------------------------------------------------------------------


# ------------- CONFIG DATACLASSES ---------------------------------------
@dataclass
class AudioConfig:
    sample_rate: int = 16000  # match model requirement
    channels: int = 1
    snippet_seconds: float = 1.0  # metamodel default snippet
    dtype: str = "float32"
    rms_floor: float = 1e-6  # avoid log(0)
    device: Optional[int] = None  # HW device index (None = default)


@dataclass
class RuntimeConfig:
    stable_id: str = "stable01"
    stall_id: str = "stall01"
    output_dir: Path = Path("edge-output")
    flush_interval_sec: int = 30
    print_events: bool = True
    print_warnings: bool = True
    max_queue: int = 64
    simulate_from_file: Optional[Path] = (
        None  # If provided, read mono wav/pcm float32 file instead of microphone
    )
    time_drift_tolerance_sec: float = 2.0
    timezone_utc: bool = True
    exit_after_seconds: Optional[int] = None
    segment_duration: float = 2.0
    segment_overlap: float = 0.5
    batch_size: int = 8  # number of segments to batch for model inference


def build_time_context(ts: datetime) -> dict:
    hour = ts.hour
    # Simple night definition: 20:00–06:00
    is_night = hour >= 20 or hour < 6
    return {
        "hour": hour,
        "is_night": is_night,
        "day_of_week": calendar.day_name[ts.weekday()],  # type: ignore
    }


# ------------- AUDIO CAPTURE THREAD -------------------------------------
class AudioStreamer(threading.Thread):
    def __init__(
        self, audio_cfg: AudioConfig, runtime_cfg: RuntimeConfig, out_q: queue.Queue
    ):
        super().__init__(daemon=True)
        self.audio_cfg = audio_cfg
        self.runtime_cfg = runtime_cfg
        self.out_q = out_q
        self.stop_flag = threading.Event()
        self.samples_per_snippet = int(
            audio_cfg.sample_rate * audio_cfg.snippet_seconds
        )
        # Internal accumulation buffer so we can accept arbitrary callback frame sizes
        self._accum = np.zeros((0,), dtype=audio_cfg.dtype)

    def run(self):
        if self.runtime_cfg.simulate_from_file:
            self._run_file_simulation()
        else:
            if not HAS_SD:
                print(
                    "sounddevice not available; install it or provide --simulate-file",
                    file=sys.stderr,
                )
                return
            self._run_mic()

    def stop(self):
        self.stop_flag.set()

    def _run_mic(self):
        ac = self.audio_cfg

        def callback(indata, frames, time_info, status):  # noqa
            if self.stop_flag.is_set():
                raise sd.CallbackStop()  # type: ignore
            if status:
                print(f"[audio] status: {status}", file=sys.stderr)
            mono = indata[:, 0].astype(ac.dtype, copy=False)
            self._accept_audio_chunk(mono)

        # Provide a blocksize hint; if device refuses it we still buffer variable frames
        try:
            stream = sd.InputStream(
                channels=ac.channels,  # type: ignore
                samplerate=ac.sample_rate,
                dtype=ac.dtype,
                device=ac.device,
                blocksize=self.samples_per_snippet,
                callback=callback,
            )
        except Exception:
            stream = sd.InputStream(
                channels=ac.channels,  # type: ignore
                samplerate=ac.sample_rate,
                dtype=ac.dtype,
                device=ac.device,
                callback=callback,
            )

        with stream:
            while not self.stop_flag.is_set():
                time.sleep(0.1)

    def _run_file_simulation(self):
        """
        Simulate real-time by reading 1s chunks from file (expects float32 .npy or raw .wav-like?
        Provide a simple adapter; for actual WAV use scipy.io.wavfile if available).
        """
        path = self.runtime_cfg.simulate_from_file
        if path is None:
            print("[simulate] No file provided", file=sys.stderr)
            return
        if not path.exists():
            print(f"[simulate] file not found {path}", file=sys.stderr)
            return
        if path.suffix.lower() == ".npy":
            data = np.load(path).astype(self.audio_cfg.dtype)
            sr = self.audio_cfg.sample_rate  # assume correct
        else:
            try:
                import soundfile as sf  # type: ignore

                data, sr = sf.read(str(path))
                if sr != self.audio_cfg.sample_rate:
                    print(
                        f"[simulate] sample_rate mismatch file={sr} expected={self.audio_cfg.sample_rate}",
                        file=sys.stderr,
                    )
                if data.ndim > 1:
                    data = data[:, 0]
                data = data.astype(self.audio_cfg.dtype)
            except ImportError:
                print(
                    "Install soundfile or provide .npy for simulation.", file=sys.stderr
                )
                return

        idx = 0
        step = self.samples_per_snippet
        total_samples = len(data)
        total_duration = total_samples / float(self.audio_cfg.sample_rate)
        est_snippets = math.ceil(total_samples / step)
        if SIMULATION_DEBUG:
            print(
                f"# SIMULATE_START file='{path}' samples={total_samples} sr={sr} duration_sec={total_duration:.2f} snippet_sec={self.audio_cfg.snippet_seconds} est_snippets={est_snippets}",
                file=sys.stderr,
            )
        emitted = 0
        while not self.stop_flag.is_set() and idx < len(data):
            chunk = data[idx : idx + step]
            if len(chunk) < step:
                # pad last
                chunk = np.pad(chunk, (0, step - len(chunk)))
            self._accumulate_and_enqueue(chunk)
            idx += step
            emitted += 1
            if SIMULATION_DEBUG and (
                emitted == 1 or emitted % SIMULATION_PROGRESS_SNIPPETS == 0
            ):
                prog = min(idx, total_samples) / total_samples * 100.0
                print(
                    f"# SIM_PROGRESS emitted={emitted} elapsed_sec={emitted * self.audio_cfg.snippet_seconds:.1f} prog={prog:.1f}%",
                    file=sys.stderr,
                )
            time.sleep(self.audio_cfg.snippet_seconds)
        if SIMULATION_DEBUG:
            print(
                f"# SIMULATE_DONE emitted={emitted} real_duration_sec={emitted * self.audio_cfg.snippet_seconds:.2f}",
                file=sys.stderr,
            )

    def _accept_audio_chunk(self, mono: np.ndarray):
        """Accumulate arbitrary sized audio and emit fixed-size snippets to queue."""
        self._accum = np.concatenate([self._accum, mono])
        while self._accum.shape[0] >= self.samples_per_snippet:
            snippet = self._accum[: self.samples_per_snippet]
            self._accum = self._accum[self.samples_per_snippet :]
            ts = (
                datetime.utcnow().replace(tzinfo=timezone.utc)
                if self.runtime_cfg.timezone_utc
                else datetime.now()
            )
            try:
                self.out_q.put_nowait((ts, snippet.copy()))
            except queue.Full:
                # Drop oldest then retry once
                try:
                    self.out_q.get_nowait()
                except Exception:
                    pass
                try:
                    self.out_q.put_nowait((ts, snippet.copy()))
                except Exception:
                    pass

    # Backwards compatibility for simulation path (which feeds already sized chunks)
    def _accumulate_and_enqueue(self, mono: np.ndarray):
        self._accept_audio_chunk(mono)


def realtime_loop(
    evaluator: MetaModelDecider,
    audio_cfg: AudioConfig,
    runtime_cfg: RuntimeConfig,
    multi_stall_manager: MultiStallDetectorManager,
):
    if not multi_stall_manager:
        print("multi_stall_manager not initialized; exiting.", file=sys.stderr)
        return

    # Enforce consistency (simplified pipeline assumes snippet == hop)
    # Validate hop alignment
    hop = SEGMENT_DURATION * (1 - SEGMENT_OVERLAP)
    if abs(hop - SNIPPET_SECONDS) > 1e-6:
        raise ValueError(
            f"SNIPPET_SECONDS ({SNIPPET_SECONDS}) must equal hop ({hop}). Adjust constants."
        )

    # Override runtime config to be consistent
    runtime_cfg.segment_duration = SEGMENT_DURATION
    runtime_cfg.segment_overlap = SEGMENT_OVERLAP

    if audio_cfg.sample_rate != MODEL_SR:
        logger.warning(
            f"Input sample rate {audio_cfg.sample_rate} != model SR {MODEL_SR}"
        )

    q: queue.Queue = queue.Queue(maxsize=runtime_cfg.max_queue)
    audio_cfg.snippet_seconds = SNIPPET_SECONDS

    streamer = AudioStreamer(audio_cfg, runtime_cfg, q)
    streamer.start()

    # Timing / bookkeeping
    stream_start_ts: Optional[datetime] = None
    total_samples = 0
    sr = audio_cfg.sample_rate
    segment_samples = int(SEGMENT_DURATION * sr)
    hop_samples = int(hop * sr)  # informative; currently not used directly

    # Rolling buffer: only need last segment window
    ring = np.zeros((0,), dtype="float32")

    # Batch accumulation (optional)
    pending_segments: list[np.ndarray] = []
    pending_times: list[tuple[float, float]] = []

    start_wall = time.time()
    last_flush = time.time()
    last_snippet_wall = time.time()
    last_segment_emit_wall = time.time()
    segments_emitted = 0

    out_dir = runtime_cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / f"segments_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
    warnings_path = out_dir / f"warnings_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"

    def safe_write(path: Path, obj: Dict[str, Any]):
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("# Realtime monitor started (simple 2s/50% pipeline)", file=sys.stderr)

    try:
        while True:
            if (
                runtime_cfg.exit_after_seconds
                and (time.time() - start_wall) > runtime_cfg.exit_after_seconds
            ):
                break

            try:
                snippet_ts, snippet_audio = q.get(timeout=0.6)
                last_snippet_wall = time.time()
            except queue.Empty:
                now_wd = time.time()
                if now_wd - last_snippet_wall > 5:
                    print("# WATCHDOG: No audio snippets >5s", file=sys.stderr)
                    last_snippet_wall = now_wd
                continue

            if stream_start_ts is None:
                stream_start_ts = snippet_ts

            # Update rolling buffer & sample counters
            snippet_audio = snippet_audio.astype("float32")
            ring = np.concatenate([ring, snippet_audio])
            total_samples += snippet_audio.size
            if ring.size > segment_samples:
                ring = ring[-segment_samples:]

            # Enough audio for at least one segment?
            if total_samples < segment_samples:
                continue  # warm-up

            # Emit exactly one new segment per hop (since snippet == hop)
            # Segment end = total_samples; start = end - segment_samples
            # Relative times (seconds since stream start)
            end_time_sec = total_samples / sr
            start_time_sec = end_time_sec - SEGMENT_DURATION

            segment_audio = ring.copy()  # already last 2s
            pending_segments.append(segment_audio)
            pending_times.append((start_time_sec, end_time_sec))

            # Batch decision
            if len(pending_segments) < BATCH_GROUP_SIZE:
                continue  # accumulate until batch size hit

            # Inference call (re-use existing manager batch processor)
            batch_results = multi_stall_manager._process_segment_batch(
                pending_segments,
                pending_times,
                runtime_cfg.stable_id,
                runtime_cfg.stall_id,
                file_path=None,  # type: ignore
            )
            pending_segments.clear()
            pending_times.clear()

            for seg in batch_results:
                midpoint_rel = seg.start_time + (seg.end_time - seg.start_time) / 2.0
                event_ts = (stream_start_ts + timedelta(seconds=midpoint_rel)).replace(
                    microsecond=0
                )  # type: ignore

                label_dist = seg.label_probabilities or {}
                if not label_dist:
                    continue

                # Top label + probability
                top_label, top_prob = max(label_dist.items(), key=lambda kv: kv[1])

                loud_val = seg.loudness
                event = {
                    "timestamp": event_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "primary_alert": None,  # leave None; can add rules later
                    "confidence": float(top_prob),
                    "contributing_factors": [],
                    "label_distribution": label_dist,
                    "loudness_metrics": {"segment": {"value": loud_val}},
                    "time_context": build_time_context(event_ts),
                }

                # --- ANOMALY INJECTION FOR METAMODEL ---
                # Check if detector flagged an anomaly
                if getattr(seg, "is_anomaly", False):
                    # We inject the anomaly label into the distribution so the metamodel sees it.
                    # We use the configured 'anomaly_abnormal' label value.
                    logger.info(
                        f"Anomaly detected in monitor loop! Score: {getattr(seg, 'anomaly_score', 0.0)}"
                    )
                    # We give it a high probability (1.0) or reuse the score.
                    # Since Metamodel uses confidence to check against floor,
                    # ensuring it's the "dominant" label is safest if we want it to trigger cluster logic.
                    event["label_distribution"]["anomaly_abnormal"] = 1.0
                    # Update top prediction display
                    event["confidence"] = 1.0

                    # (The original label is still in distribution but anomaly overrides as dominant)
                    _score = getattr(seg, "anomaly_score", 0.0)
                    _is_anom = getattr(seg, "is_anomaly", True)
                    event["original_label"] = (
                        f"{top_label} (prob:{top_prob:.2f} score:{_score:.2f} confirmed:{_is_anom})"
                    )

                # Feed metamodel (optional)
                evaluator.evaluate(
                    {
                        "timestamp": event["timestamp"],
                        "label_distribution": label_dist,
                        "loudness_metrics": event["loudness_metrics"],
                        "original_label": event.get("original_label"),
                    }
                )

                if PRINT_EVENTS and runtime_cfg.print_events:
                    print(json.dumps(event), flush=True)

                safe_write(events_path, event)

                if VERBOSE and VERBOSE_TOP_N > 0:
                    top = sorted(label_dist.items(), key=lambda kv: -kv[1])[
                        :VERBOSE_TOP_N
                    ]
                    print(
                        "# VERBOSE_INFER",
                        json.dumps({"t": event["timestamp"], "top": top}),
                        flush=True,
                    )

                segments_emitted += 1
                last_segment_emit_wall = time.time()

            # Rollover flush (daily)
            now = time.time()
            if now - last_flush >= runtime_cfg.flush_interval_sec:
                last_flush = now
                current_date = datetime.utcnow().strftime("%Y%m%d")
                if current_date not in events_path.name:
                    events_path = out_dir / f"segments_{current_date}.jsonl"
                    warnings_path = out_dir / f"warnings_{current_date}.jsonl"

    except KeyboardInterrupt:
        print("# Interrupted", file=sys.stderr)
    finally:
        streamer.stop()
        streamer.join(timeout=2)
        # evaluator._flush_current_night()
        # from smartstablemodel.scripts.metamodel_replay import plot_night_overview, plot_grouped_events
        # plot_night_overview(evaluator, adapt_resolution=False)
        # plot_grouped_events(evaluator, group_labels=False)
        print(
            f"# Shutdown complete (segments emitted={segments_emitted})",
            file=sys.stderr,
        )
        print(f"# Events log: {events_path}", file=sys.stderr)


def run_realtime_monitor(
    stable_id="stable01",
    stall_id="stall01",
    simulate_file=None,
    exit_after=None,
    print_events=True,
    print_warnings=True,
    sample_rate=SAMPLE_RATE,
    model_path: Optional[str] = None,
    label_mapping_path: Optional[str] = None,
    check_for_updates: bool = True,
):
    audio_cfg = AudioConfig(sample_rate=sample_rate, snippet_seconds=SNIPPET_SECONDS)
    runtime_cfg = RuntimeConfig(
        stable_id=stable_id,
        stall_id=stall_id,
        simulate_from_file=Path(simulate_file) if simulate_file else None,
        print_events=print_events,
        print_warnings=print_warnings,
        exit_after_seconds=exit_after,
    )

    # --- Model Update ---
    # Default path for models if not provided
    if not model_path:
        # Use 'models' directory at project root
        base_pkg_dir = Path(__file__).parent.parent
        if (
            base_pkg_dir.name == "smartstablemodel"
            and base_pkg_dir.parent.name == "src"
        ):
            project_root = base_pkg_dir.parent.parent
        else:
            project_root = base_pkg_dir
        default_models_dir = project_root / "models"
    else:
        default_models_dir = Path(model_path).parent

    if check_for_updates:
        info_url = "http://10.212.11.213:30040/api/smartstable/champion/info"
        download_url = "http://10.212.11.213:30040/api/smartstable/champion/tflite"
        updater = ModelUpdateService(str(default_models_dir), info_url, download_url)
        updater.check_and_update()

        # If model_path was NOT provided, try to find the latest after update
        if not model_path:
            latest_model, latest_mapping = updater.find_latest_model()
            if latest_model:
                model_path = str(latest_model)
                label_mapping_path = str(latest_mapping) if latest_mapping else None
                logger.info(f"Using discovered model: {model_path}")
    cfg = load_metamodel_config()
    datastore = DataStore("soundscape_realtime_db.db")
    evaluator = MetaModelDecider(config=cfg, datastore=datastore)

    # --- load multistall manager ---
    try:
        multi_stall_manager = MultiStallDetectorManager(
            model_path=model_path, label_mapping_path=label_mapping_path
        )
        logger.info("multi_detector initialized and ready to serve predictions.")
    except Exception as e:
        raise ImportError("MultiStallDetectorManager initialization failed.") from e

    realtime_loop(evaluator, audio_cfg, runtime_cfg, multi_stall_manager)


# ------------- ENTRY ----------------------------------------------------
def main():
    run_realtime_monitor(
        stable_id=STABLE_ID,
        stall_id=STALL_ID,
        simulate_file=SIMULATE_FILE,
        exit_after=EXIT_AFTER_SECONDS,
        print_events=PRINT_EVENTS,
        print_warnings=PRINT_WARNINGS,
        sample_rate=SAMPLE_RATE,
    )


if __name__ == "__main__":
    main()
