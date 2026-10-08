#!/usr/bin/env python3
"""Microphone benchmark inference driven by a local TOML config file."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

logger = logging.getLogger("mic_benchmark")
sd: Any = None
ort: Any = None
audio_preprocessing: Any = None


def require_runtime_dependencies(*, needs_audio_input: bool = True) -> None:
    global sd, ort, audio_preprocessing
    if needs_audio_input and sd is None:
        try:
            import sounddevice as _sd  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "sounddevice (and PortAudio) is required. "
                "Install system PortAudio and `pip install sounddevice`."
            ) from exc
        sd = _sd

    if ort is None:
        try:
            import onnxruntime as _ort  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "onnxruntime is required. Install with `pip install onnxruntime`."
            ) from exc
        ort = _ort

    if audio_preprocessing is None:
        try:
            import audio_preprocessing as _audio_preprocessing  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "audio_preprocessing module is required. Build/install the Rust Python package first."
            ) from exc
        audio_preprocessing = _audio_preprocessing


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def repo_root_from_script(script_path: Path) -> Path:
    return script_path.resolve().parents[2]


def parse_model_config(config_path: Path) -> tuple[float, float]:
    if not config_path.exists():
        logger.warning("Model config not found at %s, using defaults", config_path)
        return -60.0, 0.9

    with config_path.open("rb") as f:
        cfg = tomllib.load(f)

    mp = cfg.get("model_parameters", {})
    silence_threshold = float(mp.get("silence_threshold", -60.0))
    classifier_thresholds = mp.get("classifier_thresholds", {})
    classifier_threshold = float(classifier_thresholds.get("panns", 0.9))
    return silence_threshold, classifier_threshold


def load_label_mapping(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    mapping = raw.get("idx_to_label", raw)
    return {int(k): str(v) for k, v in mapping.items()}


def resolve_device(device_arg: str | None) -> int | None:
    require_runtime_dependencies(needs_audio_input=True)
    if not device_arg:
        return None

    devices = sd.query_devices()
    if device_arg.isdigit():
        idx = int(device_arg)
        if idx < 0 or idx >= len(devices):
            raise ValueError(f"Device index out of range: {idx}")
        if int(devices[idx].get("max_input_channels", 0)) <= 0:
            raise ValueError(f"Device {idx} is not an input device")
        return idx

    needle = device_arg.lower()
    for idx, dev in enumerate(devices):
        if int(dev.get("max_input_channels", 0)) <= 0:
            continue
        if needle in str(dev["name"]).lower():
            return idx
    raise ValueError(f"No input device matched '{device_arg}'")


def loudness_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -100.0
    rms = float(np.sqrt(np.mean(audio * audio)))
    if rms > 1e-9:
        return max(20.0 * float(np.log10(rms)), -100.0)
    return -100.0


def choose_top_label(probs: np.ndarray, label_map: dict[int, str]) -> tuple[dict[str, float], str, float]:
    label_probs: dict[str, float] = {}
    top_label = "unknown"
    top_prob = -1.0
    for idx, prob in enumerate(probs.tolist()):
        label = label_map.get(idx, f"class_{idx}")
        p = float(prob)
        label_probs[label] = p
        if p > top_prob:
            top_prob = p
            top_label = label
    return label_probs, top_label, top_prob


def resolve_repo_path(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def load_benchmark_config(config_path: Path, repo_root: Path) -> dict[str, Any]:
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    paths = raw.get("paths", {})
    runtime = raw.get("runtime", {})

    cfg: dict[str, Any] = {
        "embedder": resolve_repo_path(
            str(paths.get("embedder", "apps/edge-monitor/data/models/champion_v26_embedder.onnx")),
            repo_root,
        ),
        "classifier": resolve_repo_path(
            str(paths.get("classifier", "apps/edge-monitor/data/models/champion_v26_classifier.onnx")),
            repo_root,
        ),
        "label_mapping": resolve_repo_path(
            str(paths.get("label_mapping", "apps/edge-monitor/data/models/champion_v26_label_mapping.json")),
            repo_root,
        ),
        "model_config": resolve_repo_path(
            str(paths.get("model_config", "config/smart_stable_model_config.toml")),
            repo_root,
        ),
        "device": str(runtime.get("device", "")).strip() or None,
        "sample_rate": int(runtime.get("sample_rate", 16000)),
        "segment_duration": float(runtime.get("segment_duration", 2.0)),
        "segment_overlap": float(runtime.get("segment_overlap", 0.5)),
        "run_seconds": float(runtime.get("run_seconds", 0.0)),
        "log_threshold": float(runtime.get("log_threshold", 0.5)),
        "log_level": str(runtime.get("log_level", "INFO")),
    }
    return cfg


def run_benchmark(cfg: dict[str, Any]) -> int:
    require_runtime_dependencies(needs_audio_input=True)

    embedder_path: Path = cfg["embedder"]
    classifier_path: Path = cfg["classifier"]
    label_mapping_path: Path = cfg["label_mapping"]
    model_config_path: Path = cfg["model_config"]

    if not embedder_path.exists() or not classifier_path.exists() or not label_mapping_path.exists():
        raise FileNotFoundError(
            "Model files missing. Check paths in benchmark_config.toml under [paths]."
        )

    silence_threshold, classifier_threshold = parse_model_config(model_config_path)
    logger.info(
        "Using thresholds: silence=%.1f dBFS, classifier=%.2f",
        silence_threshold,
        classifier_threshold,
    )

    label_map = load_label_mapping(label_mapping_path)

    logger.info("Loading ONNX sessions...")
    embedder_sess = ort.InferenceSession(str(embedder_path), providers=["CPUExecutionProvider"])
    classifier_sess = ort.InferenceSession(str(classifier_path), providers=["CPUExecutionProvider"])
    emb_in_name = embedder_sess.get_inputs()[0].name
    cls_in_name = classifier_sess.get_inputs()[0].name

    device_idx = resolve_device(cfg["device"])
    dev_info = sd.query_devices(device_idx, "input")
    source_sr = int(dev_info["default_samplerate"])
    logger.info("Using input device: %s (index=%s, source_sr=%d)", dev_info["name"], device_idx, source_sr)

    sample_rate = int(cfg["sample_rate"])
    segment_duration = float(cfg["segment_duration"])
    segment_overlap = float(cfg["segment_overlap"])
    run_seconds = float(cfg["run_seconds"])
    log_threshold = float(cfg["log_threshold"])

    segment_samples = int(sample_rate * segment_duration)
    hop_samples = int(segment_samples * (1.0 - segment_overlap))
    if hop_samples <= 0:
        raise ValueError("segment_overlap must be < 1.0")

    read_size = max(1, int(source_sr * 0.1))

    resampled_buffer = np.zeros(0, dtype=np.float32)
    total_samples_processed = 0
    stream_start = datetime.now()
    logger.info("Starting stream. Press Ctrl+C to stop.")

    with sd.InputStream(
        samplerate=source_sr,
        channels=1,
        dtype="float32",
        device=device_idx,
        blocksize=read_size,
    ) as stream:
        try:
            while True:
                if run_seconds > 0 and (datetime.now() - stream_start).total_seconds() >= run_seconds:
                    logger.info("Reached run duration limit (%.1fs)", run_seconds)
                    break

                chunk, overflowed = stream.read(read_size)
                if overflowed:
                    logger.warning("Audio input overflow detected")

                mono = np.asarray(chunk[:, 0], dtype=np.float32)
                resampled = np.asarray(
                    audio_preprocessing.resample_numpy(
                        mono,
                        orig_sr=source_sr,
                        target_sr=sample_rate,
                    ),
                    dtype=np.float32,
                )

                if resampled.size == 0:
                    continue

                resampled_buffer = np.concatenate([resampled_buffer, resampled])

                while resampled_buffer.size >= segment_samples:
                    segment = resampled_buffer[:segment_samples]
                    center_offset = total_samples_processed + (segment_samples // 2)
                    event_ts = stream_start + timedelta(seconds=center_offset / sample_rate)

                    dbfs = loudness_dbfs(segment)
                    if dbfs < silence_threshold:
                        if 1.0 >= log_threshold:
                            print(
                                json.dumps(
                                    {
                                        "timestamp": event_ts.isoformat(),
                                        "label": "silence",
                                        "confidence": 1.0,
                                        "loudness": dbfs,
                                    }
                                )
                            )
                    else:
                        audio_vec = np.clip(segment.astype(np.float32), -1.0, 1.0)
                        emb = embedder_sess.run(None, {emb_in_name: audio_vec.reshape(1, -1)})
                        embedding = np.asarray(emb[0], dtype=np.float32)
                        cls = classifier_sess.run(None, {cls_in_name: embedding})
                        probs = np.asarray(cls[0], dtype=np.float32).reshape(-1)

                        label_probs, top_label, top_prob = choose_top_label(probs, label_map)

                        if top_prob < classifier_threshold:
                            forced_prob = float(1.0 - top_prob)
                            label_probs = {"normal": forced_prob}
                            top_label = "normal"
                            top_prob = forced_prob

                        if top_prob >= log_threshold:
                            print(
                                json.dumps(
                                    {
                                        "timestamp": event_ts.isoformat(),
                                        "label": top_label,
                                        "confidence": top_prob,
                                        "loudness": dbfs,
                                        "label_probs": label_probs,
                                    }
                                )
                            )

                    resampled_buffer = resampled_buffer[hop_samples:]
                    total_samples_processed += hop_samples

        except KeyboardInterrupt:
            logger.info("Stopped by user")

    return 0


def main(script_path: Path) -> int:
    config_path = script_path.parent / "benchmark_config.toml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing config file: {config_path}. Create it from the provided example in this folder."
        )

    repo_root = repo_root_from_script(script_path)
    cfg = load_benchmark_config(config_path, repo_root)
    setup_logging(cfg["log_level"])
    logger.info("Loaded config from %s", config_path)
    return run_benchmark(cfg)


if __name__ == "__main__":
    raise SystemExit(main(Path(__file__)))
