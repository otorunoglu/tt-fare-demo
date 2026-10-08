#!/usr/bin/env python3
"""Run edge-like inference on a single audio file.

This uses the same benchmark config and model pipeline as mic_benchmark.py,
but processes an existing audio file instead of a live microphone stream.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import mic_benchmark as mb

from mic_benchmark import (
    choose_top_label,
    load_benchmark_config,
    load_label_mapping,
    loudness_dbfs,
    parse_model_config,
    repo_root_from_script,
    require_runtime_dependencies,
    setup_logging,
    logger,
)





def run_file_inference(audio_file: Path, cfg: dict) -> int:
    require_runtime_dependencies(needs_audio_input=False)

    embedder_path: Path = cfg["embedder"]
    classifier_path: Path = cfg["classifier"]
    label_mapping_path: Path = cfg["label_mapping"]
    model_config_path: Path = cfg["model_config"]

    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    if not embedder_path.exists() or not classifier_path.exists() or not label_mapping_path.exists():
        raise FileNotFoundError(
            "Model files missing. Check paths in benchmark_config.toml under [paths]."
        )

    sample_rate = int(cfg["sample_rate"])
    segment_duration = float(cfg["segment_duration"])
    segment_overlap = float(cfg["segment_overlap"])
    log_threshold = float(cfg["log_threshold"])

    segment_samples = int(sample_rate * segment_duration)
    hop_samples = int(segment_samples * (1.0 - segment_overlap))
    if hop_samples <= 0:
        raise ValueError("segment_overlap must be < 1.0")

    silence_threshold, classifier_threshold = parse_model_config(model_config_path)
    logger.info(
        "Using thresholds: silence=%.1f dBFS, classifier=%.2f",
        silence_threshold,
        classifier_threshold,
    )

    label_map = load_label_mapping(label_mapping_path)

    logger.info("Loading ONNX sessions...")
    embedder_sess = mb.ort.InferenceSession(str(embedder_path), providers=["CPUExecutionProvider"])
    classifier_sess = mb.ort.InferenceSession(str(classifier_path), providers=["CPUExecutionProvider"])
    emb_in_name = embedder_sess.get_inputs()[0].name
    cls_in_name = classifier_sess.get_inputs()[0].name

    logger.info("Loading audio with Rust preprocessor: %s", audio_file)
    audio, sr = mb.audio_preprocessing.load_audio(str(audio_file), sample_rate)
    audio = np.asarray(audio, dtype=np.float32)
    if int(sr) != sample_rate:
        logger.warning("Loaded sample rate %s differs from target %s", sr, sample_rate)

    if audio.size < segment_samples:
        logger.warning("Audio shorter than one segment; no inference windows produced")
        return 0

    total = 0
    for start in range(0, audio.size - segment_samples + 1, hop_samples):
        end = start + segment_samples
        segment = audio[start:end]

        center_sample = start + (segment_samples // 2)
        center_seconds = center_sample / sample_rate

        dbfs = loudness_dbfs(segment)
        if dbfs < silence_threshold:
            if 1.0 >= log_threshold:
                print(
                    json.dumps(
                        {
                            "audio_file": str(audio_file),
                            "segment_start_s": start / sample_rate,
                            "segment_end_s": end / sample_rate,
                            "segment_center_s": center_seconds,
                            "label": "silence",
                            "confidence": 1.0,
                            "loudness": dbfs,
                        }
                    )
                )
            total += 1
            continue

        audio_vec = np.clip(segment.astype(np.float32), -1.0, 1.0)
        emb = embedder_sess.run(None, {emb_in_name: audio_vec.reshape(1, -1)})
        embedding = np.asarray(emb[0], dtype=np.float32)
        cls = classifier_sess.run(None, {cls_in_name: embedding})
        probs = np.asarray(cls[0], dtype=np.float32).reshape(-1)

        label_probs, top_label, top_prob = choose_top_label(probs, label_map)

        if top_prob < classifier_threshold:
            forced_prob = float(1.0 - top_prob)
            label_probs = {"unsure": forced_prob}
            top_label = "unsure"
            top_prob = forced_prob

        if top_prob >= log_threshold:
            print(
                json.dumps(
                    {
                        "audio_file": str(audio_file),
                        "segment_start_s": start / sample_rate,
                        "segment_end_s": end / sample_rate,
                        "segment_center_s": center_seconds,
                        "label": top_label,
                        "confidence": top_prob,
                        "loudness": dbfs,
                        "label_probs": label_probs,
                    }
                )
            )

        total += 1

    logger.info("Processed %d segments from %s", total, audio_file)
    return 0


def main(script_path: Path) -> int:
    config_path = script_path.parent / "benchmark_config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    repo_root = repo_root_from_script(script_path)
    audio_file = Path(input("Enter audio file path: ").strip().strip('"'))

    cfg = load_benchmark_config(config_path, repo_root)
    setup_logging(cfg["log_level"])
    logger.info("Loaded config from %s", config_path)
    logger.info("Using audio file %s", audio_file)

    return run_file_inference(audio_file, cfg)


if __name__ == "__main__":
    raise SystemExit(main(Path(__file__)))
