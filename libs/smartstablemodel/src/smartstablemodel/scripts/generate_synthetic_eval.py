import argparse
import logging
import numpy as np
import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add source directories to path if running from an unusual location
sys.path.append("/home/johannesgeisler/dev/ml_audio_core/src")
sys.path.append("/home/johannesgeisler/dev/smartstablemodel/src")
sys.path.append("/home/johannesgeisler/dev/label-studio-setup/src")

from ml_audio_core.synthesizer import SoundscapeSynthesizer
from ml_audio_core.audio_processing import AudioProcessor
from ml_audio_core.evaluation import calculate_segment_based_metrics
from smartstablemodel.soundscape_classifier import SoundscapeClassifier

# Import MLflow helper
try:
    from ml_backend.smartstable_core.mlflow_support import download_test_pool_manifest
except ImportError:
    # Attempt to find it if path above didn't work
    print(
        "Warning: Could not import download_test_pool_manifest. MLflow functionality will be limited."
    )
    download_test_pool_manifest = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("generate_synthetic_eval")


def load_samples_from_manifest(
    manifest: Dict[str, List[str]], audio_dirs: List[Path], processor: AudioProcessor
):
    """Load audio segments for each label in the manifest."""
    pool = {}
    total_loaded = 0

    for label, sample_ids in manifest.items():
        pool[label] = []
        logger.info(
            f"Loading samples for label '{label}' (count: {len(sample_ids)})..."
        )

        for sid in sample_ids:
            try:
                # sid format: filename#offset (or cv#lang#path)
                if sid.startswith("cv#"):
                    # Special handling for Common Voice ids if they are just paths
                    # For now assume sid contains enough info or use a search strategy
                    continue

                parts = sid.split("#")
                if len(parts) < 2:
                    continue

                rel_path = parts[0]
                offset = float(parts[1])

                # Search for file in audio_dirs
                found_path = None
                for d in audio_dirs:
                    full_path = d / rel_path
                    if full_path.exists():
                        found_path = full_path
                        break

                if not found_path:
                    # logger.debug(f"Could not find audio file for {sid}")
                    continue

                # Load 2-second segment
                audio, _ = processor.load_and_resample_audio(
                    str(found_path), offset=offset, duration=2.0
                )
                pool[label].append((audio, sid))
                total_loaded += 1

            except Exception as e:
                logger.warning(f"Failed to load sample {sid}: {e}")

    logger.info(f"Loaded {total_loaded} unique samples from manifest into memory pool.")
    return pool


def run_evaluation(
    classifier: SoundscapeClassifier, audio: np.ndarray, ground_truth: List[Dict]
):
    """Run classifier on synthetic track and calculate SED metrics."""
    # Split audio into 2-second windows with overlap to match typical inference
    processor = classifier.audio_processor
    # For SED evaluation, we might want no overlap to have distinct segments
    windows, times, ids = processor.create_sliding_windows(
        audio, segment_duration=2.0, overlap=0.0
    )

    logger.info(f"Running inference on {len(windows)} segments...")
    # Inference (returns probability dicts)
    results = classifier.infer_audio_segment(
        windows, batch_size=32, return_probabilities=True
    )

    # Prepare GT and Predictions per segment
    gt_segments = []
    pred_segments = []

    for i, (start_time, end_time) in enumerate(times):
        # GT: Find all events that overlap this segment
        gt_labels = []
        for event in ground_truth:
            # Simple overlap check: if event overlaps significantly with [start_time, end_time]
            e_start = event["start_time"]
            e_end = event["end_time"]

            # Intersection over segment
            overlap_start = max(start_time, e_start)
            overlap_end = min(end_time, e_end)

            if overlap_end > overlap_start:
                # If event covers more than 10% of the segment, count it
                if (overlap_end - overlap_start) > 0.1 * (end_time - start_time):
                    gt_labels.append(event["label"])

        if not gt_labels:
            gt_labels = ["uncertain"]

        gt_segments.append(gt_labels)

        # Pred: Take top prediction above threshold (or many if sigmoid)
        p_dict = results[i]
        top_label, top_conf = max(p_dict.items(), key=lambda kv: kv[1])

        if top_conf > 0.5:
            pred_segments.append([top_label])
        else:
            pred_segments.append(["uncertain"])

    # Calculate SED Metrics
    metrics = calculate_segment_based_metrics(gt_segments, pred_segments)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic evaluation track from MLflow Test Pool"
    )
    parser.add_argument(
        "--run_id", type=str, required=True, help="MLflow Run ID to get manifest from"
    )
    parser.add_argument(
        "--audio_dirs",
        type=str,
        nargs="+",
        default=["/data/audio"],
        help="Directories to search for audio files",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=600,
        help="Duration of synthetic track in seconds (default: 10 min)",
    )
    parser.add_argument(
        "--events_per_min",
        type=float,
        default=5,
        help="Average foreground events per minute",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="synthetic_eval.wav",
        help="Output path for WAV file",
    )

    args = parser.parse_args()

    if not download_test_pool_manifest:
        logger.error("MLflow support not available. Check PYTHONPATH.")
        return

    # 1. Download manifest
    manifest = download_test_pool_manifest(args.run_id)
    if not manifest:
        logger.error(f"Could not download manifest for run {args.run_id}")
        return

    # 2. Setup classifier and processor
    # We need a classifier to load the model from the same run?
    # For now assume we use a local model or the one from the run
    # (Loading model from run_id is a separate task, we'll assume a path or generic load)
    # TODO: Integration for loading specific model versions

    processor = AudioProcessor(sr=16000, win_seconds=2.0)
    audio_dirs = [Path(d) for d in args.audio_dirs]

    # 3. Load samples into memory
    sample_pool = load_samples_from_manifest(manifest, audio_dirs, processor)

    # 4. Synthesize track
    logger.info(f"Generating synthetic soundscape of {args.duration}s...")
    synthesizer = SoundscapeSynthesizer(sr=16000)
    audio, ground_truth = synthesizer.generate_random_soundscape(
        duration_seconds=args.duration,
        sample_pool=sample_pool,
        events_per_minute=args.events_per_min,
    )

    # 5. Save track
    synthesizer.save_track(audio, ground_truth, args.output_path)

    logger.info("Synthetic track generation complete.")
    logger.info(f"Result: {args.output_path}")
    logger.info(f"Annotations: {args.output_path.replace('.wav', '.json')}")


if __name__ == "__main__":
    main()
