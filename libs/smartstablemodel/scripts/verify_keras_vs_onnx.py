#!/usr/bin/env python3
"""
Verify congruence between Keras/PyTorch inference and ONNX inference.
Usage:
    python verify_keras_vs_onnx.py --audio path/to/audio.flac
"""

import argparse
import sys
import json
import logging
from pathlib import Path
import numpy as np
import librosa
from collections import Counter

# Add src paths
current_dir = Path(__file__).resolve().parent
workspace_root = current_dir.parent.parent
sys.path.insert(0, str(workspace_root / "ml_audio_core" / "src"))
sys.path.insert(0, str(workspace_root / "smartstablemodel" / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("VERIFY")


def main():
    parser = argparse.ArgumentParser(description="Verify Keras vs ONNX inference")

    # Default paths based on workspace structure
    default_keras_model = (
        workspace_root
        / "smartstablemodel"
        / "models"
        / "stable_panns_20260205_111520.keras"
    )
    # Merged model (legacy/optional)
    default_audio = (
        workspace_root
        / "stable-edge-monitor"
        / "data"
        / "audio"
        / "stable03_stall01_horse01_20251214_000008_mic.flac"
    )
    # Split models (preferred)
    default_onnx_embedder = (
        workspace_root
        / "stable-edge-monitor"
        / "data"
        / "models"
        / "full_pipeline_embedder.onnx"
    )
    default_onnx_classifier = (
        workspace_root
        / "stable-edge-monitor"
        / "data"
        / "models"
        / "full_pipeline_classifier.onnx"
    )

    default_label_mapping = (
        workspace_root
        / "stable-edge-monitor"
        / "data"
        / "models"
        / "label_mapping.json"
    )

    parser.add_argument(
        "--audio", type=str, default=str(default_audio), help="Path to audio file"
    )

    parser.add_argument("--keras_model", type=str, default=str(default_keras_model))
    parser.add_argument(
        "--onnx_model",
        type=str,
        default=None,
        help="Path to merged ONNX model (optional)",
    )
    parser.add_argument("--onnx_embedder", type=str, default=str(default_onnx_embedder))
    parser.add_argument(
        "--onnx_classifier", type=str, default=str(default_onnx_classifier)
    )
    parser.add_argument("--label_mapping", type=str, default=str(default_label_mapping))

    args = parser.parse_args()

    # --- 1. Setup ---
    logger.info(f"Audio: {args.audio}")
    logger.info(f"Keras Model: {args.keras_model}")

    # Load Label Mapping
    with open(args.label_mapping, "r") as f:
        mapping = json.load(f)
        idx_to_label = {int(k): v for k, v in mapping["idx_to_label"].items()}
    logger.info(f"Loaded {len(idx_to_label)} labels")

    # Load Audio
    logger.info("Loading audio...")
    audio, sr = librosa.load(args.audio, sr=16000)
    logger.info(f"Audio loaded: {len(audio)} samples ({len(audio) / 16000:.1f}s)")

    # Create Segments
    window_samples = int(2.0 * 16000)
    hop_samples = int(1.0 * 16000)
    segments = []
    times = []
    for start in range(0, len(audio) - window_samples + 1, hop_samples):
        segments.append(audio[start : start + window_samples])
        times.append(start / 16000.0)

    limit = 200  # Limit to first 20 segments for brevity
    segments = segments[:limit]
    times = times[:limit]
    logger.info(f"Processing first {limit} segments")

    # --- 2. Run Keras/PyTorch (Ground Truth) ---
    logger.info("--- Running Keras/PyTorch Inference ---")

    # Imports inside function to allow script to run even if some deps missing
    import tensorflow as tf
    from ml_audio_core.embedders.panns_embedder import PannsEmbedder

    # Load Embedder
    # Assuming standard checkpoint location
    panns_checkpoint = (
        workspace_root
        / "stable-ml-monitor-setup"
        / "panns_data"
        / "Cnn14_16k_mAP=0.438.pth"
    )
    embedder = PannsEmbedder(checkpoint_path=panns_checkpoint, device="cpu")

    # Load Classifier
    keras_model = tf.keras.models.load_model(args.keras_model, compile=False)

    keras_results = []
    for i, segment in enumerate(segments):
        # Embed
        segment_batch = np.expand_dims(segment, axis=0)
        emb = embedder.embed(segment_batch)
        # Classify
        probs = keras_model.predict(emb, verbose=0)
        pred_idx = np.argmax(probs[0])
        confidence = probs[0][pred_idx]
        label = idx_to_label.get(pred_idx, f"unknown_{pred_idx}")
        keras_results.append((label, confidence))

    # --- 3. Run ONNX Inference ---
    logger.info("--- Running ONNX Inference ---")
    import onnxruntime as ort

    onnx_results = []

    # Check if we should use split models or merged model
    use_split = False
    if (
        args.onnx_embedder
        and Path(args.onnx_embedder).exists()
        and args.onnx_classifier
        and Path(args.onnx_classifier).exists()
    ):
        use_split = True
    elif args.onnx_model and Path(args.onnx_model).exists():
        use_split = False
    else:
        # Relaxed check: if defaults exist, use them
        if Path(args.onnx_embedder).exists() and Path(args.onnx_classifier).exists():
            use_split = True
        else:
            logger.error("No valid ONNX models found.")
            sys.exit(1)

    if use_split:
        logger.info(
            f"Using Split Models:\n  Embedder: {args.onnx_embedder}\n  Classifier: {args.onnx_classifier}"
        )
        emb_session = ort.InferenceSession(
            args.onnx_embedder, providers=["CPUExecutionProvider"]
        )
        cls_session = ort.InferenceSession(
            args.onnx_classifier, providers=["CPUExecutionProvider"]
        )

        emb_in_name = emb_session.get_inputs()[0].name
        cls_in_name = cls_session.get_inputs()[0].name

        for i, segment in enumerate(segments):
            # Embedder
            inp = np.expand_dims(segment, axis=0).astype(np.float32)
            emb_out = emb_session.run(None, {emb_in_name: inp})[0]
            # Classifier
            cls_out = cls_session.run(None, {cls_in_name: emb_out})[0]

            probs = cls_out
            pred_idx = np.argmax(probs[0])
            confidence = probs[0][pred_idx]
            label = idx_to_label.get(pred_idx, f"unknown_{pred_idx}")
            onnx_results.append((label, confidence))

    else:
        logger.info(f"Using Merged Model: {args.onnx_model}")
        session = ort.InferenceSession(
            args.onnx_model, providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name

        for i, segment in enumerate(segments):
            # Format for ONNX: (batch, time)
            inp = np.expand_dims(segment, axis=0).astype(np.float32)
            outputs = session.run(None, {input_name: inp})
            probs = outputs[0]
            pred_idx = np.argmax(probs[0])
            confidence = probs[0][pred_idx]
            label = idx_to_label.get(pred_idx, f"unknown_{pred_idx}")
            onnx_results.append((label, confidence))

    # --- 4. Compare ---
    print("\n=== Comparison Report ===")
    print(
        f"{'Time':<10} | {'Keras Label':<15} {'Conf':<6} | {'ONNX Label':<15} {'Conf':<6} | {'Match'}"
    )
    print("-" * 75)

    matches = 0
    for i, (time, k_res, o_res) in enumerate(zip(times, keras_results, onnx_results)):
        k_label, k_conf = k_res
        o_label, o_conf = o_res
        match = "✅" if k_label == o_label else "❌"
        if k_label == o_label:
            matches += 1

        print(
            f"{time:<10.1f} | {k_label:<15} {k_conf:<6.2f} | {o_label:<15} {o_conf:<6.2f} | {match}"
        )

    print("-" * 75)
    print(
        f"Agreement: {matches}/{len(segments)} ({matches / len(segments) * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
