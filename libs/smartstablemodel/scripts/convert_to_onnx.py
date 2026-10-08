#!/usr/bin/env python3
"""
Script to convert the PyTorch PANNs Embedder and Keras Classifier into a single ONNX model.
Usage:
    python convert_to_onnx.py --model_path /path/to/model.keras --output full_pipeline.onnx

Prerequisites:
    pip install tf2onnx onnx onnxruntime
"""

import argparse
import logging
import os
import sys
import tempfile
import subprocess
from pathlib import Path

# Fix path to find ml_audio_core
sys.path.append(str(Path(__file__).parent.parent.parent / "ml_audio_core" / "src"))

# NOTE: We avoid top-level heavy imports (torch, tensorflow) here
# to prevents deadlocks when running the orchestration main()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("APP")


def run_export_embedder(output_path: str):
    """Run embedder export in a separate process."""
    cmd = [
        sys.executable,
        __file__,
        "--action",
        "export_embedder",
        "--output",
        output_path,
    ]
    subprocess.check_call(cmd)


def run_export_classifier(model_path: str, output_path: str):
    """Run classifier export in a separate process."""
    cmd = [
        sys.executable,
        __file__,
        "--action",
        "export_classifier",
        "--model_path",
        model_path,
        "--output",
        output_path,
    ]
    subprocess.check_call(cmd)


def _execute_embedder_export(output_path: str, device: str = "cpu"):
    # Heavy imports only inside the worker process
    import torch
    import sys

    # Setup paths again for the subprocess
    current_dir = Path(__file__).resolve().parent
    workspace_root = current_dir.parent.parent.parent
    if str(workspace_root) not in sys.path:
        sys.path.append(str(workspace_root / "ml_audio_core" / "src"))
        sys.path.append(str(workspace_root / "smartstablemodel" / "src"))

    from ml_audio_core.embedders.panns_embedder import PannsEmbedder

    logger.info("Exporting PANNs Embedder to ONNX...")
    embedder = PannsEmbedder(
        checkpoint_path="panns_data/Cnn14_16k_mAP=0.438.pth",
        device=device,
        onnx_export_mode=True,
    )
    # The embedder.model is an AudioTagging wrapper. The actual PyTorch model is inside .model
    base_model = embedder.model.model
    base_model.eval()

    # Wrap the model to ensure we get exactly the embedding output
    class EmbedderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            out = self.model(x)
            # Access 'embedding' key from the output dictionary
            return out["embedding"]

    wrapped_model = EmbedderWrapper(base_model)

    dummy_input = torch.randn(1, 32000, requires_grad=False)

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,  # Newer opset for STFT
        do_constant_folding=True,
        input_names=["input_audio"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_audio": {0: "batch_size", 1: "time"},
            "embeddings": {0: "batch_size"},
        },
    )
    logger.info(f"Embedder exported to {output_path}")


def _execute_classifier_export(keras_model_path: str, output_path: str):
    # Heavy imports only inside the worker process
    import numpy as np

    # PATCH: NumPy 2.0 compatibility for tf2onnx
    if not hasattr(np, "cast"):

        class MockCast:
            def __getitem__(self, dtype):
                return lambda x: np.asarray(x, dtype=dtype)

        np.cast = MockCast()

    import tensorflow as tf
    import tf2onnx

    # Disable GPU to be safe
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    logger.info(f"Loading Keras model from {keras_model_path}...")
    model = tf.keras.models.load_model(keras_model_path, compile=False)

    spec = (tf.TensorSpec((None, 2048), tf.float32, name="embeddings"),)

    logger.info("Converting Keras model to ONNX...")
    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=17, output_path=output_path
    )
    logger.info(f"Classifier exported to {output_path}")


def merge_onnx_models(embedder_path, classifier_path, output_path):
    import onnx
    from onnx.compose import merge_models

    logger.info("Merging and Simplifying models...")
    embedder = onnx.load(embedder_path)
    classifier = onnx.load(classifier_path)

    # Normalize IR versions to the higher one to allow merging
    max_ir = max(embedder.ir_version, classifier.ir_version)
    embedder.ir_version = max_ir
    classifier.ir_version = max_ir
    logger.info(f"Normalized IR versions to {max_ir}")

    emb_out = embedder.graph.output[0].name
    cls_in = classifier.graph.input[0].name

    if emb_out != cls_in:
        logger.info(
            f"Renaming classifier input {cls_in} to match embedder output {emb_out}"
        )
        classifier.graph.input[0].name = emb_out

    combined_model = merge_models(
        embedder, classifier, io_map=[(emb_out, classifier.graph.input[0].name)]
    )
    # Simplification is optional. If onnxsim is unavailable, keep going with the
    # merged model so export still succeeds in lean/runtime-only environments.
    try:
        from onnxsim import simplify  # pip install onnx-simplifier

        model_simp, check = simplify(combined_model)
        if check:
            onnx.save(model_simp, output_path)
            logger.info(f"Simplified model saved to {output_path}")
            return

        logger.warning("Simplification check failed, saving raw merged model.")
    except Exception as exc:
        logger.warning(
            "onnx simplification skipped (%s). Saving raw merged model.", exc
        )

    onnx.save(combined_model, output_path)


def main():
    script_dir = Path(__file__).parent.resolve()
    # Default Paths
    default_keras = script_dir.parent / "models" / "stable_panns_20260122_150457.keras"
    # Target directory for the stable-ml-monitor-setup project
    default_out_dir = script_dir.parent.parent / "stable-ml-monitor-setup" / "models"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=["orchestrate", "export_embedder", "export_classifier"],
        default="orchestrate",
    )
    parser.add_argument("--model_path", type=str, default=str(default_keras))

    # For orchestrator: directory. For workers: specific file path.
    parser.add_argument("--output", type=str, default=str(default_out_dir))

    parser.add_argument("--force-embedder", action="store_true")
    parser.add_argument(
        "--verify", type=str, help="Audio file for partial verification"
    )

    args = parser.parse_args()

    # --- Worker Modes ---
    if args.action == "export_embedder":
        _execute_embedder_export(args.output)
        return

    if args.action == "export_classifier":
        _execute_classifier_export(args.model_path, args.output)
        return

    # --- Orchestrator Mode ---

    # 1. Setup Paths
    if not args.model_path:
        args.model_path = str(default_keras)

    out_dir = Path(args.output)
    if out_dir.suffix:
        # If user gave a file path (legacy usage), take parent
        out_dir = out_dir.parent

    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = "full_pipeline"
    emb_onnx = out_dir / f"{base_name}_embedder.onnx"
    cls_onnx = out_dir / f"{base_name}_classifier.onnx"

    logger.info(f"Target Embedder: {emb_onnx}")
    logger.info(f"Target Classifier: {cls_onnx}")

    # 2. Export Embedder (if needed)
    if emb_onnx.exists() and not args.force_embedder:
        logger.info(f"Embedder found at {emb_onnx}, skipping export.")
        logger.info("Use --force-embedder to overwrite.")
    else:
        logger.info("Exporting Embedder...")
        run_export_embedder(str(emb_onnx))

    # 3. Export Classifier (Always)
    logger.info("Exporting Classifier...")
    run_export_classifier(args.model_path, str(cls_onnx))

    logger.info("Done exporting separate models.")

    # 4. Verification (Optional)
    if args.verify:
        verify_script = script_dir / "verify_keras_vs_onnx.py"
        if not verify_script.exists():
            logger.error(f"Verification script not found at {verify_script}")
            return

        logger.info(f"Running verification with {args.verify}...")

        cmd = [
            sys.executable,
            str(verify_script),
            "--audio",
            args.verify,
            "--keras_model",
            args.model_path,
            "--onnx_embedder",
            str(emb_onnx),
            "--onnx_classifier",
            str(cls_onnx),
        ]

        try:
            subprocess.check_call(cmd)
            logger.info("Verification PASSED.")
        except subprocess.CalledProcessError:
            logger.error("Verification FAILED.")
            sys.exit(1)


if __name__ == "__main__":
    main()
