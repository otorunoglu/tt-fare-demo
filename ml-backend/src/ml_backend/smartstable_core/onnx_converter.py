import os
import numpy as np

# PATCH: NumPy 2.0 compatibility for tf2onnx
if not hasattr(np, "cast"):

    class MockCast:
        def __getitem__(self, dtype):
            return lambda x: np.asarray(x, dtype=dtype)

    np.cast = MockCast()

if not hasattr(np, "object"):
    np.object = object

import tensorflow as tf
import tf2onnx
import logging
from typing import Optional, Dict

logger = logging.getLogger("OnnxConverter")


def export_champion_to_onnx(
    keras_model_path: str, output_dir: str, model_architecture: str = "panns"
) -> Optional[Dict[str, str]]:
    """
    Exports the champion model (Embedder + Classifier) to ONNX format.

    Args:
        keras_model_path: Path to the .keras classifier model.
        output_dir: Directory to save the ONNX models.
        model_architecture: Architecture name (default "panns").

    Returns:
        Dictionary with paths to 'embedder' and 'classifier' ONNX files, or None on failure.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # 1. Export Embedder (PyTorch -> ONNX)
        embedder_path = os.path.join(output_dir, "full_pipeline_embedder.onnx")
        if not os.path.exists(embedder_path):
            logger.info(
                f"Exporting {model_architecture} embedder to {embedder_path}..."
            )
            _export_panns_embedder(embedder_path)
        else:
            logger.info(f"Embedder already exists at {embedder_path}")

        # 2. Export Classifier (Keras -> ONNX)
        classifier_path = os.path.join(output_dir, "full_pipeline_classifier.onnx")
        logger.info(
            f"Exporting classifier from {keras_model_path} to {classifier_path}..."
        )
        _export_keras_classifier(keras_model_path, classifier_path)

        return {"embedder": embedder_path, "classifier": classifier_path}

    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {e}")
        # Clean up partial files
        return None


def _export_panns_embedder(output_path: str):
    """Exports PANNs embedder to ONNX."""
    import torch
    from ml_audio_core.embedders.panns_embedder import PannsEmbedder

    device = "cpu"
    checkpoint_path = os.environ.get(
        "PANNS_CHECKPOINT_DIR", "panns_data/Cnn14_16k_mAP=0.438.pth"
    )

    if not os.path.exists(checkpoint_path):
        # Fallback for local testing if env var not set or path invalid
        possible_paths = [
            checkpoint_path,
            "/app/panns_data/Cnn14_16k_mAP=0.438.pth",
            os.path.join(os.getcwd(), "panns_data", "Cnn14_16k_mAP=0.438.pth"),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                checkpoint_path = p
                break

    logger.info(f"Loading PANNs embedder from {checkpoint_path}")

    embedder = PannsEmbedder(
        checkpoint_path=checkpoint_path,
        device=device,
        onnx_export_mode=True,
    )

    # The embedder.model is an AudioTagging wrapper. The actual PyTorch model is inside .model
    base_model = embedder.model.model
    # base_model.eval() # Already eval?

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
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_audio"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_audio": {0: "batch_size", 1: "time"},
            "embeddings": {0: "batch_size"},
        },
    )
    logger.info(f"Embedder exported to {output_path}")


def _export_keras_classifier(keras_model_path: str, output_path: str):
    """Exports Keras classifier to ONNX."""

    logger.info(f"Loading Keras model from {keras_model_path}...")
    model = tf.keras.models.load_model(keras_model_path, compile=False)

    # Input shape for the classifier (embeddings)
    # Assuming PANNs embeddings size of 2048
    spec = (tf.TensorSpec((None, 2048), tf.float32, name="embeddings"),)

    logger.info("Converting Keras model to ONNX...")
    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=17, output_path=output_path
    )
    logger.info(f"Classifier exported to {output_path}")
