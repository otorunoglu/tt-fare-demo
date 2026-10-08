import tensorflow as tf


def infer_output_units(model: tf.keras.Model) -> int | None:
    """Robustly infer number of output classes (supports list / multi-output)."""
    # 1. Last layer with units
    try:
        last = model.layers[-1]
        if hasattr(last, "units") and isinstance(last.units, int) and last.units > 0:
            return int(last.units)
    except Exception:
        pass
    # 2. model.output_shape (handles list)
    try:
        osh = model.output_shape
        # If list of shapes, pick first shape tuple ending with int
        if isinstance(osh, (list, tuple)):
            shapes = osh if isinstance(osh, list) else [osh]
            for sh in shapes:
                if (
                    isinstance(sh, (list, tuple))
                    and len(sh) >= 2
                    and isinstance(sh[-1], int)
                ):
                    return int(sh[-1])
    except Exception:
        pass
    # 3. model.output (list vs tensor)
    try:
        out = model.output
        outs = out if isinstance(out, (list, tuple)) else [out]
        for tensor in outs:
            dim = getattr(
                getattr(tensor, "shape", None), "__getitem__", lambda *_: None
            )(-1)
            if isinstance(dim, int) and dim > 0:
                return int(dim)
    except Exception:
        pass
    return None


def export_to_tflite(model: tf.keras.Model, output_path: str) -> str:
    """
    Convert a Keras model to TFLite format and save it.

    Args:
        model: The Keras model to convert.
        output_path: Path to save the .tflite file.

    Returns:
        The path where the TFLite model was saved.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    return output_path
