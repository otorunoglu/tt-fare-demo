import logging
import os
from smartstablemodel.multi_stall_manager import MultiStallDetectorManager
from ml_backend.smartstable_core.mlflow_support import (
    resolve_model_paths,
    get_production_model_info,
    get_registered_model_name,
)

logger = logging.getLogger("SmartStableManagerLoader")

_manager = None
_model_info = {
    "source": "default",
    "model_path": None,
    "mlflow_version": None,
    "mlflow_run_id": None,
    "model_name": "soundscape_classifier_default",
    "model_architecture": "panns",
}


def get_multistall_manager():
    global _manager, _model_info

    if _manager is not None:
        return _manager

    # 1. Determine Model Architecture (from config)
    model_architecture = "panns"
    try:
        from smartstablemodel.config.metamodel_config import MetaModelConfig

        config = MetaModelConfig.from_toml()
        model_architecture = config.model_parameters.get(
            "default_model_architecture", "panns"
        )
    except Exception as e:
        logger.warning(f"Failed to load default model architecture: {e}")

    _model_info["model_architecture"] = model_architecture

    # 2. Resolve Model Paths
    model_path, label_map = resolve_model_paths(model_architecture=model_architecture)

    logger.info(
        f"Resolved paths for {model_architecture} - model: {model_path}, label_map: {label_map}"
    )

    if model_path:
        logger.info(f"Loading MLflow model from: {model_path}")

        # CRITICAL: Crash if no label mapping - we learned this the hard way!
        if not label_map:
            error_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   🚨 CRITICAL: NO LABEL MAPPING FROM MLFLOW 🚨                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ MLflow model downloaded but NO LABEL MAPPING was found!
║
║ Model path: {model_path}
║
║ WITHOUT the correct label mapping, the model will predict garbage:
║   - Model predicts index 3 = 'neigh' (what it learned)
║   - Fallback mapping says index 3 = 'bird_tweet' (random order from labels.json)
║   - Result: ALL predictions are WRONG and you won't notice for weeks!
║
║ TO FIX THIS:
║   1. Check that your MLflow run has 'label_mapping' or 'model_artifacts' artifact
║   2. The artifact must contain a *_label_mapping.json file
║   3. If missing, retrain the model - it will save the mapping automatically
║
║ REFUSING TO START WITH WRONG LABELS. Fix the model artifacts first.
╚══════════════════════════════════════════════════════════════════════════════╝
"""
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"Using label mapping from: {label_map}")

        _manager = MultiStallDetectorManager(
            model_path=model_path,
            label_mapping_path=label_map,
            model_version=_model_info["model_name"],
            model_architecture=model_architecture,
        )

        # Verify the classifier loaded the correct labels
        clf = _manager.get_soundscape_classifier()
        if clf.idx_to_label:
            logger.info(
                f"Classifier loaded with {len(clf.idx_to_label)} classes: {list(clf.idx_to_label.values())[:5]}..."
            )
        else:
            error_msg = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                🚨 CRITICAL: CLASSIFIER HAS NO LABEL MAPPING 🚨                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ The SoundscapeClassifier was initialized but has NO idx_to_label mapping!
║
║ This means predictions will return raw indices instead of label names,
║ or worse, use a fallback mapping that maps to WRONG labels.
║
║ The label_mapping file might be corrupted or in wrong format.
║ Check the file and ensure it has label_to_idx, idx_to_label, valid_labels.
╚══════════════════════════════════════════════════════════════════════════════╝
"""
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # CRITICAL: Validate that model output size matches label mapping size
        try:
            model_output_size = clf.model.output_shape[-1]
            label_mapping_size = len(clf.idx_to_label)

            if model_output_size != label_mapping_size:
                error_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           🚨 CRITICAL: MODEL/LABEL MAPPING SIZE MISMATCH 🚨                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Model output size: {model_output_size} classes
║ Label mapping size: {label_mapping_size} classes
║
║ THE MODEL AND LABEL MAPPING DO NOT MATCH!
║
║ This means the model was trained with {model_output_size} labels but the
║ label_mapping artifact only has {label_mapping_size} labels.
║
║ When the model predicts class index {label_mapping_size} or higher, we get:
║   KeyError: {label_mapping_size}
║
║ This is usually caused by:
║   1. Label mapping not being updated during training
║   2. Wrong label_mapping artifact uploaded to MLflow
║   3. min_samples_per_class filtering creating fewer labels than expected
║
║ TO FIX:
║   1. Re-train the model with proper label_mapping saving
║   2. Or manually fix the label_mapping in MLflow to have {model_output_size} classes
║   3. Check training logs to see which {model_output_size} labels were actually used
║
║ WARNING: Proceeding despite mismatch to allow re-training.
║ Inference WILL FAIL if the model predicts an unknown class.
╚══════════════════════════════════════════════════════════════════════════════╝
"""
                logger.error(error_msg)
                # raise RuntimeError(error_msg)  <-- DISABLED to allow startup for retraining
            else:
                logger.info(
                    f"✓ Model output ({model_output_size}) matches label mapping ({label_mapping_size})"
                )
        except Exception as e:
            if "CRITICAL" in str(e):
                raise
            logger.warning(f"Could not validate model/label mapping sizes: {e}")

        # Get MLflow model info for logging
        _model_info["source"] = "mlflow"
        _model_info["model_path"] = model_path

        try:
            prod_info = get_production_model_info(model_architecture=model_architecture)
            if prod_info:
                _model_info["mlflow_version"] = prod_info.get("version")
                _model_info["mlflow_run_id"] = prod_info.get("run_id")
                # Use dynamic name
                model_name = get_registered_model_name(model_architecture)
                _model_info["model_name"] = f"{model_name}_v{prod_info.get('version')}"
                logger.info(
                    f"MLflow model: {_model_info['model_name']} (run_id: {prod_info.get('run_id')})"
                )
        except Exception as e:
            logger.warning(f"Could not get MLflow model info: {e}")
            _model_info["model_name"] = os.path.basename(model_path).replace(
                ".keras", ""
            )
    else:
        logger.info("Loading default local SmartStable model")
        _manager = MultiStallDetectorManager()
        _model_info["source"] = "default"
        _model_info["model_name"] = "soundscape_classifier_default"

    logger.info(
        f"Loaded model: {_model_info['model_name']} (source: {_model_info['source']})"
    )
    return _manager


def get_model_info() -> dict:
    """Get information about the currently loaded model."""
    return _model_info.copy()


def get_model_version_string() -> str:
    """Get a version string suitable for Label Studio predictions."""
    return _model_info.get("model_name", "soundscape_classifier")
