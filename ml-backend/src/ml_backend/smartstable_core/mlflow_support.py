# ml_backend/smartstable_core/mlflow_support.py
"""
MLflow Model Registry support for SmartStable backend.

Provides functions for:
- Checking if MLflow is enabled
- Downloading production models from the registry
- Model registration and promotion
"""

import os
import logging
import re
import numpy as np
import zipfile
import tempfile
from typing import Dict, Any, Optional, List

logger = logging.getLogger("MLflowSupport")

# MLflow is optional - gracefully degrade if not available
try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Model registry features disabled.")

# Alias used to mark the production/champion model in MLflow Model Registry
REGISTERED_MODEL_NAME = "soundscape_classifier"  # DEPRECATED
CHAMPION_ALIAS = "champion"


def get_registered_model_name(model_architecture: str = "panns") -> str:
    """Get the registered model name for a specific architecture."""
    # Ensure architecture is lowercase
    arch = model_architecture.lower()
    return f"soundscape_classifier_{arch}"


def is_mlflow_enabled() -> bool:
    """Check if MLflow tracking is enabled and available."""
    if not MLFLOW_AVAILABLE:
        return False

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        logger.debug("MLFLOW_TRACKING_URI not set. MLflow tracking disabled.")
        return False

    return True


def init_mlflow(experiment_name: str = "soundscape_classifier") -> bool:
    """
    Initialize MLflow tracking.

    Args:
        experiment_name: Name of the MLflow experiment

    Returns:
        True if initialization succeeded, False otherwise
    """
    if not is_mlflow_enabled():
        return False

    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        logger.info(
            f"MLflow initialized: {tracking_uri}, experiment: {experiment_name}"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to initialize MLflow: {e}")
        return False


def get_production_model_info(
    model_architecture: str = "panns",
) -> Optional[Dict[str, Any]]:
    """
    Get information about the current production model (the one with 'champion' alias).

    Args:
        model_architecture: "panns" or "yamnet"

    Returns:
        Dict with model info or None if no production model exists
    """
    if not is_mlflow_enabled():
        return None

    try:
        client = MlflowClient()

        model_name = get_registered_model_name(model_architecture)
        try:
            mv = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
        except mlflow.exceptions.MlflowException:
            logger.info(
                f"No production model found for '{model_name}' (no '{CHAMPION_ALIAS}' alias)"
            )
            return None

        if mv is None:
            return None

        return {
            "name": mv.name,
            "version": mv.version,
            "alias": CHAMPION_ALIAS,
            "run_id": mv.run_id,
            "source": mv.source,
            "description": mv.description,
            "creation_timestamp": mv.creation_timestamp,
            "last_updated_timestamp": mv.last_updated_timestamp,
        }

    except Exception as e:
        logger.error(f"Failed to get production model info: {e}")
        return None


def get_production_model_run_metrics(
    model_architecture: str = "panns",
) -> Optional[Dict[str, Any]]:
    """
    Fetch metrics from the MLflow run backing the current champion model.

    Extracts per-label precision/recall/f1/support metrics when present.
    Recognizes:
    - eval_{label}_{metric}
    - test_{label}_{metric}
    - labeled_{label}_{metric}

    Args:
        model_architecture: "panns" or "yamnet"

    Returns:
        Dict with run/model metadata and extracted per_label metrics, or None.
    """
    if not is_mlflow_enabled():
        return None

    prod = get_production_model_info(model_architecture=model_architecture)
    if not prod:
        return None

    run_id = prod.get("run_id")
    if not run_id:
        logger.warning("Production model info has no run_id")
        return None

    try:
        import json

        client = MlflowClient()
        run = client.get_run(run_id)
        run_metrics = dict(run.data.metrics or {})

        per_label: Dict[str, Dict[str, Any]] = {}
        prefixes_seen = set()
        pattern = re.compile(
            r"^(eval|test|labeled)_(.+)_(precision|recall|f1|support)$"
        )

        for metric_name, value in run_metrics.items():
            match = pattern.match(metric_name)
            if not match:
                continue

            prefix, safe_label, metric_kind = match.groups()
            prefixes_seen.add(prefix)
            label_bucket = per_label.setdefault(safe_label, {})
            label_bucket[metric_kind] = value

        # If multiple prefixes exist, prefer eval_ then test_ then labeled_.
        preferred_prefix = None
        for candidate in ("eval", "test", "labeled"):
            if candidate in prefixes_seen:
                preferred_prefix = candidate
                break

        if preferred_prefix and len(prefixes_seen) > 1:
            collapsed: Dict[str, Dict[str, Any]] = {}
            for metric_name, value in run_metrics.items():
                match = pattern.match(metric_name)
                if not match:
                    continue
                prefix, safe_label, metric_kind = match.groups()
                if prefix != preferred_prefix:
                    continue
                bucket = collapsed.setdefault(safe_label, {})
                bucket[metric_kind] = value
            per_label = collapsed

        # Fallback: parse logged per-class table artifact if scalar metrics are missing.
        if not per_label:
            try:
                table_path = client.download_artifacts(
                    run_id=run_id,
                    path="tables/per_class_metrics.json",
                    dst_path=tempfile.gettempdir(),
                )
                with open(table_path, "r", encoding="utf-8") as f:
                    table_data = json.load(f)

                classes = table_data.get("class", []) if isinstance(table_data, dict) else []
                precision_vals = table_data.get("precision", []) if isinstance(table_data, dict) else []
                recall_vals = table_data.get("recall", []) if isinstance(table_data, dict) else []
                f1_vals = table_data.get("f1_score", []) if isinstance(table_data, dict) else []
                support_vals = table_data.get("support", []) if isinstance(table_data, dict) else []

                for idx, class_name in enumerate(classes):
                    label_key = str(class_name)
                    per_label[label_key] = {
                        "precision": precision_vals[idx] if idx < len(precision_vals) else None,
                        "recall": recall_vals[idx] if idx < len(recall_vals) else None,
                        "f1": f1_vals[idx] if idx < len(f1_vals) else None,
                        "support": support_vals[idx] if idx < len(support_vals) else None,
                    }
                if per_label:
                    prefixes_seen.add("table_artifact")
                    if preferred_prefix is None:
                        preferred_prefix = "table_artifact"
            except Exception:
                pass

        return {
            "success": True,
            "model_name": prod.get("name"),
            "model_version": prod.get("version"),
            "run_id": run_id,
            "metric_prefix_used": preferred_prefix if prefixes_seen else None,
            "available_metric_prefixes": sorted(prefixes_seen),
            "per_label": per_label,
            "raw_metric_count": len(run_metrics),
        }
    except Exception as e:
        logger.error(f"Failed to read production model run metrics: {e}")
        return None


def download_production_model(
    model_architecture: str = "panns", target_dir: str = "/tmp/mlflow_models"
) -> Optional[Dict[str, str]]:
    """
    Download the production model (with 'champion' alias) from MLflow registry.

    Args:
        model_architecture: "panns" or "yamnet"
        target_dir: Directory to download the model to

    Returns:
        Dict with 'model_dir' and optionally 'label_mapping_path', or None if failed
    """
    if not is_mlflow_enabled():
        return None

    try:
        import glob

        client = MlflowClient()

        model_name = get_registered_model_name(model_architecture)

        try:
            mv = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
        except mlflow.exceptions.MlflowException:
            logger.info(f"No production model found for '{model_name}'")
            return None

        if mv is None:
            return None

        os.makedirs(target_dir, exist_ok=True)
        result = {}
        result["version"] = mv.version
        result["run_id"] = mv.run_id

        # Try different model artifact paths (different training runs use different structures)
        model_paths_to_try = ["keras_model", "model"]
        keras_download_path = None

        for model_path in model_paths_to_try:
            try:
                keras_download_path = client.download_artifacts(
                    run_id=mv.run_id, path=model_path, dst_path=target_dir
                )
                logger.info(
                    f"Downloaded model from '{model_path}' to: {keras_download_path}"
                )
                result["model_dir"] = keras_download_path
                break
            except Exception as e:
                logger.debug(f"Model path '{model_path}' not found: {e}")
                continue

        if not keras_download_path:
            logger.error(
                f"Could not find model artifacts in any of: {model_paths_to_try}"
            )
            return None

        # Try different label mapping artifact paths
        label_mapping_paths_to_try = ["model_artifacts", "label_mapping"]
        label_mapping_found = False

        for label_path in label_mapping_paths_to_try:
            try:
                label_mapping_dir = client.download_artifacts(
                    run_id=mv.run_id, path=label_path, dst_path=target_dir
                )
                logger.info(
                    f"Downloaded label mapping from '{label_path}' to: {label_mapping_dir}"
                )

                # Look for label mapping JSON files
                label_files = glob.glob(
                    os.path.join(label_mapping_dir, "label_mapping.json")
                )
                if not label_files:
                    label_files = glob.glob(
                        os.path.join(label_mapping_dir, "*_label_mapping.json")
                    )
                if not label_files:
                    # Also try any .json file in the directory
                    label_files = glob.glob(os.path.join(label_mapping_dir, "*.json"))

                if label_files:
                    result["label_mapping_path"] = label_files[0]
                    logger.info(f"Found label mapping: {label_files[0]}")
                    label_mapping_found = True
                    break
                else:
                    logger.debug(f"No JSON files found in {label_mapping_dir}")
            except Exception as e:
                logger.debug(f"Label mapping path '{label_path}' not found: {e}")
                continue

        if not label_mapping_found:
            logger.warning(
                f"Could not find label mapping in any of: {label_mapping_paths_to_try}"
            )

        logger.info(
            f"Downloaded production model v{mv.version} to {keras_download_path}"
        )
        return result

    except Exception as e:
        logger.error(f"Failed to download production model: {e}")
        return None


def get_champion_tflite_bundle(
    model_architecture: str = "panns",
) -> Optional[str]:
    """
    Get a TFLite model bundle (ZIP) for the champion model.
    Converts to TFLite on-the-fly if not already present in the run.

    Args:
        model_architecture: "panns" or "yamnet"

    Returns:
        Path to the ZIP bundle or None if failed
    """
    if not is_mlflow_enabled():
        return None

    try:
        import glob
        from ml_audio_core.model_utils import export_to_tflite
        import tensorflow as tf

        client = MlflowClient()

        model_name = get_registered_model_name(model_architecture)

        try:
            mv = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
        except mlflow.exceptions.MlflowException:
            logger.info(f"No production model found for '{model_name}'")
            return None

        if mv is None:
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Download/Resolve TFLite model
            tflite_model_path = None

            # Check if .tflite already exists in artifacts
            artifacts = client.list_artifacts(mv.run_id)
            for art in artifacts:
                if art.path.endswith(".tflite"):
                    tflite_model_path = client.download_artifacts(
                        run_id=mv.run_id, path=art.path, dst_path=tmpdir
                    )
                    break

            if not tflite_model_path:
                # Need to convert from Keras
                logger.info(
                    "TFLite model not found in artifacts, converting from Keras..."
                )
                model_data = download_production_model(
                    model_architecture=model_architecture, target_dir=tmpdir
                )
                if not model_data or "model_dir" not in model_data:
                    logger.error("Failed to download Keras model for conversion")
                    return None

                model_dir = model_data["model_dir"]
                keras_files = glob.glob(
                    os.path.join(model_dir, "**", "*.keras"), recursive=True
                )
                if not keras_files:
                    logger.error(f"No .keras file found in {model_dir}")
                    return None

                model_path = keras_files[0]
                keras_model = tf.keras.models.load_model(model_path, compile=False)
                tflite_model_path = os.path.join(tmpdir, "model.tflite")
                export_to_tflite(keras_model, tflite_model_path)
                logger.info(f"Converted and saved TFLite model to {tflite_model_path}")

            # 2. Resolve Label Mapping
            label_mapping_path = None
            if "model_data" in locals() and model_data.get("label_mapping_path"):  # type: ignore
                label_mapping_path = model_data["label_mapping_path"]  # type: ignore
            else:
                # Try to download it separately if we didn't download the whole model data yet
                label_mapping_paths_to_try = ["model_artifacts", "label_mapping"]
                for label_path in label_mapping_paths_to_try:
                    try:
                        label_mapping_dir = client.download_artifacts(
                            run_id=mv.run_id, path=label_path, dst_path=tmpdir
                        )
                        label_files = glob.glob(
                            os.path.join(label_mapping_dir, "*.json")
                        )
                        if label_files:
                            label_mapping_path = label_files[0]
                            break
                    except Exception:
                        continue

            # 3. Create ZIP bundle
            zip_path = os.path.join(
                tempfile.gettempdir(), f"champion_v{mv.version}_tflite.zip"
            )
            with zipfile.ZipFile(zip_path, "w") as zipf:
                if tflite_model_path:
                    zipf.write(tflite_model_path, "model.tflite")
                if label_mapping_path and os.path.exists(label_mapping_path):
                    zipf.write(label_mapping_path, "label_mapping.json")

            logger.info(f"Created TFLite bundle at {zip_path}")
            return zip_path

    except Exception as e:
        logger.error(f"Failed to create TFLite bundle: {e}")
        return None


def get_champion_onnx_bundle(
    model_architecture: str = "panns",
) -> Optional[str]:
    """
    Get an ONNX model bundle (ZIP) for the champion model.
    Converts to ONNX on-the-fly if not already present in the run.

    Args:
        model_architecture: "panns" or "yamnet"

    Returns:
        Path to the ZIP bundle or None if failed
    """
    if not is_mlflow_enabled():
        return None

    try:
        import glob
        from ml_backend.smartstable_core.onnx_converter import export_champion_to_onnx
        import tensorflow as tf

        client = MlflowClient()
        model_name = get_registered_model_name(model_architecture)

        try:
            mv = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
        except mlflow.exceptions.MlflowException:
            logger.info(f"No production model found for '{model_name}'")
            return None

        if mv is None:
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Download/Resolve ONNX models
            # We look for cached ONNX files or convert them
            # For now, we will perform conversion in the temp dir to ensure freshness relative to the champion model

            logger.info("Resolving champion model for ONNX conversion...")
            model_data = download_production_model(
                model_architecture=model_architecture, target_dir=tmpdir
            )

            if not model_data or "model_dir" not in model_data:
                logger.error("Failed to download Keras model for conversion")
                return None

            model_dir = model_data["model_dir"]
            keras_files = glob.glob(
                os.path.join(model_dir, "**", "*.keras"), recursive=True
            )
            if not keras_files:
                logger.error(f"No .keras file found in {model_dir}")
                return None

            model_path = keras_files[0]

            # Perform conversion
            logger.info("Converting champion model to ONNX...")
            onnx_files = export_champion_to_onnx(model_path, tmpdir, model_architecture)

            if not onnx_files:
                logger.error("ONNX conversion failed")
                return None

            embedder_path = onnx_files.get("embedder")
            classifier_path = onnx_files.get("classifier")

            # 2. Resolve Label Mapping
            label_mapping_path = None
            if "model_data" in locals() and model_data.get("label_mapping_path"):  # type: ignore
                label_mapping_path = model_data["label_mapping_path"]  # type: ignore

            # 3. Create ZIP bundle
            zip_path = os.path.join(
                tempfile.gettempdir(), f"champion_v{mv.version}_onnx.zip"
            )
            # Save model files
            with zipfile.ZipFile(zip_path, "w") as zipf:
                if embedder_path and os.path.exists(embedder_path):
                    zipf.write(embedder_path, f"champion_v{mv.version}_embedder.onnx")
                if classifier_path and os.path.exists(classifier_path):
                    zipf.write(
                        classifier_path, f"champion_v{mv.version}_classifier.onnx"
                    )
                if label_mapping_path and os.path.exists(label_mapping_path):
                    zipf.write(
                        label_mapping_path, f"champion_v{mv.version}_label_mapping.json"
                    )

            logger.info(f"Created ONNX bundle at {zip_path}")
            return zip_path

    except Exception as e:
        logger.error(f"Failed to create ONNX bundle: {e}")
        return None


def resolve_model_paths(model_architecture: str = "panns") -> tuple:
    """
    Resolve model paths, preferring MLflow production model if available.

    Args:
        model_architecture: "panns" or "yamnet"

    Returns:
        (model_path, label_mapping_path) or (None, None) to use local default
    """
    if not os.getenv("MLFLOW_LOAD_PRODUCTION_ON_STARTUP", "").lower() == "true":
        logger.info("MLFLOW_LOAD_PRODUCTION_ON_STARTUP not enabled, using local model")
        return None, None

    if not is_mlflow_enabled():
        return None, None

    try:
        result = download_production_model(model_architecture=model_architecture)
        if result:
            model_dir = result.get("model_dir")
            label_map = result.get("label_mapping_path")

            # Find the .keras file in the downloaded directory
            import glob

            keras_files = glob.glob(
                os.path.join(model_dir, "**", "*.keras"), recursive=True
            )
            if keras_files:
                model_path = keras_files[0]

                # Log the resolved paths for debugging
                logger.info(f"Resolved model path: {model_path}")
                if label_map:
                    if os.path.exists(label_map):
                        logger.info(f"Resolved label mapping path: {label_map}")
                    else:
                        logger.warning(
                            f"Label mapping path does not exist: {label_map}"
                        )
                        label_map = None
                else:
                    logger.warning("No label mapping path found in download result")

                return model_path, label_map

            # Try to load via TF SavedModel format
            if os.path.exists(os.path.join(model_dir, "saved_model.pb")):
                return model_dir, label_map

            logger.warning(f"No .keras file found in {model_dir}")
            return None, None

        return None, None

    except Exception as e:
        logger.error(f"Failed to resolve MLflow model paths: {e}")
        return None, None


def list_model_versions(
    model_architecture: str = "panns",
) -> List[Dict[str, Any]]:
    """
    List all versions of a registered model.

    Args:
        model_architecture: "panns" or "yamnet"

    Returns:
        List of model version info dicts
    """
    if not is_mlflow_enabled():
        return []

    try:
        client = MlflowClient()

        # Get registered model to check aliases
        model_name = get_registered_model_name(model_architecture)
        try:
            registered_model = client.get_registered_model(model_name)
            version_aliases = {}
            for alias_info in registered_model.aliases:
                version = alias_info.version
                alias = alias_info.alias
                if version not in version_aliases:
                    version_aliases[version] = []
                version_aliases[version].append(alias)
        except Exception:
            version_aliases = {}

        versions = client.search_model_versions(f"name='{model_name}'")

        result = []
        for mv in versions:
            run_info = {}
            try:
                run = client.get_run(mv.run_id)
                run_info = {
                    "accuracy": run.data.metrics.get("eval_accuracy"),
                    "f1_weighted": run.data.metrics.get("eval_f1_weighted"),
                    "num_classes": run.data.params.get("num_classes"),
                }
            except Exception:
                pass

            aliases = version_aliases.get(mv.version, [])
            is_champion = CHAMPION_ALIAS in aliases

            result.append(
                {
                    "version": mv.version,
                    "aliases": aliases,
                    "is_champion": is_champion,
                    "stage": "Production" if is_champion else "None",
                    "run_id": mv.run_id,
                    "description": mv.description,
                    "creation_timestamp": mv.creation_timestamp,
                    **run_info,
                }
            )

        result.sort(key=lambda x: int(x["version"]), reverse=True)
        return result

    except Exception as e:
        logger.error(f"Failed to list model versions: {e}")
        return []


def get_model_version_info(
    model_version: str, model_architecture: str = "panns"
) -> Optional[Dict[str, Any]]:
    """
    Resolve a registered model version to its run information.

    Accepts both plain numeric versions ("29") and prefixed forms ("v29").

    Args:
        model_version: Version identifier, e.g. "29" or "v29"
        model_architecture: "panns" or "yamnet"

    Returns:
        Dict with model version metadata, or None if not found.
    """
    if not is_mlflow_enabled():
        return None

    raw = str(model_version or "").strip()
    if not raw:
        logger.warning("model_version was empty")
        return None

    normalized = raw[1:] if raw.lower().startswith("v") else raw
    if not normalized.isdigit():
        logger.warning(f"Invalid model version format: {raw}")
        return None

    try:
        client = MlflowClient()
        model_name = get_registered_model_name(model_architecture)
        mv = client.get_model_version(name=model_name, version=normalized)

        return {
            "name": mv.name,
            "version": mv.version,
            "run_id": mv.run_id,
            "source": mv.source,
            "description": mv.description,
            "creation_timestamp": mv.creation_timestamp,
            "last_updated_timestamp": mv.last_updated_timestamp,
        }
    except Exception as e:
        logger.warning(
            f"Failed to resolve model version '{raw}' for architecture '{model_architecture}': {e}"
        )
        return None


def register_model(
    model_path: str,
    label_mapping_path: str,
    run_id: Optional[str] = None,
    model_architecture: str = "panns",
    description: Optional[str] = None,
) -> Optional[str]:
    """
    Register a trained model to MLflow Model Registry.

    Args:
        model_path: Path to the .keras model file
        label_mapping_path: Path to the label_mapping.json file
        run_id: MLflow run ID (uses active run if not provided)
        model_architecture: "panns" or "yamnet"
        description: Optional description for the model version

    Returns:
        Model version string if successful, None otherwise
    """
    if not is_mlflow_enabled():
        logger.warning("MLflow not enabled, cannot register model")
        return None

    try:
        import tempfile
        import shutil

        if run_id is None:
            active_run = mlflow.active_run()
            if active_run:
                run_id = active_run.info.run_id
            else:
                logger.error("No run_id provided and no active run")
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            model_filename = os.path.basename(model_path)
            model_dest = os.path.join(tmpdir, model_filename)
            shutil.copy2(model_path, model_dest)

            local_label_mapping = os.path.join(tmpdir, "label_mapping.json")
            if os.path.exists(label_mapping_path):
                shutil.copy2(label_mapping_path, local_label_mapping)

            try:
                import tensorflow as tf

                # Determine model name
                model_name = get_registered_model_name(model_architecture)

                keras_model = tf.keras.models.load_model(model_dest, compile=False)

                mlflow.tensorflow.log_model(
                    keras_model,
                    artifact_path="keras_model",
                    registered_model_name=model_name,
                )

                if os.path.exists(local_label_mapping):
                    # Standardize on model_artifacts/label_mapping.json
                    mlflow.log_artifact(local_label_mapping, "model_artifacts")

                client = MlflowClient()
                versions = client.search_model_versions(f"name='{model_name}'")
                if versions:
                    latest_version = max(versions, key=lambda v: int(v.version))
                    version = latest_version.version

                    if description:
                        client.update_model_version(
                            name=model_name, version=version, description=description
                        )

                    logger.info(f"Registered model '{model_name}' version {version}")
                    return version
                else:
                    logger.error("Model registered but could not find version")
                    return None

            except ImportError:
                logger.error("TensorFlow not available for model registration")
                return None

    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        return None


def promote_model_to_production(
    model_architecture: str = "panns", version: Optional[str] = None
) -> bool:
    """
    Promote a model version to Production by setting the 'champion' alias.

    Args:
        model_architecture: "panns" or "yamnet"
        version: Version to promote (latest if not specified)

    Returns:
        True if successful, False otherwise
    """
    if not is_mlflow_enabled():
        return False

    try:
        client = MlflowClient()

        if version is None:
            model_name = get_registered_model_name(model_architecture)
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                logger.error(f"No versions found for model '{model_name}'")
                return False
            version = max(versions, key=lambda v: int(v.version)).version

        client.set_registered_model_alias(
            name=model_name, alias=CHAMPION_ALIAS, version=version
        )

        logger.info(f"Promoted model '{model_name}' v{version} to Production")
        return True

    except Exception as e:
        logger.error(f"Failed to promote model to production: {e}")
        return False


# ─── Training Metrics Logging ─────────────────────────────────────────────────


def log_training_params(
    learning_rate: float,
    epochs: int,
    batch_size: int,
    num_classes: int,
    samples_per_class: Dict[str, int],
    **extra_params,
) -> None:
    """Log training parameters to MLflow."""
    if not is_mlflow_enabled():
        return

    try:
        mlflow.log_params(
            {
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "num_classes": num_classes,
                "total_samples": sum(samples_per_class.values()),
                **extra_params,
            }
        )

        for class_name, count in samples_per_class.items():
            mlflow.log_metric(f"samples_{class_name}", count)

    except Exception as e:
        logger.warning(f"Failed to log training params to MLflow: {e}")


def log_training_metrics(history: Any) -> None:
    """Log training metrics from Keras history to MLflow."""
    if not is_mlflow_enabled():
        return

    try:
        if history is not None and hasattr(history, "history"):
            for metric_name, values in history.history.items():
                for epoch, value in enumerate(values):
                    mlflow.log_metric(metric_name, float(value), step=epoch)

            for metric_name, values in history.history.items():
                if values:
                    mlflow.log_metric(f"final_{metric_name}", float(values[-1]))

    except Exception as e:
        logger.warning(f"Failed to log training metrics to MLflow: {e}")


def log_model_artifact(model_path: str, artifact_name: str = "model") -> None:
    """Log a trained model as an MLflow artifact."""
    if not is_mlflow_enabled():
        return

    try:
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, artifact_name)
            logger.info(f"Logged model artifact: {model_path}")
        else:
            logger.warning(f"Model path does not exist: {model_path}")
    except Exception as e:
        logger.warning(f"Failed to log model artifact to MLflow: {e}")


def log_label_mapping(label_mapping: Dict[str, Any]) -> None:
    """Log label mapping as MLflow artifact."""
    if not is_mlflow_enabled():
        return

    try:
        import json
        import tempfile
        import shutil

        # Use a consistent filename for the artifact
        label_mapping_dir = tempfile.mkdtemp()
        temp_path = os.path.join(label_mapping_dir, "label_mapping.json")
        with open(temp_path, "w") as f:
            json.dump(label_mapping, f, indent=2)

        # Standardize on model_artifacts/label_mapping.json
        mlflow.log_artifact(temp_path, "model_artifacts")

        # Cleanup
        shutil.rmtree(label_mapping_dir)

    except Exception as e:
        logger.warning(f"Failed to log label mapping to MLflow: {e}")


def log_test_pool_manifest(test_pool: Dict[str, List[str]]) -> None:
    """Log test pool manifest (SampleIDs holdout) as MLflow artifact."""
    if not is_mlflow_enabled():
        return

    try:
        import json
        import tempfile
        import shutil

        # Use a consistent filename for the artifact
        manifest_dir = tempfile.mkdtemp()
        temp_path = os.path.join(manifest_dir, "test_pool_manifest.json")
        with open(temp_path, "w") as f:
            json.dump(test_pool, f, indent=2)

        # Standardize on model_artifacts/test_pool_manifest.json
        mlflow.log_artifact(temp_path, "model_artifacts")

        # Cleanup
        shutil.rmtree(manifest_dir)
        logger.info("Test pool manifest logged successfully")

    except Exception as e:
        logger.warning(f"Failed to log test pool manifest to MLflow: {e}")


def download_test_pool_manifest(run_id: str) -> Optional[Dict[str, List[str]]]:
    """Download test pool manifest (SampleIDs holdout) from MLflow."""
    if not is_mlflow_enabled():
        return None

    try:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Download artifact
            artifact_path = "model_artifacts/test_pool_manifest.json"
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=artifact_path, dst_path=temp_dir
            )

            with open(local_path, "r") as f:
                manifest = json.load(f)

            logger.info(f"Test pool manifest downloaded for run {run_id}")
            return manifest

    except Exception as e:
        logger.warning(f"Failed to download test pool manifest from MLflow: {e}")
        return None


# ─── Evaluation Visualization Helpers ─────────────────────────────────────────


def _log_evaluation_tables(
    eval_results: Dict[str, Any],
    training_samples: Optional[Dict[str, int]] = None,
    test_samples: Optional[Dict[str, int]] = None,
) -> None:
    """
    Log evaluation data as MLflow tables for future visualization features.
    """
    if not is_mlflow_enabled():
        return

    try:
        # 1. Per-class metrics table
        if "per_class" in eval_results:
            per_class = eval_results["per_class"]
            classes = sorted(per_class.keys())

            metrics_table = {
                "class": classes,
                "precision": [per_class[c].get("precision", 0) for c in classes],
                "recall": [per_class[c].get("recall", 0) for c in classes],
                "f1_score": [per_class[c].get("f1", 0) for c in classes],
                "support": [per_class[c].get("support", 0) for c in classes],
            }

            if training_samples:
                metrics_table["training_samples"] = [
                    training_samples.get(c, 0) for c in classes
                ]
            if test_samples:
                metrics_table["test_samples"] = [
                    test_samples.get(c, 0) for c in classes
                ]

            mlflow.log_table(
                data=metrics_table, artifact_file="tables/per_class_metrics.json"
            )

        # 2. Overall metrics table
        if "overall" in eval_results:
            overall = eval_results["overall"]
            overall_table = {"metric": [], "value": []}
            for key, value in overall.items():
                if isinstance(value, (int, float)):
                    overall_table["metric"].append(key)
                    overall_table["value"].append(float(value))

            if overall_table["metric"]:
                mlflow.log_table(
                    data=overall_table, artifact_file="tables/overall_metrics.json"
                )

        # 3. Sample distribution table
        if training_samples or test_samples:
            all_classes = set()
            if training_samples:
                all_classes.update(training_samples.keys())
            if test_samples:
                all_classes.update(test_samples.keys())

            classes = sorted(all_classes)
            sample_table = {
                "class": classes,
                "training_samples": [
                    training_samples.get(c, 0) if training_samples else 0
                    for c in classes
                ],
                "test_samples": [
                    test_samples.get(c, 0) if test_samples else 0 for c in classes
                ],
                "total_samples": [
                    (training_samples.get(c, 0) if training_samples else 0)
                    + (test_samples.get(c, 0) if test_samples else 0)
                    for c in classes
                ],
            }
            mlflow.log_table(
                data=sample_table, artifact_file="tables/sample_distribution.json"
            )

        logger.info("Evaluation tables logged successfully")

    except Exception as e:
        logger.warning(f"Failed to log evaluation tables to MLflow: {e}")


def _log_evaluation_plots(
    eval_results: Dict[str, Any],
    training_samples: Optional[Dict[str, int]] = None,
    test_samples: Optional[Dict[str, int]] = None,
    samples_before_augmentation: Optional[Dict[str, int]] = None,
) -> None:
    """Generate and log visualization plots to MLflow."""
    if not is_mlflow_enabled():
        return

    active_run = mlflow.active_run()
    if not active_run:
        logger.warning("No active MLflow run, cannot log plots")
        return

    logger.info(f"Generating evaluation plots for run {active_run.info.run_id}")

    # Get display labels (shows merged labels in parentheses)
    try:
        from smartstablemodel.labels import get_display_labels

        display_labels = get_display_labels()
    except Exception as e:
        logger.debug(f"Could not load display labels: {e}")
        display_labels = {}

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # 1. Sample distribution plots
        if training_samples or test_samples:
            logger.info("Generating sample distribution plots...")
            _log_sample_distribution_plots(
                training_samples,
                test_samples,
                samples_before_augmentation,
                display_labels,
            )

        # 2. Per-class metrics plots
        if "per_class" in eval_results:
            logger.info("Generating metrics bar chart...")
            _log_metrics_bar_chart(eval_results["per_class"], display_labels)
            logger.info("Generating precision-recall scatter...")
            _log_precision_recall_scatter(eval_results["per_class"], display_labels)

        # 3. Confusion matrix heatmap
        if "confusion_matrix" in eval_results:
            logger.info("Generating confusion matrix plot...")
            _log_confusion_matrix_plot(eval_results["confusion_matrix"], display_labels)

        logger.info("All evaluation plots logged successfully")

    except ImportError as e:
        logger.warning(f"Matplotlib not available for plots: {e}")
    except Exception as e:
        logger.warning(f"Failed to generate evaluation plots: {e}")


def _log_sample_distribution_plots(
    training_samples: Optional[Dict[str, int]] = None,
    test_samples: Optional[Dict[str, int]] = None,
    samples_before_augmentation: Optional[Dict[str, int]] = None,
    display_labels: Optional[Dict[str, str]] = None,
) -> None:
    """Log sample distribution plots (training, test, total)."""
    import matplotlib.pyplot as plt
    import numpy as np

    if display_labels is None:
        display_labels = {}

    all_classes = set()
    if training_samples:
        all_classes.update(training_samples.keys())
    if test_samples:
        all_classes.update(test_samples.keys())

    if not all_classes:
        return

    classes = sorted(all_classes)
    display_names: list[str] = [display_labels.get(c) or str(c) for c in classes]

    train_counts = [
        training_samples.get(c, 0) if training_samples else 0 for c in classes
    ]
    test_counts = [test_samples.get(c, 0) if test_samples else 0 for c in classes]
    pre_aug_counts = [
        samples_before_augmentation.get(c, 0) if samples_before_augmentation else 0
        for c in classes
    ]
    total_counts = [t + s for t, s in zip(train_counts, test_counts)]

    # Sort by total count descending
    sorted_data = sorted(
        zip(display_names, train_counts, test_counts, total_counts, pre_aug_counts),
        key=lambda x: x[3],
        reverse=True,
    )
    display_names = [d[0] for d in sorted_data]
    train_counts = [d[1] for d in sorted_data]
    test_counts = [d[2] for d in sorted_data]
    total_counts = [d[3] for d in sorted_data]
    pre_aug_counts = [d[4] for d in sorted_data]

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(max(12, len(display_names) * 0.6), 7))
    x = np.arange(len(display_names))

    bars_train = ax.bar(
        x, train_counts, label="Training Set (after augmentation)", color="#4C78A8"
    )
    bars_test = ax.bar(
        x, test_counts, bottom=train_counts, label="Test Set", color="#F58518"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha="right")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Samples")

    if samples_before_augmentation:
        ax.set_title(
            "Sample Distribution per Class (Training + Test)\nFormat: total (original before augmentation)"
        )
    else:
        ax.set_title("Sample Distribution per Class (Training + Test)")
    ax.legend()

    for i, (bar_train, bar_test, total, pre_aug) in enumerate(
        zip(bars_train, bars_test, total_counts, pre_aug_counts)
    ):
        if samples_before_augmentation and pre_aug > 0 and pre_aug != train_counts[i]:
            label_text = f"{total} ({pre_aug})"
        else:
            label_text = str(total)
        ax.text(
            i,
            total + 0.5,
            label_text,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    mlflow.log_figure(fig, "plots/sample_distribution_stacked.png")
    plt.close(fig)


def _log_metrics_bar_chart(
    per_class: Dict[str, Dict], display_labels: Optional[Dict[str, str]] = None
) -> None:
    """Log precision/recall/F1 per class grouped bar chart."""
    import matplotlib.pyplot as plt
    import numpy as np

    if display_labels is None:
        display_labels = {}

    classes = list(per_class.keys())

    # Sort by F1 descending
    f1_scores = [per_class[c].get("f1", 0) or 0 for c in classes]
    sorted_pairs = sorted(zip(classes, f1_scores), key=lambda x: x[1], reverse=True)
    classes = [p[0] for p in sorted_pairs]

    display_names: list[str] = [display_labels.get(c) or str(c) for c in classes]

    precision = [per_class[c].get("precision", 0) or 0 for c in classes]
    recall = [per_class[c].get("recall", 0) or 0 for c in classes]
    f1 = [per_class[c].get("f1", 0) or 0 for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(12, len(classes) * 0.7), 6))
    ax.bar(x - width, precision, width, label="Precision", color="#4C78A8")
    ax.bar(x, recall, width, label="Recall", color="#F58518")
    ax.bar(x + width, f1, width, label="F1", color="#54A24B")

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 per Class")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    mlflow.log_figure(fig, "plots/metrics_per_class.png")
    plt.close(fig)


def _log_precision_recall_scatter(
    per_class: Dict[str, Dict], display_labels: Optional[Dict[str, str]] = None
) -> None:
    """Log precision vs recall scatter plot with F1 coloring and support sizing."""
    import matplotlib.pyplot as plt
    import numpy as np

    if display_labels is None:
        display_labels = {}

    classes = list(per_class.keys())
    display_names: list[str] = [display_labels.get(c) or str(c) for c in classes]

    precision = np.array([per_class[c].get("precision", 0) or 0 for c in classes])
    recall = np.array([per_class[c].get("recall", 0) or 0 for c in classes])
    f1 = np.array([per_class[c].get("f1", 0) or 0 for c in classes])
    support = np.array([per_class[c].get("support", 1) or 1 for c in classes])

    s_max = max(support.max(), 1)
    sizes = 50 + 300 * (support / s_max)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        recall,
        precision,
        s=sizes,
        c=f1,
        cmap="viridis",
        edgecolor="black",
        alpha=0.7,
        vmin=0,
        vmax=1,
    )

    for i, cls_name in enumerate(display_names):
        ax.annotate(
            cls_name,
            (recall[i], precision[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall (size=support, color=F1)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("F1 Score")

    plt.tight_layout()
    mlflow.log_figure(fig, "plots/precision_recall_scatter.png")
    plt.close(fig)


def _log_confusion_matrix_plot(
    cm_data: Dict[str, Any], display_labels: Optional[Dict[str, str]] = None
) -> None:
    """Log confusion matrix heatmaps."""
    import matplotlib.pyplot as plt
    import numpy as np

    if display_labels is None:
        display_labels = {}

    labels = cm_data.get("labels", [])
    matrix = np.array(cm_data.get("matrix", []))

    if len(labels) == 0 or matrix.size == 0:
        return

    display_names: list[str] = [display_labels.get(lbl) or str(lbl) for lbl in labels]

    row_sums = matrix.sum(axis=1, keepdims=True)
    col_sums = matrix.sum(axis=0, keepdims=True)
    row_sums[row_sums == 0] = 1
    col_sums[col_sums == 0] = 1

    matrix_recall = matrix.astype(float) / row_sums
    matrix_precision = matrix.astype(float) / col_sums

    precision_per_class = np.diag(matrix_precision)

    # Plot: Recall-focused confusion matrix
    fig, ax = plt.subplots(
        figsize=(max(12, len(labels) * 0.8), max(10, len(labels) * 0.7))
    )

    im = ax.imshow(matrix_recall, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(display_names, rotation=45, ha="right")
    ax.set_yticklabels(display_names)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix_recall[i, j]
            count = matrix[i, j]
            text_color = "white" if val > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}\n({count})",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        "Confusion Matrix (row-normalized = Recall)\nDiagonal shows recall per class"
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Recall (row proportion)")

    plt.tight_layout()
    mlflow.log_figure(fig, "plots/confusion_matrix.png")
    plt.close(fig)

    # Plot: Combined matrix with precision annotations
    fig, ax = plt.subplots(
        figsize=(max(14, len(labels) * 0.9), max(12, len(labels) * 0.8))
    )

    im = ax.imshow(matrix_recall, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(display_names, rotation=45, ha="right")
    ax.set_yticklabels(display_names)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix_recall[i, j]
            count = matrix[i, j]
            text_color = "white" if val > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}\n({count})",
                ha="center",
                va="center",
                color=text_color,
                fontsize=6,
            )

    for j, prec in enumerate(precision_per_class):
        ax.text(
            j,
            len(labels) + 0.1,
            f"P:{prec:.2f}",
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            color="#E45756",
        )

    ax.set_xlabel(
        "Predicted\n(P = Precision: of all predictions for this class, how many were correct)"
    )
    ax.set_ylabel("True\n(R = Recall: of all actual samples, how many were found)")
    ax.set_title("Confusion Matrix with Precision (P) and Recall (R) per Class")

    ax.set_xlim(-0.5, len(labels) + 0.8)
    ax.set_ylim(len(labels) + 0.8, -0.5)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Recall (row proportion)")

    plt.tight_layout()
    mlflow.log_figure(fig, "plots/confusion_matrix_with_precision.png")
    plt.close(fig)


def log_evaluation_results(
    eval_results: Dict[str, Any],
    training_samples: Optional[Dict[str, int]] = None,
    test_samples: Optional[Dict[str, int]] = None,
    samples_before_augmentation: Optional[Dict[str, int]] = None,
) -> None:
    """Log evaluation results to MLflow."""
    if not is_mlflow_enabled():
        return

    try:
        if "overall" in eval_results:
            overall = eval_results["overall"]
            for key, value in overall.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"eval_{key}", float(value))

        if "per_class" in eval_results:
            for class_name, metrics in eval_results["per_class"].items():
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)) and value is not None:
                        mlflow.log_metric(
                            f"eval_{class_name}_{metric_name}", float(value)
                        )

        if "confusion_matrix" in eval_results:
            import json
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(eval_results["confusion_matrix"], f, indent=2)
                temp_path = f.name

            mlflow.log_artifact(temp_path, "confusion_matrix")
            os.unlink(temp_path)

        # Generate and log visualization plots
        _log_evaluation_plots(
            eval_results, training_samples, test_samples, samples_before_augmentation
        )

        # Log evaluation tables for future MLflow visualization features
        _log_evaluation_tables(eval_results, training_samples, test_samples)

    except Exception as e:
        logger.warning(f"Failed to log evaluation results to MLflow: {e}")


class CentroidTrainingTracker:
    """High-level wrapper for tracking centroid model training with MLflow."""

    def __init__(self, experiment_name: str = "soundscape_centroid"):
        self.experiment_name = experiment_name
        self.enabled = init_mlflow(experiment_name) if is_mlflow_enabled() else False
        self._active_run = None

    def start_training_run(
        self,
        task_count: int,
        max_segments: int,
        max_normal_segments: int,
        include_human_speech: bool,
        human_speech_samples: int,
        tags: Optional[Dict[str, str]] = None,
        run_name: Optional[str] = None,
    ):
        """Start a new training run and log initial parameters."""
        if not self.enabled:
            return

        from datetime import datetime

        if run_name is None:
            run_name = f"centroid_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            self._active_run = mlflow.start_run(run_name=run_name)

            if tags:
                mlflow.set_tags(tags)

            mlflow.log_params(
                {
                    "task_count": task_count,
                    "max_segments": max_segments,
                    "max_normal_segments": max_normal_segments,
                    "include_human_speech": include_human_speech,
                    "human_speech_samples": human_speech_samples,
                }
            )

            logger.info(f"Started MLflow run: {run_name}")

        except Exception as e:
            logger.error(f"Failed to start centroid training run: {e}")
            self._active_run = None

    def log_centroid_model(
        self, centroid_path: str, model_architecture: str = "panns"
    ) -> Optional[str]:
        """Log the numpy centroid model to MLflow and register it in the registry.
        Returns the model version string.
        """
        if not self.enabled or not self._active_run:
            logger.warning(
                "MLflow not enabled or no active run, cannot log centroid model."
            )
            return None

        try:
            # We need to log it as an MLflow model to use the model registry
            class CentroidPyFunc(mlflow.pyfunc.PythonModel):
                def load_context(self, context):
                    import numpy as np

                    self.centroid = np.load(context.artifacts["centroid_npy"])

                def predict(self, context, model_input) -> np.ndarray:
                    return self.centroid

            registered_model_name = f"soundscape_centroid_{model_architecture}"

            mlflow.pyfunc.log_model(
                name="centroid_model",
                python_model=CentroidPyFunc(),
                artifacts={"centroid_npy": centroid_path},
                registered_model_name=registered_model_name,
            )

            # Fetch the version
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            versions = client.search_model_versions(f"name='{registered_model_name}'")
            if versions:
                latest_version = max(versions, key=lambda v: int(v.version))
                logger.info(
                    f"Registered centroid model '{registered_model_name}' version {latest_version.version}"
                )
                return latest_version.version
            return None

        except Exception as e:
            logger.error(f"Failed to log centroid model: {e}", exc_info=True)
            return None

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log training metrics."""
        if self.enabled and self._active_run:
            mlflow.log_metrics(metrics)

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run."""
        if self.enabled and self._active_run:
            try:
                mlflow.end_run(status=status)
                logger.info(f"Ended MLflow run with status: {status}")
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")
            finally:
                self._active_run = None


def promote_centroid_to_production(
    model_architecture: str = "panns", version: Optional[str] = None
) -> bool:
    """Promote a centroid model version to production by setting the 'champion' alias."""
    if not is_mlflow_enabled():
        return False

    try:
        client = MlflowClient()
        model_name = f"soundscape_centroid_{model_architecture}"

        if version is None:
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                logger.error(f"No versions found for model '{model_name}'")
                return False
            version = max(versions, key=lambda v: int(v.version)).version

        client.set_registered_model_alias(
            name=model_name, alias=CHAMPION_ALIAS, version=version
        )
        logger.info(f"Promoted centroid model '{model_name}' v{version} to champion")
        return True

    except Exception as e:
        logger.error(f"Failed to promote centroid model: {e}")
        return False


def download_champion_centroid(
    model_architecture: str = "panns",
) -> Optional[str]:
    """Download the champion centroid (.npy) from MLflow registry.

    Returns:
        Path to the downloaded .npy file, or None if unavailable.
    """
    if not is_mlflow_enabled():
        return None

    try:
        client = MlflowClient()
        model_name = f"soundscape_centroid_{model_architecture}"

        try:
            mv = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
        except mlflow.exceptions.MlflowException:
            logger.info(f"No champion centroid found for '{model_name}'")
            return None

        if mv is None:
            return None

        # Download the full model directory via the model URI
        import tempfile
        import glob

        target_dir = os.path.join(tempfile.gettempdir(), "mlflow_centroid")
        os.makedirs(target_dir, exist_ok=True)

        model_uri = f"models:/{model_name}@{CHAMPION_ALIAS}"
        local_dir = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri, dst_path=target_dir
        )

        # Search for .npy files in the downloaded directory
        npy_files = glob.glob(os.path.join(local_dir, "**", "*.npy"), recursive=True)

        if not npy_files:
            logger.warning(f"No .npy file found in downloaded model at {local_dir}")
            return None

        npy_path = npy_files[0]
        logger.info(f"Downloaded champion centroid v{mv.version} from: {npy_path}")
        return npy_path

    except Exception as e:
        logger.error(f"Failed to download champion centroid: {e}")
        return None


# ─── Soundscape Training Tracker Class ────────────────────────────────────────


class SoundscapeTrainingTracker:
    """High-level wrapper for tracking soundscape classifier training with MLflow."""

    def __init__(self, experiment_name: str = "soundscape_classifier"):
        self.experiment_name = experiment_name
        self.enabled = init_mlflow(experiment_name) if is_mlflow_enabled() else False
        self._active_run = None

    def start_training_run(
        self,
        training_segments_by_label: Dict[str, list],
        learning_rate: float,
        epochs: int,
        batch_size: int,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        dropout_rates: Optional[list] = None,
        min_samples_per_class: int = 5,
        max_samples_per_class: Optional[int] = None,
        augmentation_factor: int = 2,
        label_remapping: Optional[Dict[str, str]] = None,
        exclude_labels: Optional[list] = None,
        training_metric: str = "accuracy",
        label_weights: Optional[Dict[str, float]] = None,
        audio_backend: Optional[str] = "rust",
    ):
        """Start a new training run and log initial parameters."""
        if not self.enabled:
            return

        from datetime import datetime

        if run_name is None:
            run_name = f"soundscape_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            self._active_run = mlflow.start_run(run_name=run_name)

            if tags:
                mlflow.set_tags(tags)

            samples_per_class = {
                label: len(segs) for label, segs in training_segments_by_label.items()
            }

            extra_params = {
                "min_samples_per_class": min_samples_per_class,
                "augmentation_factor": augmentation_factor,
                "training_metric": training_metric,
                "audio_backend": audio_backend,
            }

            if max_samples_per_class:
                extra_params["max_samples_per_class"] = max_samples_per_class

            if dropout_rates:
                extra_params["dropout_layer1"] = (
                    dropout_rates[0] if len(dropout_rates) > 0 else None
                )
                extra_params["dropout_layer2"] = (
                    dropout_rates[1] if len(dropout_rates) > 1 else None
                )
                extra_params["dropout_layer3"] = (
                    dropout_rates[2] if len(dropout_rates) > 2 else None
                )

            if label_remapping:
                extra_params["label_remapping"] = str(label_remapping)

            if exclude_labels:
                extra_params["exclude_labels"] = str(exclude_labels)

            if label_weights:
                non_default_weights = {
                    k: v for k, v in label_weights.items() if v != 1.0
                }
                if non_default_weights:
                    extra_params["label_weights"] = str(non_default_weights)

            log_training_params(
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                num_classes=len(training_segments_by_label),
                samples_per_class=samples_per_class,
                **extra_params,
            )

            logger.info(f"Started MLflow run: {run_name}")

        except Exception as e:
            logger.error(f"Failed to start training run: {e}")
            self._active_run = None

    def log_history(self, history: Any) -> None:
        """Log Keras training history."""
        if self.enabled and self._active_run:
            log_training_metrics(history)

    def log_model(self, model_path: str) -> None:
        """Log saved model as artifact."""
        if self.enabled and self._active_run:
            log_model_artifact(model_path)

    def log_labels(self, label_mapping: Dict[str, Any]) -> None:
        """Log label mapping."""
        if self.enabled and self._active_run:
            log_label_mapping(label_mapping)

    def log_evaluation(
        self,
        eval_results: Dict[str, Any],
        training_samples: Optional[Dict[str, int]] = None,
        test_samples: Optional[Dict[str, int]] = None,
        samples_before_augmentation: Optional[Dict[str, int]] = None,
    ) -> None:
        """Log evaluation results with optional sample distribution info."""
        if self.enabled and self._active_run:
            log_evaluation_results(
                eval_results,
                training_samples,
                test_samples,
                samples_before_augmentation,
            )

    def log_final_class_info(self, actual_classes: list, total_samples: int) -> None:
        """Log the actual number of classes after training."""
        if not self.enabled or not self._active_run:
            return
        try:
            mlflow.log_metrics(
                {
                    "actual_num_classes": len(actual_classes),
                    "actual_total_samples": total_samples,
                }
            )
            mlflow.set_tag("actual_classes", str(actual_classes))
        except Exception as e:
            logger.warning(f"Failed to log final class info: {e}")

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run."""
        if self.enabled and self._active_run:
            try:
                mlflow.end_run(status=status)
                logger.info(f"Ended MLflow run with status: {status}")
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")
            finally:
                self._active_run = None


# ─── Evaluation Helper ────────────────────────────────────────────────────────


def evaluate_segments(
    segments_by_label: Dict[str, list],
    batch_size: int = 32,
    classifier=None,
    track_misclassifications: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate segments using a classifier.

    Args:
        segments_by_label: Dict of label -> list of audio segments
        batch_size: Batch size for inference
        classifier: Classifier to use for evaluation
        track_misclassifications: If True, include misclassification details in output

    Returns:
        Dict with dataset_summary, approx_memory_bytes_used_segments, and metrics
    """
    if classifier is None:
        raise ValueError("classifier must be provided for evaluation")

    metrics = classifier.evaluate_segments_by_label(
        segments_by_label,
        batch_size=batch_size,
        track_misclassifications=track_misclassifications,
    )

    dataset_summary = {label: len(segs) for label, segs in segments_by_label.items()}

    try:
        total_samples = 0
        for segs in segments_by_label.values():
            for seg in segs:
                # Handle both raw audio and (audio, id) tuples
                audio_data = seg[0] if isinstance(seg, (list, tuple)) else seg
                if hasattr(audio_data, "shape") and len(audio_data.shape) > 0:
                    total_samples += audio_data.shape[0]
        approx_memory_bytes = total_samples * 4  # float32
    except Exception:
        approx_memory_bytes = None

    return {
        "dataset_summary": dataset_summary,
        "approx_memory_bytes_used_segments": approx_memory_bytes,
        "metrics": metrics,
    }


def log_synthetic_evaluation_run(
    run_id: str,
    sed_metrics: Dict[str, Any],
    audio_track: np.ndarray,
    annotations: List[Dict[str, Any]],
    predicted_events: Optional[List[Dict[str, Any]]] = None,
    sr: int = 16000,
) -> None:
    """
    Log synthetic evaluation results and artifacts to an existing MLflow run.
    """
    if not is_mlflow_enabled():
        return

    try:
        import json
        import tempfile
        import shutil
        import soundfile as sf

        # Prefix metrics to distinguish them from standard evaluation
        log_metrics = {
            f"synth_{k}": float(v)
            for k, v in sed_metrics.items()
            if isinstance(v, (int, float))
        }
        # Also log confidence distribution buckets
        conf_dist = sed_metrics.get("confidence_distribution", {})
        for bucket, count in conf_dist.items():
            safe_key = bucket.replace(".", "_").replace("-", "_").replace("+", "plus")
            log_metrics[f"synth_conf_{safe_key}"] = float(count)

        # Use MlflowClient directly to avoid global state issues (experiment mismatch)
        client = MlflowClient()

        # Log metrics
        for k, v in log_metrics.items():
            try:
                client.log_metric(run_id, k, v)
            except Exception as me:
                logger.warning(f"Failed to log metric {k} to run {run_id}: {me}")

        # Save artifacts locally first
        temp_dir = tempfile.mkdtemp()
        try:
            # Save audio
            audio_path = os.path.join(temp_dir, "test_track.wav")
            sf.write(audio_path, audio_track, sr)

            # Save annotations
            annot_path = os.path.join(temp_dir, "test_track.json")
            with open(annot_path, "w") as f:
                json.dump(
                    {
                        "sr": sr,
                        "duration": float(len(audio_track) / sr),
                        "annotations": annotations,
                        "sed_metrics": sed_metrics,
                    },
                    f,
                    indent=2,
                )

            if predicted_events:
                try:
                    duration = float(len(audio_track) / sr)

                    # 1. SED Event Roll (Spectrogram + Gantt)
                    _generate_sed_event_roll(
                        audio_track,
                        sr,
                        annotations,
                        predicted_events,
                        duration,
                        os.path.join(temp_dir, "sed_event_roll.png"),
                    )

                    # 2. Class-wise Error Rates (Stacked Bar Chart)
                    _generate_error_rate_plot(
                        annotations,
                        predicted_events,
                        duration,
                        os.path.join(temp_dir, "sed_error_rates.png"),
                    )

                    logger.info("Generated SED visualization plots")
                except Exception as pe:
                    logger.warning(f"Failed to generate SED plots: {pe}", exc_info=True)

            # Upload to synthetic_eval folder
            client.log_artifacts(run_id, temp_dir, artifact_path="synthetic_eval")
            logger.info(
                f"Logged synthetic evaluation artifacts to run {run_id} (synthetic_eval/)"
            )
        finally:
            shutil.rmtree(temp_dir)

    except Exception as e:
        logger.warning(f"Failed to log synthetic evaluation to MLflow: {e}")


def _generate_sed_event_roll(
    audio, sr, ground_truth, predictions, duration, output_path
):
    """
    Generate a clear event-roll timeline comparing Reference vs Prediction.

    Each label gets two sub-rows:
      - Top sub-row: Reference (solid fill)
      - Bottom sub-row: Prediction (lighter fill + border)

    Missed events (reference only) and false alarms (prediction only) are
    immediately visible.  No mel-spectrogram panel.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        import matplotlib.ticker as mticker

        # ── Collect labels (exclude non-event labels) ─────────────────────
        NON_EVENT = {"normal", "silence"}
        labels = sorted(
            set(e["label"] for e in ground_truth if e["label"] not in NON_EVENT)
            | set(e["label"] for e in predictions if e["label"] not in NON_EVENT)
        )
        if not labels:
            return

        n_labels = len(labels)
        ROW_HEIGHT = 0.35  # height of each sub-row (ref / pred)
        LABEL_SPACING = 1.0  # vertical spacing per label group
        REF_ALPHA = 0.85
        PRED_ALPHA = 0.55

        # ── Color palette ─────────────────────────────────────────────────
        # Use a qualitative palette that is colour-blind friendly
        BASE_COLORS = [
            "#4C72B0",
            "#55A868",
            "#C44E52",
            "#8172B2",
            "#CCB974",
            "#64B5CD",
            "#E57C25",
            "#8C564B",
            "#E377C2",
            "#7F7F7F",
            "#BCBD22",
            "#17BECF",
            "#AEC7E8",
            "#FFBB78",
            "#98DF8A",
            "#FF9896",
            "#C5B0D5",
            "#C49C94",
            "#F7B6D2",
            "#DBDB8D",
        ]
        label_colors = {
            lbl: BASE_COLORS[i % len(BASE_COLORS)] for i, lbl in enumerate(labels)
        }

        # ── Figure ────────────────────────────────────────────────────────
        fig_height = max(4, n_labels * 0.8 + 1.5)
        fig, ax = plt.subplots(figsize=(18, fig_height))

        y_ticks = []
        y_tick_labels = []

        for idx, label in enumerate(labels):
            y_center = idx * LABEL_SPACING
            y_ref = y_center + ROW_HEIGHT * 0.05  # reference sits on top
            y_pred = y_center - ROW_HEIGHT - ROW_HEIGHT * 0.05  # prediction below

            y_ticks.append(y_center)
            y_tick_labels.append(label)

            colour = label_colors[label]

            # Alternating background shading for readability
            if idx % 2 == 0:
                ax.axhspan(
                    y_center - LABEL_SPACING / 2,
                    y_center + LABEL_SPACING / 2,
                    color="#f5f5f5",
                    zorder=0,
                )

            # ── Reference events (solid) ──────────────────────────────────
            gt_events = [e for e in ground_truth if e["label"] == label]
            for e in gt_events:
                w = e["end_time"] - e["start_time"]
                ax.barh(
                    y_ref,
                    w,
                    left=e["start_time"],
                    height=ROW_HEIGHT,
                    align="edge",
                    color=colour,
                    alpha=REF_ALPHA,
                    linewidth=0.5,
                    edgecolor="white",
                )

            # ── Predicted events (lighter + border) ───────────────────────
            pred_events = [e for e in predictions if e["label"] == label]
            for e in pred_events:
                w = e["end_time"] - e["start_time"]
                ax.barh(
                    y_pred,
                    w,
                    left=e["start_time"],
                    height=ROW_HEIGHT,
                    align="edge",
                    color=colour,
                    alpha=PRED_ALPHA,
                    linewidth=1.2,
                    edgecolor=colour,
                    linestyle="--",
                )

        # ── Axes formatting ───────────────────────────────────────────────
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels, fontsize=9)
        ax.set_ylim(-LABEL_SPACING, (n_labels - 1) * LABEL_SPACING + LABEL_SPACING)
        ax.invert_yaxis()

        # Time axis: format as M:SS
        def _fmt_time(x, _pos):
            m, s = divmod(int(x), 60)
            return f"{m}:{s:02d}"

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_time))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(30))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
        ax.set_xlim(0, duration)
        ax.set_xlabel("Time", fontsize=10)
        ax.grid(True, axis="x", alpha=0.25, which="major")
        ax.grid(True, axis="x", alpha=0.10, which="minor")

        # ── Title & Legend ────────────────────────────────────────────────
        ax.set_title(
            "SED Event Roll  —  Reference (solid, bottom) vs Prediction (dashed, top)",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        legend_elements = [
            Patch(facecolor="#888888", alpha=REF_ALPHA, label="Reference (bottom)"),
            Patch(
                facecolor="#888888",
                alpha=PRED_ALPHA,
                edgecolor="#888888",
                linestyle="--",
                linewidth=1.2,
                label="Prediction (top)",
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=9,
            framealpha=0.9,
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    except Exception as e:
        logger.warning(f"Error plotting SED event roll: {e}", exc_info=True)


def _compute_segment_based_classwise_metrics(
    ground_truth, predictions, time_resolution=1.0
):
    """
    Compute segment-based class-wise SED metrics (replaces sed_eval).

    Discretises events into a binary segment grid and counts TP / FP / FN
    per class, then derives deletion, insertion and match rates.

    Args:
        ground_truth: List of dicts with 'label', 'start_time', 'end_time'.
        predictions:  List of dicts with 'label', 'start_time', 'end_time'.
        time_resolution: Segment length in seconds (default 1.0).

    Returns:
        Dict  label -> {deletion_rate, insertion_rate, match_rate, Nref, Nsys}
    """
    import math

    NON_EVENT = {"normal", "silence"}
    all_labels = sorted(
        (set(e["label"] for e in ground_truth) | set(e["label"] for e in predictions))
        - NON_EVENT
    )
    if not all_labels:
        return {}

    # Determine total length in segments
    max_offset = 0.0
    for e in ground_truth + predictions:
        max_offset = max(max_offset, e.get("end_time", 0.0))
    n_segments = int(math.ceil(max_offset / time_resolution))
    if n_segments == 0:
        return {}

    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
    n_labels = len(all_labels)

    # Build binary event rolls  (n_segments x n_labels)
    ref_roll = np.zeros((n_segments, n_labels), dtype=np.int32)
    est_roll = np.zeros((n_segments, n_labels), dtype=np.int32)

    def _fill_roll(roll, events):
        for e in events:
            col = label_to_idx.get(e["label"])
            if col is None:
                continue
            seg_start = int(e["start_time"] / time_resolution)
            seg_end = int(math.ceil(e["end_time"] / time_resolution))
            seg_start = max(0, min(seg_start, n_segments))
            seg_end = max(0, min(seg_end, n_segments))
            roll[seg_start:seg_end, col] = 1

    _fill_roll(ref_roll, ground_truth)
    _fill_roll(est_roll, predictions)

    # Per-class statistics
    results = {}
    for lbl_idx, lbl in enumerate(all_labels):
        ref_col = ref_roll[:, lbl_idx]
        est_col = est_roll[:, lbl_idx]

        ntp = int(np.sum((ref_col + est_col) > 1))
        nfp = int(np.sum((est_col - ref_col) > 0))
        nfn = int(np.sum((ref_col - est_col) > 0))
        nref = int(np.sum(ref_col))
        nsys = int(np.sum(est_col))

        if nref > 0:
            del_rate = nfn / nref
            ins_rate = nfp / nref
            match_rate = ntp / nref
        else:
            del_rate = 0.0
            ins_rate = float(nfp) if nfp > 0 else 0.0
            match_rate = 0.0

        results[lbl] = {
            "deletion_rate": del_rate,
            "insertion_rate": ins_rate,
            "match_rate": match_rate,
            "Nref": nref,
            "Nsys": nsys,
        }

    return results


def _generate_error_rate_plot(ground_truth, predictions, duration, output_path):
    """
    Generate a stacked bar chart of Error Rates (Deletion, Insertion, Match) per class.
    Uses segment-based metrics (Mesaros et al. 2016) with 1-second resolution.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        # Get all unique labels (exclude non-event labels)
        NON_EVENT = {"normal", "silence"}
        all_labels = sorted(
            (
                set(e["label"] for e in ground_truth)
                | set(e["label"] for e in predictions)
            )
            - NON_EVENT
        )
        if not all_labels:
            return

        # Compute segment-based class-wise metrics (replaces sed_eval)
        class_results = _compute_segment_based_classwise_metrics(
            ground_truth, predictions, time_resolution=1.0
        )

        stats = []
        for label in all_labels:
            if label not in class_results:
                continue
            res = class_results[label]
            # Cap rates at 5.0 for visualization sanity
            del_rate = min(5.0, res["deletion_rate"])
            ins_rate = min(5.0, res["insertion_rate"])
            match_rate = max(0.0, res["match_rate"])

            stats.append(
                {
                    "Label": label,
                    "Match": match_rate,
                    "Deletion": del_rate,
                    "Insertion": ins_rate,
                }
            )

        if not stats:
            return

        df = pd.DataFrame(stats).set_index("Label")

        fig, ax = plt.subplots(figsize=(10, len(all_labels) * 0.6 + 2))

        df[["Match", "Deletion", "Insertion"]].plot(
            kind="barh",
            stacked=True,
            color=["#2ecc71", "#e74c3c", "#3498db"],
            ax=ax,
        )

        ax.set_title("Error Rate Breakdown per Class (Lower is Better)")
        ax.set_xlabel("Error Rate (Events / N_ref)")
        ax.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    except Exception as e:
        logger.warning(f"Error plotting error rates: {e}")


def _generate_sed_metrics_heatmap(
    ground_truth, predictions, duration, output_path, bin_size=60
):
    """Generate F1-score heatmap per label per time bin."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        labels = sorted(
            list(
                set(
                    [e["label"] for e in ground_truth]
                    + [e["label"] for e in predictions]
                )
            )
        )
        if not labels:
            return

        num_bins = int(duration // bin_size) + 1
        heatmap_data = []

        for label in labels:
            row = []
            for b in range(num_bins):
                t_start = b * bin_size
                t_end = min((b + 1) * bin_size, duration)

                # Filter events in this bin
                # An event belongs to bin if it overlaps
                gt_in_bin = [
                    e
                    for e in ground_truth
                    if e["label"] == label
                    and max(e["start_time"], t_start) < min(e["end_time"], t_end)
                ]

                pred_in_bin = [
                    e
                    for e in predictions
                    if e["label"] == label
                    and max(e["start_time"], t_start) < min(e["end_time"], t_end)
                ]

                has_gt = len(gt_in_bin) > 0
                has_pred = len(pred_in_bin) > 0

                if has_gt and has_pred:
                    score = 1.0  # TP
                elif has_gt and not has_pred:
                    score = 0.0  # FN
                elif not has_gt and has_pred:
                    score = 0.0  # FP
                else:
                    score = np.nan  # TN

                row.append(score)
            heatmap_data.append(row)

        df = pd.DataFrame(
            heatmap_data, index=labels, columns=[f"{i}m" for i in range(num_bins)]
        )

        plt.figure(figsize=(max(10, num_bins * 0.5), len(labels) * 0.8 + 2))
        sns.heatmap(
            df,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".1f",
            cbar_kws={"label": "Detection Success"},
        )
        plt.title(
            "Detection Consistency per Time Bin (1 = Hit, 0 = Miss/Hallucination)"
        )
        plt.xlabel("Time Bin")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    except Exception as e:
        logger.warning(f"Error plotting heatmap: {e}")
