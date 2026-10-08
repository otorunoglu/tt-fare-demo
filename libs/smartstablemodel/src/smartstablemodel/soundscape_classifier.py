#!/usr/bin/env python3
"""
Horse Stable Soundscape Monitor
Continuously monitors stable audio and detects kicks using a cascaded ML model
"""

import os
import logging
from pathlib import Path

from ml_audio_core.embedders.panns_embedder import PannsEmbedder
import numpy as np

# TensorFlow optional for inference-only environments (e.g., Raspberry Pi with tflite_runtime)
try:
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover
    tf = None  # type: ignore

from typing import Dict, List, cast
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)

from ml_audio_core.base_classifier import BaseAudioClassifier
# from ml_audio_core.augmentation import augment_audio_data
# from ml_audio_core.evaluation import evaluate_segments_by_label

# Base directory of this module (…/src/smartstablemodel)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Find project root (handle src/ layout)
if (
    os.path.basename(BASE_DIR) == "smartstablemodel"
    and os.path.basename(os.path.dirname(BASE_DIR)) == "src"
):
    PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
else:
    PROJECT_ROOT = BASE_DIR

MODELS_DIR = os.getenv("SMARTSTABLE_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))

# Backward compatible names (retain old constants used below)
SOUNDSCAPE_CLASSIFIER_MODEL_PATH_DEFAULT = os.path.join(
    MODELS_DIR, "kick_detector_panns.keras"
)

logger = logging.getLogger(__name__)


class SoundscapeClassifier(BaseAudioClassifier):
    """Kick detection classifier using shared ML audio core."""

    def __init__(
        self,
        model_path: str | None = None,
        label_mapping_path: str | None = None,
        model_version: str | None = None,
        model_architecture: str | None = None,  # None = use config default
    ):
        """
        Initialize the SoundscapeClassifier.

        Args:
            model_path: Optional explicit path to a .keras model file.
            label_mapping_path: Optional explicit path to a label_mapping.json file.
            model_version: Optional explicit version string for the model.
            model_architecture: "panns" or "yamnet". If None, loads from config.
        """
        if model_architecture is None:
            try:
                from smartstablemodel.config.metamodel_config import MetaModelConfig

                config = MetaModelConfig.from_toml()
                model_architecture = config.model_parameters.get(
                    "default_model_architecture", "panns"
                )
            except Exception as e:
                logger.warning(f"Failed to load default model architecture: {e}")
                model_architecture = "panns"

        self.model_architecture = (model_architecture or "panns").lower()

        # 2. Resolve Model Path from MLflow (if not provided)
        if not model_path:
            try:
                # Import here to avoid circular dependencies
                from ml_backend.smartstable_core.mlflow_support import (
                    resolve_model_paths,
                )

                resolved_model_path, resolved_label_map = resolve_model_paths(
                    model_architecture=self.model_architecture
                )

                if resolved_model_path:
                    logger.info(
                        f"Using MLflow production model for {self.model_architecture}: {resolved_model_path}"
                    )
                    model_path = resolved_model_path
                    # Only override label mapping if we got one and didn't have one
                    if resolved_label_map and not label_mapping_path:
                        label_mapping_path = resolved_label_map

            except ImportError:
                # This is expected in environments without ml_backend (e.g. edge devices)
                pass
            except Exception as e:
                logger.warning(f"Error checking MLflow for champion model: {e}")

        # 3. Initialize Embedder
        if self.model_architecture == "panns":
            embedder = PannsEmbedder(
                checkpoint_path=os.getenv(
                    "PANNS_CHECKPOINT_DIR", "./panns_data/Cnn14_mAP=0.438.pth"
                ),
                device=os.getenv("PANNS_DEVICE", "cpu"),
            )
            model_prefix = "stable_panns"
        elif self.model_architecture == "yamnet":
            from ml_audio_core.embedders.yamnet_embedder import YamNetEmbedder

            embedder = YamNetEmbedder(
                model_dir=os.getenv("YAMNET_MODEL_DIR", "./yamnet_data")
            )
            model_prefix = "stable_yamnet"
        else:
            raise ValueError(f"Unknown model_architecture: {self.model_architecture}")

        backend = "keras" if tf is not None else "tflite"
        if model_path and str(model_path).lower().endswith(".tflite"):
            backend = "tflite"

        super().__init__(
            model_dir=MODELS_DIR,
            embedder=embedder,
            backend=backend,
            model_prefix=model_prefix,
            model_path=model_path,  # Pass explicit model path to base class
            label_mapping_path=label_mapping_path,  # Pass explicit label mapping path to base class
            model_version=model_version,  # Pass explicit model version to base class
        )
        from .labels import get_label_registry

        self.label_registry = get_label_registry()

        # Load normal centroid if available
        self.normal_centroid = None
        self.load_normal_centroid()

    def _load_label_mapping(self, model_path, label_mapping_path: str | None = None):
        """Load label mapping for a given model (list-output safe).

        Args:
            model_path: Path to the model file (used to derive default label mapping path)
            label_mapping_path: Optional explicit path to label_mapping.json.
                              If provided, this path is used instead of deriving
                              from model_path.
        """
        from .labels import load_model_label_mapping

        # If explicit label mapping path is provided, use it directly
        if label_mapping_path:
            import json
            from pathlib import Path

            label_file = Path(label_mapping_path)
            if label_file.exists():
                with open(label_file, "r") as f:
                    mapping = json.load(f)
                self.label_to_idx = mapping.get("label_to_idx", {})
                idx_to_label_raw = mapping.get("idx_to_label", {})
                self.idx_to_label = {int(k): v for k, v in idx_to_label_raw.items()}
                self.valid_labels = mapping.get("valid_labels", [])
                logger.info(
                    f"Loaded label mapping from explicit path with {len(self.label_to_idx)} classes: {list(self.label_to_idx.keys())}"
                )
                return
            else:
                logger.warning(
                    f"Explicit label mapping path not found: {label_mapping_path}, falling back to model path"
                )

        # Determine num_classes safely
        num_classes = 2  # default
        if hasattr(self, "model"):
            try:
                out = self.model.output
                outs = out if isinstance(out, (list, tuple)) else [out]
                # take first with numeric last dim
                for t in outs:
                    sh = getattr(t, "shape", None)
                    if sh is not None:
                        last = sh[-1]
                        if isinstance(last, int) and last > 0:
                            num_classes = int(last) if last != 1 else 2
                            break
            except Exception as e:
                logger.debug(f"List output inspection failed: {e}")
        model_path_str = model_path if isinstance(model_path, str) else str(model_path)
        mapping = load_model_label_mapping(model_path_str, num_classes)
        self.label_to_idx = mapping["label_to_idx"] or {}
        self.idx_to_label = mapping["idx_to_label"] or {}
        self.valid_labels = mapping["valid_labels"] or []
        logger.info(
            f"Loaded label mapping with {len(self.label_to_idx)} classes: {list(self.label_to_idx.keys())}"
        )
        if num_classes != len(self.label_to_idx):
            logger.warning(
                f"Model output classes ({num_classes}) do not match label mapping size ({len(self.label_to_idx)})"
            )

    def get_model_version(self):
        """Get the version of the soundscape classifier model."""
        # Use base class implementation which handles explicit version and path-based version
        version = super().get_model_version()
        if version == "untrained_model":
            # Fallback to default model constant if no model is loaded yet
            return os.path.basename(SOUNDSCAPE_CLASSIFIER_MODEL_PATH_DEFAULT).replace(
                ".keras", ""
            )
        return version

    def build_model(self, embedding_dim=None, num_classes=None, dropout_rates=None):
        """Build and return a new soundscape classifier model that takes PANNs embeddings as input.

        Args:
            embedding_dim: Input embedding dimension (default: self.embedding_dim)
            num_classes: Number of output classes
            dropout_rates: List of 3 dropout rates for each layer [layer1, layer2, layer3]
                          Default: [0.5, 0.3, 0.2]
        """
        from .labels import get_trainable_labels_for_classifier

        if embedding_dim is None:
            embedding_dim = self.embedding_dim

        if num_classes is None:
            num_classes = len(get_trainable_labels_for_classifier())

        if dropout_rates is None:
            dropout_rates = [0.5, 0.3, 0.2]

        # Ensure we have 3 dropout rates
        if len(dropout_rates) != 3:
            logger.warning(
                f"Expected 3 dropout rates, got {len(dropout_rates)}. Using defaults."
            )
            dropout_rates = [0.5, 0.3, 0.2]

        if tf is None:
            raise RuntimeError(
                "TensorFlow is required to build models. Install TensorFlow or use TFLite backend for inference."
            )

        logger.info(f"Building model with dropout rates: {dropout_rates}")

        input_name = (
            f"{self.model_architecture}_embedding"
            if hasattr(self, "model_architecture")
            else "embedding"
        )
        input_layer = tf.keras.layers.Input(shape=(embedding_dim,), name=input_name)

        # Dense layers for classification
        x = tf.keras.layers.Dense(512, activation="relu")(input_layer)
        x = tf.keras.layers.Dropout(dropout_rates[0])(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout_rates[1])(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout_rates[2])(x)

        # Output layer for multi-class classification
        if num_classes == 2:
            output = tf.keras.layers.Dense(
                1, activation="sigmoid", name="class_probability"
            )(x)
        else:
            output = tf.keras.layers.Dense(
                num_classes, activation="softmax", name="class_probabilities"
            )(x)

        model = tf.keras.Model(input_layer, output, name="soundscape_classifier")

        return model

    def score_anomaly(
        self,
        audio_batch: np.ndarray | list = None,
        embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute anomaly score for valid audio batch using the loaded normal centroids.
        Calculates distance to the *nearest* known label centroid (min distance).
        Returns array of scores (0.0 to 2.0).
        """
        if self.normal_centroid is None:
            # Try loading again (just in case it was created after init)
            if self.load_normal_centroid() is None:
                if embeddings is not None:
                    n = (
                        embeddings.shape[0]
                        if isinstance(embeddings, np.ndarray)
                        else len(embeddings)
                    )
                elif isinstance(audio_batch, list):
                    n = len(audio_batch)
                else:
                    n = audio_batch.shape[0]
                return np.zeros(n, dtype=np.float32)

        # embeddings: (Batch, Dim)
        if embeddings is None:
            if audio_batch is None:
                raise ValueError("Either audio_batch or embeddings must be provided.")
            embeddings = self.get_embeddings(audio_batch)

        if len(embeddings) == 0:
            return np.array([])

        centroid_matrix = self.normal_centroid
        # Ensure matrix is at least 2D: (N_clusters, Dim)
        if centroid_matrix.ndim == 1:
            centroid_matrix = centroid_matrix.reshape(1, -1)

        from sklearn.metrics.pairwise import cosine_distances

        # dists: (Batch, N_clusters)
        dists = cosine_distances(embeddings, centroid_matrix)

        # Anomaly score is distance to the NEAREST valid cluster
        min_dists = np.min(dists, axis=1)

        return min_dists

    def infer_audio_segment(
        self,
        audio_batch=None,
        batch_size=None,
        return_probabilities=False,
        file_path=None,
        embeddings=None,
    ):
        """Inference with multi-class support."""
        file_name = file_path.split("/")[-1] if file_path else "unknown file"
        logger.info(f"Inference on {file_name} - model: {self.get_model_version()}")
        try:
            return self.predict_batch(
                audio_batch,
                return_probabilities=return_probabilities,
                batch_size=batch_size,
                embeddings=embeddings,
            )
        except Exception as e:
            logger.error(f"Inference failed on {file_name}: {type(e).__name__}: {e}")
            raise

    # ------------------------------------------------------------------
    # Anomaly Detection / Normal Centroid Management
    # ------------------------------------------------------------------
    def get_centroid_path(self, timestamp: str | None = None) -> Path:
        """Get the path to the normal centroid file. If timestamp is provided, gets that timestamped version.
        Otherwise globs for the latest available version."""
        if timestamp:
            return Path(MODELS_DIR) / f"normal_centroid_{timestamp}.npy"

        # Try to find the latest version
        import glob

        pattern = str(Path(MODELS_DIR) / "normal_centroid_*.npy")
        candidates = glob.glob(pattern)

        if candidates:
            # Sort by creation time or by timestamp string
            latest = max(candidates, key=os.path.getctime)
            return Path(latest)

        return Path(MODELS_DIR) / "normal_centroid.npy"

    def save_normal_centroid(
        self, centroid: np.ndarray, timestamp: str | None = None
    ) -> Path:
        """Save the normal centroid to disk."""
        path = self.get_centroid_path(timestamp=timestamp)
        np.save(path, centroid)
        logger.info(f"Saved normal centroid to {path}")
        # Update cache
        self.normal_centroid = centroid
        return path

    def load_normal_centroid(self) -> np.ndarray | None:
        """Load the normal centroid, preferring the MLflow champion model."""
        # 1. Try MLflow champion first
        try:
            from ml_backend.smartstable_core.mlflow_support import (
                download_champion_centroid,
            )

            champion_path = download_champion_centroid(
                model_architecture=self.model_architecture
            )
            if champion_path and os.path.exists(champion_path):
                centroid = np.load(champion_path)
                logger.info(f"Loaded champion centroid from MLflow: {champion_path}")
                self.normal_centroid = centroid
                return centroid
        except ImportError:
            pass  # Expected in environments without ml_backend
        except Exception as e:
            logger.warning(f"Error loading champion centroid from MLflow: {e}")

        # 2. Fallback to local file
        path = self.get_centroid_path()
        if path.exists():
            try:
                centroid = np.load(path)
                logger.info(f"Loaded normal centroid from {path}")
                self.normal_centroid = centroid
                return centroid
            except Exception as e:
                logger.error(f"Failed to load normal centroid from {path}: {e}")
                return None
        else:
            logger.warning(f"Normal centroid file not found at {path}")
            return None

    def train_normal_centroid(
        self,
        training_data_dict: Dict[str, List[np.ndarray]],
        use_augmentation: bool = False,
        include_human_speech: bool = True,
        human_speech_samples: int = 400,
        human_speech_language: str = "mixed",
    ) -> np.ndarray:
        """
        Compute the centroid of 'normal' sounds.

        Args:
            training_data_dict: Dictionary mapping label names to lists of audio segments.
            use_augmentation: Whether to augment data before computing centroid.
                              (Usually not needed for centroid, but optional)
            include_human_speech: Whether to include external human speech data.
            human_speech_samples: Number of speech samples to include.
            human_speech_language: Language of speech samples.

        Returns:
            The computed centroid vector (embedding_dim,)
        """
        # Add human speech data if requested
        if include_human_speech:
            from .human_speech_database import HumanSpeechDataset

            try:
                logger.info(
                    f"Loading {human_speech_samples} human speech samples ({human_speech_language}) for centroid training..."
                )
                speech_dataset = HumanSpeechDataset()
                speech_segments = speech_dataset.prepare_speech_segments(
                    language=human_speech_language, max_samples=human_speech_samples
                )

                if speech_segments:
                    # training_data_dict might be a copy or original, let's be safe
                    training_data_dict = training_data_dict.copy()
                    training_data_dict["human_talking"] = speech_segments
                    logger.info(
                        f"Added {len(speech_segments)} human speech segments to centroid training data"
                    )
                else:
                    logger.warning("No human speech segments loaded for centroid")

            except Exception as e:
                logger.error(f"Failed to load human speech data for centroid: {str(e)}")

        # 1. Filter data based on known labels for anomaly training
        valid_labels = set(self.label_registry.get_anomaly_detector_labels())
        if not valid_labels:
            logger.warning(
                "No labels allowed for anomaly training found. Using all provided data for centroid."
            )
            valid_labels = set(training_data_dict.keys())
        else:
            logger.info(
                f"Training centroid with labels allowed for anomaly training: {valid_labels}"
            )

        # 2. Collect segments
        all_segments = []
        for label, segments in training_data_dict.items():
            if label in valid_labels:
                # Handle (audio, id) tuples if present
                clean_segments = []
                for item in segments:
                    if (
                        isinstance(item, (tuple, list))
                        and len(item) == 2
                        and isinstance(item[0], np.ndarray)
                    ):
                        clean_segments.append(item[0])
                    elif isinstance(item, np.ndarray):
                        clean_segments.append(item)

                logger.info(
                    f"  Adding {len(clean_segments)} segments from label '{label}'"
                )
                all_segments.extend(clean_segments)
            else:
                logger.debug(
                    f"  Skipping label '{label}' (not in anomaly allowed list)"
                )

        if not all_segments:
            raise ValueError("No valid segments found for centroid training.")

        # 3. Get Embeddings
        # Convert to array ? `get_embeddings` handles list
        logger.info(f"Extracting embeddings for {len(all_segments)} segments...")
        embeddings = self.get_embeddings(all_segments, batch_size=32)

        if len(embeddings) == 0:
            raise ValueError("Failed to extract embeddings.")

        # 4. Compute Centroids using the new multi-centroid method
        # We need to map back which embedding belongs to which label to do this properly.

        embeddings_by_label = {}
        current_idx = 0

        # Re-iterate to slice the big embeddings array back into label groups
        for label, segments in training_data_dict.items():
            if label not in valid_labels:
                continue

            # Count valid segments in this batch
            count = 0
            for item in segments:
                if (
                    isinstance(item, (tuple, list))
                    and len(item) == 2
                    and isinstance(item[0], np.ndarray)
                ):
                    count += 1
                elif isinstance(item, np.ndarray):
                    count += 1

            if count > 0:
                embeddings_by_label[label] = embeddings[
                    current_idx : current_idx + count
                ]
                current_idx += count

        return self.train_centroid_from_embeddings(embeddings_by_label)

    def train_centroid_from_embeddings(
        self, embeddings_by_label: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute centroids from pre-computed embeddings.
        Instead of a single global centroid, keeps a centroid for EACH label group.
        This models the normal distribution as a set of clusters rather than a single sphere.
        """
        label_centroids = []

        for label, embeddings in embeddings_by_label.items():
            if len(embeddings) == 0:
                continue

            # Mean for this specific label
            label_mean = np.mean(embeddings, axis=0)
            label_centroids.append(label_mean)
            logger.info(
                f"  Computed centroid for label '{label}' from {len(embeddings)} samples."
            )

        if not label_centroids:
            raise ValueError("No valid embeddings to compute centroid.")

        # Stack into a matrix (N_labels, Dim)
        final_centroids = np.vstack(label_centroids)

        logger.info(
            f"Final normal model computed from {len(label_centroids)} label clusters."
        )

        self.save_normal_centroid(final_centroids)
        return final_centroids

    def train_soundscape_classifier(
        self,
        training_data_dict,
        learning_rate=5e-5,
        epochs=10,
        batch_size=16,
        min_samples_per_class=30,
        max_samples_per_class=None,
        max_samples_per_class_overrides=None,
        use_augmentation=True,
        augmentation_factor=2,
        dropout_rates=None,
        training_metric="accuracy",
        label_weights=None,
    ):
        # # Add human speech data if requested
        # if include_human_speech:
        #     from .human_speech_database import HumanSpeechDataset

        #     try:
        #         logger.info(
        #             f"Loading {human_speech_samples} human speech samples ({human_speech_language})..."
        #         )
        #         speech_dataset = HumanSpeechDataset()
        #         speech_segments = speech_dataset.prepare_speech_segments(
        #             language=human_speech_language, max_samples=human_speech_samples
        #         )

        #         if speech_segments:
        #             # Add to training data
        #             training_data_dict["human_talking"] = speech_segments
        #             logger.info(
        #                 f"Added {len(speech_segments)} human speech segments to training data"
        #             )
        #         else:
        #             logger.warning("No human speech segments loaded")

        #     except Exception as e:
        #         logger.error(f"Failed to load human speech data: {str(e)}")
        #         logger.info("Continuing training without human speech data...")

        return self.train_from_segments(
            training_data_dict,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            min_samples_per_class=min_samples_per_class,
            max_samples_per_class=max_samples_per_class,
            max_samples_per_class_overrides=max_samples_per_class_overrides,
            use_augmentation=use_augmentation,
            augmentation_factor=augmentation_factor,
            dropout_rates=dropout_rates,
            training_metric=training_metric,
            label_weights=label_weights,
        )

    def evaluate_segments_by_label(
        self,
        segments_by_label: Dict[str, List[np.ndarray]],
        batch_size: int = 32,
        track_misclassifications: bool = True,
    ) -> dict:
        """
        Evaluate classifier on a dict of label -> list[np.ndarray segments].
        Returns per-class precision/recall/F1/support, overall accuracy and confusion matrix.

        Args:
            segments_by_label: Dict of label -> list of audio segments
            batch_size: Batch size for inference
            track_misclassifications: If True, include misclassification details in output
        """
        # Require label mapping
        if not hasattr(self, "label_to_idx") or not self.label_to_idx:
            return {
                "error": "Classifier has no label mapping. Train the classifier first.",
                "model_version": self.get_model_version(),
            }

        # Flatten dataset - also track sample indices for misclassification tracking
        X: List[np.ndarray] = []
        y_true_labels: List[str] = []
        sample_indices: List[tuple] = []  # (label, index_in_label_list)

        for label, segs in segments_by_label.items():
            if not isinstance(segs, list) or len(segs) == 0:
                continue
            # Only include labels the classifier knows
            if label not in self.label_to_idx:
                logger.warning(f"Skipping unknown label for model mapping: {label}")
                continue
            for idx, seg in enumerate(segs):
                # Handle both raw audio and (audio, sample_id) tuples
                if isinstance(seg, tuple):
                    audio_data = seg[0]
                else:
                    audio_data = seg

                X.append(audio_data)
                y_true_labels.append(label)
                sample_indices.append((label, idx))

        if not X:
            return {
                "error": "No evaluable segments after filtering by known labels.",
                "model_version": self.get_model_version(),
            }

        # Inference: get probability dicts to be robust to binary/multiclass
        prob_dicts: List[Dict[str, float]] = cast(
            List[Dict[str, float]],
            self.infer_audio_segment(
                np.stack(X), batch_size=batch_size, return_probabilities=True
            ),
        )

        # Convert y_true and predictions to indices
        label_to_idx = self.label_to_idx
        idx_to_label = (
            self.idx_to_label
            if hasattr(self, "idx_to_label")
            else {v: k for k, v in label_to_idx.items()}
        )

        y_true_idx: List[int] = []
        y_pred_idx: List[int] = []
        pred_confidences: List[float] = []
        pred_labels: List[
            str
        ] = []  # Store predicted labels for misclassification tracking
        valid_sample_indices: List[int] = []  # Track which samples are valid

        for i, (true_label, p) in enumerate(zip(y_true_labels, prob_dicts)):
            # Predicted label and its confidence
            if not p:
                # Empty probability dict; skip sample
                continue
            pred_label, pred_conf = max(p.items(), key=lambda kv: kv[1])
            if pred_label not in label_to_idx or true_label not in label_to_idx:
                # Skip unknown labels
                continue

            y_true_idx.append(label_to_idx[true_label])
            y_pred_idx.append(label_to_idx[pred_label])
            pred_confidences.append(float(pred_conf))
            pred_labels.append(pred_label)
            valid_sample_indices.append(i)

        if not y_true_idx:
            return {
                "error": "No evaluable samples after mapping to indices.",
                "model_version": self.get_model_version(),
            }

        y_true = np.array(y_true_idx, dtype=int)
        y_pred = np.array(y_pred_idx, dtype=int)

        # Determine classes present in the data (order by index)
        classes_in_data = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        class_labels = [idx_to_label[i] for i in classes_in_data]

        # Metrics
        overall_acc = float(accuracy_score(y_true, y_pred))
        precisions, recalls, f1_scores, supports = precision_recall_fscore_support(
            y_true, y_pred, labels=classes_in_data, zero_division=0
        )
        # Ensure array types for indexing (handles single-class edge cases)
        precisions = np.atleast_1d(precisions)
        recalls = np.atleast_1d(recalls)
        f1_scores = np.atleast_1d(f1_scores)
        supports = np.atleast_1d(cast(np.ndarray, supports))

        conf_mat = confusion_matrix(y_true, y_pred, labels=classes_in_data)

        # Per-class accuracy (TP/support)
        per_class = {}
        # Confidence summaries per predicted class
        correct_conf_sum = {i: 0.0 for i in classes_in_data}
        correct_conf_cnt = {i: 0 for i in classes_in_data}
        incorrect_conf_sum = {i: 0.0 for i in classes_in_data}
        incorrect_conf_cnt = {i: 0 for i in classes_in_data}

        for t, p, conf in zip(y_true, y_pred, pred_confidences):
            if t == p:
                correct_conf_sum[p] += conf
                correct_conf_cnt[p] += 1
            else:
                incorrect_conf_sum[p] += conf
                incorrect_conf_cnt[p] += 1

        # Map class id -> position in confusion matrix
        idx_pos_map = {cls_id: pos for pos, cls_id in enumerate(classes_in_data)}

        # Build per-class dict
        for cls_id, lbl in zip(classes_in_data, class_labels):
            pos = idx_pos_map[cls_id]
            tp = int(conf_mat[pos, pos]) if conf_mat.size > 0 else 0
            sup = int(supports[pos]) if pos < len(supports) else 0
            cls_acc = float(tp / sup) if sup > 0 else 0.0

            avg_conf_correct = (
                float(correct_conf_sum[cls_id] / correct_conf_cnt[cls_id])
                if correct_conf_cnt[cls_id] > 0
                else None
            )
            avg_conf_incorrect = (
                float(incorrect_conf_sum[cls_id] / incorrect_conf_cnt[cls_id])
                if incorrect_conf_cnt[cls_id] > 0
                else None
            )

            per_class[lbl] = {
                "support": sup,
                "precision": float(precisions[pos]) if pos < len(precisions) else 0.0,
                "recall": float(recalls[pos]) if pos < len(recalls) else 0.0,
                "f1": float(f1_scores[pos]) if pos < len(f1_scores) else 0.0,
                "accuracy": cls_acc,
                "avg_confidence_correct": avg_conf_correct,
                "avg_confidence_incorrect": avg_conf_incorrect,
            }

        # Macro/weighted F1
        macro_f1 = float(np.mean(f1_scores)) if len(f1_scores) > 0 else 0.0
        total = int(np.sum(supports))
        weighted_f1 = (
            float(np.sum(f1_scores * (supports / max(total, 1)))) if total > 0 else 0.0
        )

        # Build misclassifications summary if requested
        misclassifications = None
        if track_misclassifications and idx_to_label is not None:
            # Group misclassifications by (true_label -> predicted_label)
            misclassifications = {}
            for i, (true_idx, pred_idx, conf) in enumerate(
                zip(y_true_idx, y_pred_idx, pred_confidences)
            ):
                if true_idx != pred_idx:
                    true_lbl = idx_to_label[true_idx]
                    pred_lbl = idx_to_label[pred_idx]
                    key = f"{true_lbl} -> {pred_lbl}"
                    if key not in misclassifications:
                        misclassifications[key] = {
                            "true_label": true_lbl,
                            "predicted_label": pred_lbl,
                            "count": 0,
                            "avg_confidence": 0.0,
                            "sample_indices": [],  # Original indices in segments_by_label[true_label]
                        }
                    misclassifications[key]["count"] += 1
                    orig_idx = valid_sample_indices[i]
                    sample_label, sample_idx_in_label = sample_indices[orig_idx]
                    misclassifications[key]["sample_indices"].append(
                        {"index": sample_idx_in_label, "confidence": conf}
                    )

            # Calculate avg confidence for each misclassification type
            for key, data in misclassifications.items():
                if data["sample_indices"]:
                    data["avg_confidence"] = sum(
                        s["confidence"] for s in data["sample_indices"]
                    ) / len(data["sample_indices"])

            # Sort by count descending
            misclassifications = dict(
                sorted(
                    misclassifications.items(),
                    key=lambda x: x[1]["count"],
                    reverse=True,
                )
            )

        result = {
            "model_version": self.get_model_version(),
            "overall": {
                "samples": int(len(y_true)),
                "accuracy": overall_acc,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "num_classes": int(len(classes_in_data)),
            },
            "per_class": per_class,
            "confusion_matrix": {"labels": class_labels, "matrix": conf_mat.tolist()},
        }

        if misclassifications:
            result["misclassifications"] = misclassifications

        return result
