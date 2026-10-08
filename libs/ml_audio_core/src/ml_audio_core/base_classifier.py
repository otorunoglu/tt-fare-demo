import os
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np

from ml_audio_core.embedders.base_embedder import BaseEmbedder

# TensorFlow is optional (for inference-only on Raspberry Pi with tflite_runtime)
try:
    import tensorflow as tf  # type: ignore

    # Configure GPU memory growth to prevent TF from hoarding all memory
    # This leaves memory available for PyTorch (used by embedders like PANNs)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(f"Warning: Could not set memory growth: {e}")
except Exception:  # pragma: no cover
    tf = None  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from abc import ABC, abstractmethod

from ml_audio_core.augmentation import augment_audio_data

from ml_audio_core.audio_processing import AudioProcessor


class BaseAudioClassifier(ABC):
    """
    Generic base class for embedding-based audio classifiers.
    Embeddings are produced by any provided BaseEmbedder instance.
    """

    def __init__(
        self,
        model_dir,
        embedder: BaseEmbedder,
        backend="keras",
        model_prefix="audio_classifier",
        model_path: str | None = None,
        label_mapping_path: str | None = None,
        model_version: str | None = None,
    ):
        """
        Initialize the audio classifier.

        Args:
            model_dir: Directory containing model files
            embedder: Embedding model instance
            backend: "keras" or "tflite"
            model_prefix: Prefix for saved model files
            model_path: Optional explicit path to a specific .keras model file.
                       If provided, this model will be loaded instead of
                       discovering the latest model in model_dir.
                       This allows external control of which model is used
                       (e.g., loading from MLflow registry).
            label_mapping_path: Optional explicit path to a label_mapping.json file.
                       If provided, this label mapping will be used instead of
                       looking for a file next to the model. Useful when the
                       model and label mapping are stored in different locations
                       (e.g., when downloading from MLflow).
            model_version: Optional explicit version string for the model.
                          If provided, this will be returned by get_model_version()
                          instead of deriving it from the filename.
        """
        self.model_dir = Path(model_dir)
        self.embedder = embedder
        self.backend = backend.lower()
        self.model_prefix = model_prefix
        self._explicit_model_path = model_path  # Store for use in model loading
        self._explicit_label_mapping_path = (
            label_mapping_path  # Store for use in label mapping loading
        )
        self.model_version = model_version  # Store explicit version if provided

        self.embedding_dim = embedder.embedding_dim
        self.device = getattr(embedder, "device", "cpu")

        self.logger = logging.getLogger(self.__class__.__name__)

        # Label mapping
        self.label_to_idx: Optional[Dict[str, int]] = None
        self.idx_to_label: Optional[Dict[int, str]] = None
        self.valid_labels: Optional[list[str]] = None

        # TFLite state
        self.tflite_interpreter = None
        self.tflite_input_index = None
        self.tflite_output_index = None
        self.tflite_input_dtype = None
        self.tflite_input_shape = None

        # Optional audio processor (subclasses may override)
        self.audio_processor = None

        # Backend selection
        if self.backend == "tflite":
            if not self._load_tflite_model():
                self.logger.warning(
                    "No TFLite model found; falling back to Keras backend."
                )
                self.backend = "keras"
                self._init_or_create_keras_model()
        else:
            self._init_or_create_keras_model()

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------
    def _init_or_create_keras_model(self):
        """Load or create a Keras classification head.

        If an explicit model_path was provided at construction time,
        that model is loaded directly. Otherwise, we discover the latest
        model in model_dir.
        """
        # If explicit model path was provided, load it directly
        if self._explicit_model_path:
            if self.load_model_from_path(self._explicit_model_path):
                self.logger.info(
                    f"Loaded explicit model from: {self._explicit_model_path}"
                )
                return
            else:
                self.logger.warning(
                    f"Failed to load explicit model from {self._explicit_model_path}, "
                    "falling back to latest model discovery"
                )

        # Default behavior: discover latest model
        if not self._load_latest_model():
            if tf is None:
                raise RuntimeError("TensorFlow not available for Keras backend.")
            self.logger.info("No Keras model found — creating new model.")
            self.model = self.build_model(embedding_dim=self.embedding_dim)
            self.model_path = "untrained_model"

    def build_dense_head(self, embedding_dim=2048, num_classes=2) -> Any:
        """Simple default dense classifier head."""
        inp = tf.keras.layers.Input(shape=(embedding_dim,), name="embedding_input")
        x = tf.keras.layers.Dense(512, activation="relu")(inp)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        act = "sigmoid" if num_classes == 2 else "softmax"
        out = tf.keras.layers.Dense(
            1 if num_classes == 2 else num_classes, activation=act
        )(x)
        return tf.keras.Model(inp, out, name="audio_classifier")

    @abstractmethod
    def build_model(self, embedding_dim=2048, num_classes=2, dropout_rates=None) -> Any:
        """Implemented by subclasses."""
        pass

    # ------------------------------------------------------------------
    # Label Mapping
    # ------------------------------------------------------------------
    def _load_label_mapping(self, model_path, label_mapping_path: str | None = None):
        """Load label mapping for a given model (list-output safe).

        Args:
            model_path: Path to the model file (used to derive default label mapping path)
            label_mapping_path: Optional explicit path to label_mapping.json.
                              If provided, this path is used instead of deriving
                              from model_path.
        """
        import json
        from pathlib import Path

        # If explicit label mapping path is provided, use it
        if label_mapping_path:
            label_file = Path(label_mapping_path)
        else:
            # Construct label file path using parent and stem
            model_path = Path(model_path)
            label_file = model_path.parent / f"{model_path.stem}_label_mapping.json"

        if label_file.exists():
            with open(label_file, "r") as f:
                mapping = json.load(f)
            self.label_to_idx = mapping.get("label_to_idx", {})
            # Ensure idx_to_label has integer keys (JSON converts int keys to strings)
            idx_to_label_raw = mapping.get("idx_to_label", {})
            self.idx_to_label = {int(k): v for k, v in idx_to_label_raw.items()}
            self.valid_labels = mapping.get("valid_labels", [])
            self.logger.info(
                f"Loaded label mapping with {len(self.label_to_idx or {})} labels."
            )
        else:
            self.logger.warning(
                f"No label mapping found at {label_file}; classifier may return numeric indices."
            )

    # ------------------------------------------------------------------
    # TFLite Loading
    # ------------------------------------------------------------------
    def _load_tflite_model(self) -> bool:
        """
        Load a TensorFlow Lite model (.tflite). Returns True on success.

        Note: On development machines (Mac/Windows), TensorFlow's built-in TFLite interpreter
        is used. On Raspberry Pi, install tflite-runtime for a lighter footprint.
        """
        # Resolve path
        tflite_path_env = os.getenv("TFLITE_MODEL_PATH")
        tflite_path: Optional[Path] = None
        if tflite_path_env:
            tflite_path = Path(tflite_path_env).expanduser().resolve()
            if not tflite_path.exists():
                self.logger.error("TFLite model path does not exist: %s", tflite_path)
                return False
        else:
            candidates = list(self.model_dir.glob("*.tflite"))
            if not candidates:
                self.logger.warning("No .tflite models found in %s", self.model_dir)
                return False
            tflite_path = max(candidates, key=os.path.getctime)

        # Load interpreter - try tflite_runtime first (Pi), fallback to tensorflow.lite (dev)
        interpreter = None
        try:
            try:
                from tflite_runtime.interpreter import Interpreter  # type: ignore

                self.logger.debug("Using tflite_runtime.Interpreter")
            except (ImportError, ModuleNotFoundError):
                if tf is None:
                    raise RuntimeError(
                        "Neither tflite_runtime nor tensorflow is available. "
                        "Install tensorflow (dev) or tflite-runtime (Pi). uv pip install tflite-runtime (just tflite is not the right one!)"
                    )
                from tensorflow.lite import Interpreter  # type: ignore

                self.logger.debug("Using tensorflow.lite.Interpreter")
            interpreter = Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
        except Exception as e:
            self.logger.error("Failed to load TFLite model '%s': %s", tflite_path, e)
            return False

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        if len(input_details) != 1 or len(output_details) != 1:
            self.logger.warning(
                "Unexpected TFLite IO configuration (inputs=%d, outputs=%d). Using first of each.",
                len(input_details),
                len(output_details),
            )
        self.tflite_interpreter = interpreter
        self.tflite_input_index = input_details[0]["index"]
        self.tflite_output_index = output_details[0]["index"]
        self.tflite_input_dtype = input_details[0]["dtype"]
        self.tflite_input_shape = input_details[0].get("shape", None)
        self.model_path = str(tflite_path)
        self.backend = "tflite"

        # Load label mapping associated to this model, if present
        self._load_label_mapping(tflite_path)
        self.logger.info("Loaded TFLite model: %s", tflite_path)
        return True

    # ------------------------------------------------------------------
    # Inference / Embedding Utilities
    # ------------------------------------------------------------------
    def get_embeddings(
        self, audio_data: np.ndarray | list, batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract embeddings for a list or array of audio segments.

        Args:
            audio_data: List of audio segments or numpy array (N, samples)
            batch_size: Batch size for embedding extraction

        Returns:
            Numpy array of embeddings (N, embedding_dim)
        """
        # Type check / conversion
        if isinstance(audio_data, list):
            # If empty list, return empty
            if len(audio_data) == 0:
                return np.zeros((0, self.embedding_dim), dtype=np.float32)
            # Check if items are numpy arrays
            if isinstance(audio_data[0], np.ndarray):
                # We need to stack if they are consistent, or handle one by one?
                # The embedders usually expect a batch (N, samples).
                # If audio_data is a list of segments, let's treat it as a batch.
                pass

        # If input is a list of arrays, stack them safely?
        # But if they are different lengths, they can't be stacked into (N, samples) easily
        # unless the embedder handles padding or we do it.
        # PANNs embedder input expectation: (N, samples). Fixed samples?
        # Cnn14 usually handles variable length (it averages pooled features),
        # but `panns_inference` might expect valid tensor construction.
        # Let's assume calling code provides valid batchable data (e.g. same length or padded).
        # We will iterate in batches.

        embeddings = []
        total_samples = len(audio_data)

        for i in range(0, total_samples, batch_size):
            batch = audio_data[i : i + batch_size]
            # Convert list to array for embedder if needed
            if isinstance(batch, list):
                # Try to stack; if fails (inconsistent length), embedder might complain
                # For now assumes consistent length or robust embedder
                try:
                    batch = np.stack(batch)
                except ValueError:
                    # Fallback: process one by one if stacking fails?
                    # Or just let it fail. PANNs usually needs roughly consistent input or padding.
                    pass

            emb = self.embedder.embed(batch)
            embeddings.append(emb)

        if embeddings:
            return np.vstack(embeddings)
        else:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

    def compute_anomaly_score(
        self,
        audio_data: np.ndarray | list,
        reference_embeddings: Optional[np.ndarray] = None,
        reference_centroid: Optional[np.ndarray] = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Compute anomaly scores for audio segments based on distance to a reference distribution.

        Uses Cosine Distance to the centroid of the reference embeddings.
        Score range: [0, 2] (0 = identical to centroid, 2 = opposite).
        Typical range for PANNs is [0, 1].

        Args:
            audio_data: Audio segments to score
            reference_embeddings: Array of "normal" embeddings (N_ref, dim).
                                 Used to compute centroid if reference_centroid is not provided.
            reference_centroid: Pre-computed centroid vector (dim,).
                                If provided, reference_embeddings is ignored.
            batch_size: Batch size for processing audio_data

        Returns:
            scores: Array of anomaly scores (N,)
        """
        from sklearn.metrics.pairwise import cosine_distances

        # 1. Get embeddings for the query audio
        embeddings = self.get_embeddings(audio_data, batch_size=batch_size)
        if len(embeddings) == 0:
            return np.array([])

        # 2. Determine reference centroid
        if reference_centroid is not None:
            centroid = reference_centroid
        elif reference_embeddings is not None and len(reference_embeddings) > 0:
            centroid = np.mean(reference_embeddings, axis=0)
        else:
            raise ValueError(
                "Must provide either 'reference_embeddings' (with data) or 'reference_centroid' "
                "to compute anomaly scores."
            )

        # Ensure centroid is 2D (1, dim) for cdist
        if centroid.ndim == 1:
            centroid = centroid.reshape(1, -1)

        # 3. Compute Cosine Distance
        # cosine_distances returns (N_samples, N_ref), here (N, 1)
        dists = cosine_distances(embeddings, centroid)

        return dists.flatten()

    # ------------------------------------------------------------------
    # Keras model loading
    # ------------------------------------------------------------------
    def _load_latest_model(self) -> bool:
        """Load most recent .keras model in model_dir with basic integrity checks. Returns false if tf not available."""
        if tf is None:
            self.logger.warning(
                "TensorFlow not available; cannot load Keras model. Trying to load TFLite model instead."
            )
            return False

        # Filter by model_prefix to avoid loading incompatible models (e.g. panns vs yamnet)
        pattern = f"{self.model_prefix}*.keras"
        model_files = list(self.model_dir.glob(pattern))

        if not model_files:
            # Fallback to older behavior (any .keras) if strict prefix match fails?
            # Or better to be strict. Strict is safer for architecture separation.
            self.logger.warning(
                f"No saved model found matching '{pattern}' in {self.model_dir}"
            )
            return False

        latest_model = max(model_files, key=os.path.getctime)
        self.logger.info(f"Loading model: {latest_model}")
        try:
            model = tf.keras.models.load_model(latest_model, compile=False)
        except Exception as e:
            self.logger.error(f"Failed to load model file: {e}")
            return False

        # Materialize shapes to catch lazy-built subclass models
        if not model.built:
            try:
                model.build((None, self.embedding_dim))
            except Exception as e:
                self.logger.debug(f"Model build failed (may be subclassed): {e}")

        # Quick sanity check
        try:
            model.predict(
                np.zeros((1, self.embedding_dim), dtype=np.float32), verbose=0
            )
        except Exception as e:
            self.logger.debug(f"Sanity check predict failed: {e}")

        # Infer output size for debug
        try:
            from ml_audio_core.model_utils import infer_output_units

            units = infer_output_units(model)
            self.logger.info(f"Detected model output units: {units}")
        except Exception:
            units = None
            self.logger.warning(
                "Could not infer output units (will rely on label mapping if available)."
            )

        self.model = model
        self.model_path = latest_model

        # Load label mapping if it exists
        self._load_label_mapping(latest_model)

        self.logger.info("Model loaded successfully.")
        return True

    def load_model_from_path(
        self, model_path, label_mapping_path: str | None = None
    ) -> bool:
        """
        Load a specific Keras model from a given path.

        Args:
            model_path: Path to the .keras model file
            label_mapping_path: Optional explicit path to label_mapping.json.
                              If not provided, looks for label mapping next to the model.

        Returns:
            True if successful, False otherwise
        """
        from pathlib import Path

        model_path = Path(model_path)

        # Use explicit label mapping path if provided, otherwise fall back to instance variable
        effective_label_mapping_path = label_mapping_path or getattr(
            self, "_explicit_label_mapping_path", None
        )

        if not model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            return False

        self.logger.info(f"Loading model from: {model_path}")
        try:
            model = tf.keras.models.load_model(str(model_path), compile=False)
        except Exception as e:
            self.logger.error(f"Failed to load model file: {e}")
            return False

        # Quick sanity check
        try:
            model.predict(
                np.zeros((1, self.embedding_dim), dtype=np.float32), verbose=0
            )
        except Exception as e:
            self.logger.debug(f"Sanity check predict failed: {e}")

        self.model = model
        self.model_path = str(model_path)

        # Load label mapping - use explicit path if provided
        self._load_label_mapping(model_path, effective_label_mapping_path)

        self.logger.info(f"Model loaded successfully from {model_path}")
        return True

    def get_model_version(self):
        """Get the version of the model."""
        if hasattr(self, "model_version") and self.model_version:
            return self.model_version

        if hasattr(self, "model_path") and self.model_path:
            # return filename as version
            return os.path.basename(self.model_path).replace(".keras", "")
        else:
            return os.path.basename("untrained_model")

    def save_model(self, model, label_mapping, prefix="model"):
        import json

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.model_dir / f"{prefix}_{timestamp}.keras"
        model.save(path)

        # Save label mapping to separate JSON file
        if label_mapping:
            # Use parent and stem to construct the label file path
            label_file = path.parent / f"{path.stem}_label_mapping.json"
            with open(label_file, "w") as f:
                # Convert any non-serializable types
                serializable_mapping = {}
                for key, value in label_mapping.items():
                    if isinstance(value, dict):
                        serializable_mapping[key] = {
                            str(k): v for k, v in value.items()
                        }
                    else:
                        serializable_mapping[key] = value
                json.dump(serializable_mapping, f, indent=2)
            self.logger.info(f"Saved label mapping to {label_file}")

        return path

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, embeddings, labels, num_classes, lr=5e-5, epochs=20, batch_size=32):
        if tf is None:
            raise RuntimeError(
                "TensorFlow is required for training but is not installed."
            )
        model = self.build_dense_head(num_classes=num_classes)
        loss = "binary_crossentropy" if num_classes == 2 else "categorical_crossentropy"
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr), loss=loss, metrics=["accuracy"]
        )

        X_train, X_val, y_train, y_val = train_test_split(
            embeddings, labels, test_size=0.15, stratify=None, random_state=42
        )

        # compute class weights
        if num_classes == 2:
            cw = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(y_train),
                y=y_train,
            )
            class_weight = {0: cw[0], 1: cw[1]}
        else:
            cw = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(np.argmax(labels, axis=1)),
                y=np.argmax(labels, axis=1),
            )
            class_weight = {i: w for i, w in enumerate(cw)}

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
                monitor="val_accuracy",
                mode="max",
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5
            ),
        ]

        hist = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

        self.model = model
        return model, hist

    def train_from_segments(
        self,
        training_data: Dict[str, list],
        learning_rate: float = 5e-5,
        epochs: int = 20,
        batch_size: int = 16,
        min_samples_per_class: int = 30,
        max_samples_per_class: Optional[int] = None,
        max_samples_per_class_overrides: Optional[Dict[str, int]] = None,
        use_augmentation: bool = True,
        augmentation_factor: int = 2,
        full_retraining: bool = True,
        dropout_rates: Optional[list] = None,
        training_metric: str = "accuracy",
        label_weights: Optional[Dict[str, float]] = None,
    ):
        """
        High-level training pipeline that takes raw audio segments grouped by label,
        performs optional augmentation, extracts embeddings, trains a classifier,
        and saves the trained model + label mapping.

        Each entry in training_data can be either a list of np.ndarray segments
        OR a list of (np.ndarray, str_id) tuples.

        Args:
            max_samples_per_class: If set, cap the number of samples per class to this value
            max_samples_per_class_overrides: Optional dict of per-label caps that
                override max_samples_per_class for matching labels.
            training_metric: Metric to use for training and early stopping.
            label_weights: Optional dict mapping label names to importance weights.
        """
        if tf is None:
            raise RuntimeError(
                "TensorFlow is required for training but is not installed."
            )
        from sklearn.model_selection import train_test_split
        from sklearn.utils.class_weight import compute_class_weight

        # ----------------------------
        # 1. Validate input and apply max_samples_per_class cap
        # ----------------------------
        samples_after_cap = {}  # Track actual samples after capping
        has_global_cap = max_samples_per_class is not None
        has_per_label_caps = bool(max_samples_per_class_overrides)
        if has_global_cap or has_per_label_caps:
            self.logger.info(
                "Applying sample caps "
                f"(global={max_samples_per_class}, per_label={bool(max_samples_per_class_overrides)})"
            )
            for lbl, items in list(training_data.items()):
                cap = None
                if max_samples_per_class_overrides and lbl in max_samples_per_class_overrides:
                    cap = max_samples_per_class_overrides[lbl]
                else:
                    cap = max_samples_per_class

                if cap is not None and len(items) > cap:
                    # Random sample to cap
                    indices = np.random.choice(len(items), cap, replace=False)
                    training_data[lbl] = [items[i] for i in indices]
                    self.logger.info(
                        f"  {lbl}: {len(items)} -> {cap} samples"
                    )
                samples_after_cap[lbl] = len(training_data[lbl])
        else:
            # No capping, just record original counts
            samples_after_cap = {lbl: len(segs) for lbl, segs in training_data.items()}

        valid_labels = [
            lbl
            for lbl, segs in training_data.items()
            if len(segs) >= min_samples_per_class
        ]
        if len(valid_labels) < 2:
            self.logger.error("Need at least 2 valid classes to train.")
            return None

        self.logger.info(f"Training with {len(valid_labels)} classes: {valid_labels}")

        # ----------------------------
        # 2. Label mappings
        # ----------------------------
        self.label_to_idx = {label: i for i, label in enumerate(valid_labels)}
        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}
        self.valid_labels = valid_labels

        # ----------------------------
        # 3. Aggregate and augment
        # ----------------------------
        all_segments, all_labels, all_ids = [], [], []
        samples_after_augmentation = {}  # Track samples after augmentation
        for lbl, items in training_data.items():
            if lbl not in valid_labels:
                continue

            # Unpack items (might be audio or (audio, id))
            segs = []
            ids = []
            for item in items:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) == 2
                    and isinstance(item[0], np.ndarray)
                ):
                    segs.append(item[0])
                    ids.append(item[1])
                else:
                    segs.append(item)
                    ids.append(f"unidentified_{lbl}_{len(ids)}")

            original_count = len(segs)
            applied_aug_factor = 0
            if use_augmentation and len(segs) < 200:
                applied_aug_factor = augmentation_factor * (2 if len(segs) < 50 else 1)
                # Note: augmentation currently doesn't track IDs, so we'll duplicate IDs with a suffix
                segs = augment_audio_data(segs, applied_aug_factor)

                # Update IDs to match augmented segments
                # augment_audio_data returns len(segs) * applied_aug_factor (if > 1)
                if len(segs) > len(ids):
                    new_ids = []
                    # Originals (first original_count segments)
                    new_ids.extend(ids)

                    # Augmented IDs (remaining segments)
                    num_augs_needed = len(segs) - len(ids)
                    for j in range(num_augs_needed):
                        original_id = ids[j % len(ids)]
                        aug_num = (j // len(ids)) + 1
                        new_ids.append(f"{original_id}#aug{aug_num}")
                    ids = new_ids

            samples_after_augmentation[lbl] = len(segs)
            if len(segs) != original_count:
                self.logger.info(
                    f"  Augmented {lbl}: {original_count} -> {len(segs)} samples (factor={applied_aug_factor})"
                )

            all_segments.extend(segs)
            all_labels.extend([self.label_to_idx[lbl]] * len(segs))
            all_ids.extend(ids)

        all_segments = np.array(all_segments)
        all_labels = np.array(all_labels)
        all_ids = np.array(all_ids)

        # ----------------------------
        # 4. Embedding generation
        # ----------------------------
        self.logger.info(f"Generating embeddings for {len(all_segments)} segments…")
        embeddings = self.get_embeddings(all_segments, batch_size=batch_size)

        # ----------------------------
        # 5. Prepare labels and model
        # ----------------------------
        num_classes = len(valid_labels)
        y = (
            tf.keras.utils.to_categorical(all_labels, num_classes)
            if num_classes > 2
            else all_labels
        )
        loss = "categorical_crossentropy" if num_classes > 2 else "binary_crossentropy"

        if full_retraining or not hasattr(self, "model") or self.model is None:
            self.logger.info("Building new model (full retraining)")
            tf.keras.backend.clear_session()
            # Pass dropout_rates if the subclass build_model supports it
            try:
                self.model = self.build_model(
                    num_classes=num_classes, dropout_rates=dropout_rates
                )
            except TypeError:
                # Fallback for subclasses that don't support dropout_rates
                self.model = self.build_model(num_classes=num_classes)
        else:
            self.logger.info("Reusing existing model for fine-tuning")

        # ----------------------------
        # 5b. Configure training metrics
        # ----------------------------
        metrics_list = ["accuracy"]
        early_stop_monitor = "val_accuracy"

        if training_metric == "macro_f1" and num_classes > 2:
            # Define Macro F1 metric for imbalanced data
            class MacroF1Score(tf.keras.metrics.Metric):
                """Macro F1 score - treats all classes equally regardless of size."""

                def __init__(self, num_classes, name="macro_f1", **kwargs):
                    super().__init__(name=name, **kwargs)
                    self._num_classes = num_classes
                    self.true_positives = self.add_weight(
                        name="tp", shape=(num_classes,), initializer="zeros"
                    )
                    self.false_positives = self.add_weight(
                        name="fp", shape=(num_classes,), initializer="zeros"
                    )
                    self.false_negatives = self.add_weight(
                        name="fn", shape=(num_classes,), initializer="zeros"
                    )

                def update_state(self, y_true, y_pred, sample_weight=None):
                    y_true_idx = tf.argmax(y_true, axis=-1)
                    y_pred_idx = tf.argmax(y_pred, axis=-1)

                    # One-hot encode predictions and ground truth
                    y_true_onehot = tf.one_hot(y_true_idx, self._num_classes)
                    y_pred_onehot = tf.one_hot(y_pred_idx, self._num_classes)

                    # Compute per-class counts (sum over batch dimension)
                    tp_batch = tf.reduce_sum(y_true_onehot * y_pred_onehot, axis=0)
                    fp_batch = tf.reduce_sum(
                        (1 - y_true_onehot) * y_pred_onehot, axis=0
                    )
                    fn_batch = tf.reduce_sum(
                        y_true_onehot * (1 - y_pred_onehot), axis=0
                    )

                    self.true_positives.assign_add(tp_batch)
                    self.false_positives.assign_add(fp_batch)
                    self.false_negatives.assign_add(fn_batch)

                def result(self):
                    precision = self.true_positives / (
                        self.true_positives
                        + self.false_positives
                        + tf.keras.backend.epsilon()
                    )
                    recall = self.true_positives / (
                        self.true_positives
                        + self.false_negatives
                        + tf.keras.backend.epsilon()
                    )
                    f1 = (
                        2
                        * precision
                        * recall
                        / (precision + recall + tf.keras.backend.epsilon())
                    )
                    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
                    return tf.reduce_mean(f1)

                def reset_state(self):
                    self.true_positives.assign(tf.zeros((self._num_classes,)))
                    self.false_positives.assign(tf.zeros((self._num_classes,)))
                    self.false_negatives.assign(tf.zeros((self._num_classes,)))

            metrics_list.append(MacroF1Score(num_classes, name="macro_f1"))
            early_stop_monitor = "val_macro_f1"
            self.logger.info(
                "Using macro_f1 metric for imbalanced multi-class training"
            )

        elif training_metric == "balanced_accuracy" and num_classes > 2:
            # Balanced accuracy = mean recall per class
            class BalancedAccuracy(tf.keras.metrics.Metric):
                """Balanced accuracy - mean recall across all classes."""

                def __init__(self, num_classes, name="balanced_accuracy", **kwargs):
                    super().__init__(name=name, **kwargs)
                    self._num_classes = num_classes
                    self.true_positives = self.add_weight(
                        name="tp", shape=(num_classes,), initializer="zeros"
                    )
                    self.class_totals = self.add_weight(
                        name="totals", shape=(num_classes,), initializer="zeros"
                    )

                def update_state(self, y_true, y_pred, sample_weight=None):
                    y_true_idx = tf.argmax(y_true, axis=-1)
                    y_pred_idx = tf.argmax(y_pred, axis=-1)

                    # One-hot encode
                    y_true_onehot = tf.one_hot(y_true_idx, self._num_classes)
                    y_pred_onehot = tf.one_hot(y_pred_idx, self._num_classes)

                    # Compute per-class counts
                    tp_batch = tf.reduce_sum(y_true_onehot * y_pred_onehot, axis=0)
                    totals_batch = tf.reduce_sum(y_true_onehot, axis=0)

                    self.true_positives.assign_add(tp_batch)
                    self.class_totals.assign_add(totals_batch)

                def result(self):
                    recall_per_class = self.true_positives / (
                        self.class_totals + tf.keras.backend.epsilon()
                    )
                    recall_per_class = tf.where(
                        tf.math.is_nan(recall_per_class),
                        tf.zeros_like(recall_per_class),
                        recall_per_class,
                    )
                    return tf.reduce_mean(recall_per_class)

                def reset_state(self):
                    self.true_positives.assign(tf.zeros((self._num_classes,)))
                    self.class_totals.assign(tf.zeros((self._num_classes,)))

            metrics_list.append(BalancedAccuracy(num_classes, name="balanced_accuracy"))
            early_stop_monitor = "val_balanced_accuracy"
            self.logger.info(
                "Using balanced_accuracy metric for imbalanced multi-class training"
            )

        elif training_metric == "weighted_f1" and num_classes > 2:
            # Weighted F1 score - weights classes by training_importance
            # Get weights for each class index from label_weights
            importance_weights = []
            for i in range(num_classes):
                label_name = self.idx_to_label[i]
                weight = label_weights.get(label_name, 1.0) if label_weights else 1.0
                importance_weights.append(weight)
            weights_tensor = tf.constant(importance_weights, dtype=tf.float32)

            # Log the weights being used
            weights_info = {
                self.idx_to_label[i]: importance_weights[i] for i in range(num_classes)
            }
            self.logger.info(
                f"Using weighted_f1 metric with class weights: {weights_info}"
            )

            class WeightedF1Score(tf.keras.metrics.Metric):
                """Weighted F1 score - weights classes by training_importance."""

                def __init__(
                    self, num_classes, class_weights, name="weighted_f1", **kwargs
                ):
                    super().__init__(name=name, **kwargs)
                    self._num_classes = num_classes
                    self._class_weights = class_weights  # TF constant tensor
                    self.true_positives = self.add_weight(
                        name="tp", shape=(num_classes,), initializer="zeros"
                    )
                    self.false_positives = self.add_weight(
                        name="fp", shape=(num_classes,), initializer="zeros"
                    )
                    self.false_negatives = self.add_weight(
                        name="fn", shape=(num_classes,), initializer="zeros"
                    )

                def update_state(self, y_true, y_pred, sample_weight=None):
                    y_true_idx = tf.argmax(y_true, axis=-1)
                    y_pred_idx = tf.argmax(y_pred, axis=-1)

                    # One-hot encode
                    y_true_onehot = tf.one_hot(y_true_idx, self._num_classes)
                    y_pred_onehot = tf.one_hot(y_pred_idx, self._num_classes)

                    # Compute per-class counts
                    tp_batch = tf.reduce_sum(y_true_onehot * y_pred_onehot, axis=0)
                    fp_batch = tf.reduce_sum(
                        (1 - y_true_onehot) * y_pred_onehot, axis=0
                    )
                    fn_batch = tf.reduce_sum(
                        y_true_onehot * (1 - y_pred_onehot), axis=0
                    )

                    self.true_positives.assign_add(tp_batch)
                    self.false_positives.assign_add(fp_batch)
                    self.false_negatives.assign_add(fn_batch)

                def result(self):
                    precision = self.true_positives / (
                        self.true_positives
                        + self.false_positives
                        + tf.keras.backend.epsilon()
                    )
                    recall = self.true_positives / (
                        self.true_positives
                        + self.false_negatives
                        + tf.keras.backend.epsilon()
                    )
                    f1 = (
                        2
                        * precision
                        * recall
                        / (precision + recall + tf.keras.backend.epsilon())
                    )
                    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
                    # Weighted average using training_importance weights
                    weighted_f1 = tf.reduce_sum(
                        f1 * self._class_weights
                    ) / tf.reduce_sum(self._class_weights)
                    return weighted_f1

                def reset_state(self):
                    self.true_positives.assign(tf.zeros((self._num_classes,)))
                    self.false_positives.assign(tf.zeros((self._num_classes,)))
                    self.false_negatives.assign(tf.zeros((self._num_classes,)))

            metrics_list.append(
                WeightedF1Score(num_classes, weights_tensor, name="weighted_f1")
            )
            early_stop_monitor = "val_weighted_f1"

        else:
            if training_metric != "accuracy":
                self.logger.warning(
                    f"Unknown or inapplicable training_metric '{training_metric}', falling back to accuracy"
                )
            self.logger.info("Using accuracy metric for training")

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=loss,
            metrics=metrics_list,
        )

        # ----------------------------
        # 6. Train/test split + weights
        # ----------------------------
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            embeddings,
            y,
            all_ids,
            test_size=0.15,
            stratify=(y if num_classes == 2 else np.argmax(y, axis=1)),
            random_state=42,
        )

        # Ensure y is a flat numpy array of class indices
        if num_classes > 2:
            y_classes = np.argmax(y, axis=1)
        else:
            y_classes = np.asarray(y).flatten()

        cw = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_classes),
            y=y_classes,
        )
        class_weight = {i: w for i, w in enumerate(cw)}

        # ----------------------------
        # 7. Train
        # ----------------------------
        self.logger.info(f"Early stopping will monitor: {early_stop_monitor}")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=15,
                restore_best_weights=True,
                monitor=early_stop_monitor,
                mode="max",
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7
            ),
        ]

        hist = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )

        # ----------------------------
        # 8. Save model + mapping
        # ----------------------------
        label_mapping = {
            "label_to_idx": self.label_to_idx,
            "idx_to_label": self.idx_to_label,
            "valid_labels": self.valid_labels,
        }
        # Always save locally
        save_path = self.save_model(self.model, label_mapping, prefix=self.model_prefix)
        self.model_path = str(save_path)
        self.logger.info(f"Model trained and saved to {save_path}")

        # Store training info for reporting
        # Filter out augmented IDs from the test pool to keep only "real" segments
        real_test_ids = [str(id_val) for id_val in ids_val if "#aug" not in str(id_val)]

        # Group test IDs by label for better manifest structure
        test_pool_by_label = {}
        if num_classes > 2:
            test_preds = np.argmax(y_val, axis=1)
        else:
            test_preds = y_val

        for rid, pred_idx in zip(ids_val, test_preds):
            rid_str = str(rid)
            if "#aug" in rid_str:
                continue
            lbl = self.idx_to_label[int(pred_idx)]
            test_pool_by_label.setdefault(lbl, []).append(rid_str)

        self._last_training_info = {
            "samples_after_cap": samples_after_cap,
            "samples_after_augmentation": samples_after_augmentation,
            "total_training_samples": len(all_segments),
            "test_pool": test_pool_by_label,
            "total_test_samples": len(ids_val),
            "real_test_samples": len(real_test_ids),
        }

        return hist

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def process_audio_file(
        self,
        file_path: str,
        return_probabilities: bool = True,
        segment_duration: float = 2.0,
        overlap: float = 0.5,
        batch_size: int | None = None,
    ) -> list[dict]:
        """
        Process a full audio file using sliding windows.
        Returns:
        list of dicts with keys:
            start_time, end_time, prediction, probability, probabilities
        """

        # Ensure an AudioProcessor exists
        if self.audio_processor is None:
            self.audio_processor = AudioProcessor()

        # Load + segment audio
        audio, _ = self.audio_processor.load_and_resample_audio(file_path)
        windows, times = self.audio_processor.create_sliding_windows(
            audio, segment_duration, overlap
        )

        if batch_size is None:
            batch_size = int(os.getenv("INFERENCE_BATCH_SIZE", "32"))

        results = []

        # Process in batches
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i : i + batch_size]
            batch_times = times[i : i + batch_size]

            # Run inference
            preds = self.predict_batch(
                np.array(batch_windows), return_probabilities=return_probabilities
            )

            # Format final results
            for (start_time, end_time), pred in zip(batch_times, preds):
                if return_probabilities:
                    # pred is a probability dict
                    top = max(pred.items(), key=lambda x: x[1])
                    pred_label, pred_prob = top

                    results.append(
                        {
                            "start_time": start_time,
                            "end_time": end_time,
                            "prediction": pred_label,
                            "probability": pred_prob,
                            "probabilities": pred,
                        }
                    )
                else:
                    # pred is a label string
                    results.append(
                        {
                            "start_time": start_time,
                            "end_time": end_time,
                            "prediction": pred,
                            "probability": None,
                            "probabilities": None,
                        }
                    )

        return results

    def predict_batch(
        self,
        audio_batch: np.ndarray | None = None,
        batch_size: int | None = None,
        return_probabilities: bool = False,
        embeddings: np.ndarray | None = None,
    ) -> list | np.ndarray:
        if embeddings is None:
            if audio_batch is None:
                raise ValueError("Either audio_batch or embeddings must be provided.")
            if batch_size is None:
                batch_size = len(audio_batch)

            # Compute embeddings in batches
            embeddings_list = []
            for i in range(0, len(audio_batch), batch_size):
                emb = self.embedder.embed(audio_batch[i : i + batch_size])
                embeddings_list.append(emb)
            embeddings = np.vstack(embeddings_list)
        else:
            # Ensure 2D array
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

        preds = self._predict_from_embeddings(embeddings)

        # Binary classifier
        if preds.ndim == 1 or preds.shape[1] == 1:
            preds = preds.flatten()

            if return_probabilities:
                prob_dicts = []
                for prob in preds:
                    d = {0: float(1 - prob), 1: float(prob)}
                    if self.idx_to_label:
                        d = {
                            self.idx_to_label.get(0, "class_0"): d[0],
                            self.idx_to_label.get(1, "class_1"): d[1],
                        }
                    prob_dicts.append(d)
                return prob_dicts

            # Hard class
            labels = (preds > 0.5).astype(int)
            if self.idx_to_label:
                return [self.idx_to_label.get(int(x), str(x)) for x in labels]
            return labels

        # Multi-class classifier
        if return_probabilities:
            result = []
            for row in preds:
                d = dict(enumerate(row))
                if self.idx_to_label:
                    d = {self.idx_to_label[k]: float(v) for k, v in d.items()}
                result.append(d)
            return result

        # Hard max
        labels = np.argmax(preds, axis=1)
        if self.idx_to_label:
            return [self.idx_to_label.get(int(x), str(x)) for x in labels]
        return labels

    def _predict_from_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict from embeddings using chosen backend."""
        # TFLite backend
        if self.backend == "tflite" and self.tflite_interpreter is not None:
            outputs = []
            bs = 64
            for i in range(0, len(embeddings), bs):
                chunk = embeddings[i : i + bs]

                try:
                    self.tflite_interpreter.resize_tensor_input(
                        self.tflite_input_index, chunk.shape, strict=False
                    )
                    self.tflite_interpreter.allocate_tensors()
                except Exception:
                    pass

                chunk_cast = chunk.astype(self.tflite_input_dtype)
                self.tflite_interpreter.set_tensor(self.tflite_input_index, chunk_cast)
                self.tflite_interpreter.invoke()
                out = self.tflite_interpreter.get_tensor(self.tflite_output_index)
                outputs.append(out)
            return np.vstack(outputs)

        # Keras backend
        if tf is None:
            raise RuntimeError("TensorFlow not available for Keras backend.")
        outputs = []
        bs = 64
        for i in range(0, len(embeddings), bs):
            pred = self.model.predict(embeddings[i : i + bs], verbose=0)
            outputs.append(pred)
        return np.vstack(outputs)
