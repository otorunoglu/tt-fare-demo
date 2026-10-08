import logging
import os
import numpy as np

# import librosa # Removing direct dependency in favor of Rust backend
import soundfile as sf
import torch
from typing import cast
from .generated_types import LabelProbabilities, OTHER_VOCALIZATIONS

from .smartstable_types import MultiStallBatchResultReturn
from .soundscape_classifier import SoundscapeClassifier
from .model_registry import ModelRegistry
from .config import load_anomaly_config, load_metamodel_config
from ml_audio_core.audio_processing import (
    AudioProcessor,
    extract_loudness_features,
    extract_spectral_features,
)

logger = logging.getLogger("MultiStallDetectorManager")
# Set log level debug
logging.basicConfig(level=logging.DEBUG)

# Audio parameters
SR = 16000
WIN_SECONDS = 2.0
WIN_OVERLAP = 0.5
N_MELS = 64
HOP_LENGTH = 512
FIXED_TBINS = 96


class MultiStallDetectorManager:
    def __init__(
        self,
        model_path: str | None = None,
        label_mapping_path: str | None = None,
        model_version: str | None = None,
        model_architecture: str = "panns",
    ):
        """
        Initialize the MultiStallDetectorManager.

        Args:
            model_path: Optional explicit path to a .keras model file.
            label_mapping_path: Optional explicit path to a label_mapping.json file.
            model_version: Optional explicit version string for the model.
            model_architecture: "panns" or "yamnet".
        """
        self.detectors = {}  # Cache of loaded detectors
        self.max_cached_models = 10  # Limit memory usage
        self.soundscape_classifier = SoundscapeClassifier(
            model_path=model_path,
            label_mapping_path=label_mapping_path,
            model_version=model_version,
            model_architecture=model_architecture,
        )
        self.model_registry = ModelRegistry()

        # Add metamodel tracking
        self.metamodels = {}  # stable_id -> MetaModel
        self.metamodel_results = {}  # stable_id -> latest MetaModelResult
        self.last_metamodel_analysis = {}  # stable_id -> timestamp
        self.last_prediction_time = {}  # stable_id -> timestamp (lightweight tracking)

        logger.info("MultiStallDetectorManager initialized with metamodel support")

        self.has_gpu = torch.cuda.is_available()
        self.device = "cuda" if self.has_gpu else "cpu"
        # Log device info
        if self.has_gpu:
            logger.info(f"🏎️  GPU detected: Using {self.device} for acceleration")
        else:
            logger.info(f"🐢  No GPU detected: Using {self.device} for processing")

        # Load global config to check for audio backend preference
        metamodel_config = load_metamodel_config()
        audio_backend = metamodel_config.model_parameters.get("audio_backend", "rust")

        self.audio_processor = AudioProcessor(
            sr=SR,
            win_seconds=WIN_SECONDS,
            win_overlap=WIN_OVERLAP,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH,
            fixed_tbins=FIXED_TBINS,
            backend=audio_backend,
        )

    def get_soundscape_classifier(self) -> SoundscapeClassifier:
        """Get soundscape classifier"""
        return self.soundscape_classifier

    def extract_stall_info_from_filename(self, filename):
        """Extract stable/stall info from filename"""
        # find stable and stall from filename at arbitrary position and extract the numbers alongside
        # even if it is a url
        # Parse: /label-studio/data/audio/stable01_stall01_horse01_mic.wav

        # strip the path and get the filename
        filename = os.path.basename(filename)
        # Split by underscores
        parts = filename.split("_")
        if len(parts) >= 3:
            stall_id = None
            stable_id = None
            # Check each part for stable/stall identifiers
            for p in parts:
                if p.startswith("stable"):
                    stable_id = p
                if p.startswith("stall"):
                    stall_id = p
            if stable_id is None or stall_id is None:
                print(
                    "Warning: Filename does not contain valid stable/stall identifiers returning default values"
                )
                return "default", "default"  # Default fallback
            # print(f"Extracted stable_id: {stable_id}, stall_id: {stall_id}")
            return stable_id, stall_id
        return "default", "default"  # Default fallback

    def _evict_oldest_detector(self):
        """Evict the least recently used detector"""
        if self.detectors:
            oldest_key = next(iter(self.detectors))
            del self.detectors[oldest_key]
            print(f"Evicted detector for {oldest_key} to free up memory")

    def train_soundscape_classifier_multiclass(
        self,
        segments_by_label: dict,
        learning_rate: float = 1e-4,
        epochs: int = 15,
        batch_size: int = 16,
        stall_id: str = "default",
        stable_id: str = "default",
        min_samples_per_class: int = 30,
        max_samples_per_class: int | None = None,
        max_samples_per_class_overrides: dict[str, int] | None = None,
        augmentation_factor: int = 2,
        full_retraining: bool = True,
        dropout_rates: list = [],
        training_metric: str = "accuracy",
        label_weights: dict | None = None,
        model_architecture: str = "panns",
    ):
        """
        Train the global soundscape classifier with multi-class data.
        segments_by_label: Dict[label_value] -> List[np.ndarray]
        dropout_rates: Optional list of 3 dropout rates [layer1, layer2, layer3]
        max_samples_per_class: If set, cap the number of samples per class (helps with imbalance)
        max_samples_per_class_overrides: Optional per-label cap overrides.
        augmentation_factor: Factor for data augmentation on small classes
        training_metric: Metric for training/early stopping ("accuracy", "macro_f1", "balanced_accuracy", "weighted_f1")
        label_weights: Dict of label -> importance weight for weighted_f1 metric
        model_architecture: "panns" or "yamnet".
        """
        if not isinstance(segments_by_label, dict) or not segments_by_label:
            raise ValueError(
                "segments_by_label must be a non-empty dict of label -> segments"
            )

        # Log counts
        totals = {k: len(v) for k, v in segments_by_label.items()}
        logger.info(
            f"Training soundscape classifier (multiclass) with samples per label: {totals}"
        )

        # Check if we need to switch architecture
        current_arch = getattr(
            self.soundscape_classifier, "model_architecture", "panns"
        )
        if current_arch != model_architecture:
            logger.info(
                f"Switching model architecture from {current_arch} to {model_architecture}..."
            )
            self.soundscape_classifier = SoundscapeClassifier(
                model_architecture=model_architecture
            )

        detector = self.get_soundscape_classifier()
        history = detector.train_soundscape_classifier(
            training_data_dict=segments_by_label,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            min_samples_per_class=min_samples_per_class,
            max_samples_per_class=max_samples_per_class,
            max_samples_per_class_overrides=max_samples_per_class_overrides,
            augmentation_factor=augmentation_factor,
            dropout_rates=dropout_rates,
            training_metric=training_metric,
            label_weights=label_weights,
        )
        return history

    # Inference of an audio file with
    def process_audio_file(
        self,
        file_path,
        segment_duration=2.0,
        overlap=0.5,
        batch_size=None,
        stable_id="default",
        stall_id="default",
    ) -> list[MultiStallBatchResultReturn]:
        """
        Unified audio processing with automatic GPU/CPU selection and smart memory management
        """
        # Auto-detect optimal batch size based on device and file size
        if batch_size is None:
            batch_size = self._get_optimal_batch_size(file_path)

        logger.info(
            f"Processing {file_path} with device={self.device}, batch_size={batch_size}"
        )

        # Get file info (unused here, used in sub-methods)
        # info = sf.info(file_path)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        # Choose processing strategy based on file size and available memory
        # Increasing limit to 500MB as DGX has plenty of RAM and full-load is more reliable
        if file_size_mb < 500 and self.has_gpu:  # Files under 500MB - load entirely
            logger.info(
                f"🏎️  Using full-file loading strategy (file: {file_size_mb:.1f}MB)"
            )
            return self._process_file_full_load(
                file_path, segment_duration, overlap, batch_size, stable_id, stall_id
            )
        else:  # Large files or limited memory - stream
            logger.info(f"🌊  Using streaming strategy (file: {file_size_mb:.1f}MB)")
            return self._process_file_streaming(
                file_path, segment_duration, overlap, batch_size, stable_id, stall_id
            )

    def _get_optimal_batch_size(self, file_path):
        """Determine optimal batch size based on device and file size"""
        if self.has_gpu:
            # GPU can handle larger batches
            return 32  # Lowered from 128 (for now)
        else:
            # CPU should use smaller batches
            return 16

    def _process_file_full_load(
        self, file_path, segment_duration, overlap, batch_size, stable_id, stall_id
    ) -> list[MultiStallBatchResultReturn]:
        """Use consistent preprocessing."""

        audio, _ = self.audio_processor.load_and_resample_audio(file_path)
        windows, times, _ = self.audio_processor.create_sliding_windows(
            audio, segment_duration, overlap
        )

        results = []
        batch_audio = []
        # batch_mels = [] # Unused
        batch_times = []

        for window, (start_time, end_time) in zip(windows, times):
            # USE CONSISTENT PREPROCESSING - for inference, not training

            # Pass raw audio instead of pre-computed mels.
            # The model now handles MelSpectrogram calculation internally.

            batch_audio.append(window)

            batch_times.append((start_time, end_time))

            if len(batch_audio) >= batch_size:
                self._process_batch(
                    batch_audio,
                    batch_times,
                    results,
                    stable_id,
                    stall_id,
                    file_path,
                )
                batch_audio, batch_times = [], []

        # Process remaining
        if batch_audio:
            self._process_batch(
                batch_audio,
                batch_times,
                results,
                stable_id,
                stall_id,
                file_path,
            )

        return results

    def _process_file_streaming(
        self, file_path, segment_duration, overlap, batch_size, stable_id, stall_id
    ) -> list[MultiStallBatchResultReturn]:
        """Stream file in chunks (for large files or limited memory)"""
        info = sf.info(file_path)
        segment_samples_target = int(segment_duration * SR)
        hop_samples_target = int(segment_samples_target * (1 - overlap))

        results = []
        processed_segments = 0

        # Smaller chunk size for streaming (at original sample rate)
        # We read chunks of ~30 seconds worth of audio
        chunk_size_frames = int(max(10 * segment_duration, 30.0) * info.samplerate)

        with sf.SoundFile(file_path) as f:
            while f.tell() < info.frames:
                # Read chunk
                chunk_start_frame = f.tell()
                remaining_frames = info.frames - chunk_start_frame
                current_chunk_size = min(chunk_size_frames, remaining_frames)

                chunk = f.read(current_chunk_size)

                # Preprocessing
                if len(chunk.shape) > 1:
                    chunk = np.mean(chunk, axis=1)

                if info.samplerate != SR:
                    # Use Rust-based resampling for consistency/speed
                    import audio_preprocessing

                    chunk = audio_preprocessing.resample_numpy(
                        chunk, orig_sr=info.samplerate, target_sr=SR
                    )

                # Process segments in this chunk
                # chunk_start_time must use the ORIGINAL start frame
                chunk_start_time = chunk_start_frame / info.samplerate

                batch_segments = []
                batch_times = []

                for i in range(
                    0, len(chunk) - segment_samples_target + 1, hop_samples_target
                ):
                    segment = chunk[i : i + segment_samples_target]
                    start_time = chunk_start_time + (i / SR)

                    batch_segments.append(segment)
                    batch_times.append((start_time, start_time + segment_duration))

                    # Process batch when full
                    if len(batch_segments) >= batch_size:
                        batch_results = self._process_segment_batch(
                            batch_segments, batch_times, stable_id, stall_id, file_path
                        )
                        results.extend(batch_results)
                        processed_segments += len(batch_segments)
                        batch_segments, batch_times = [], []

                        if processed_segments % 1000 == 0:
                            logger.info(f"Processed {processed_segments} segments")

                # Process remaining segments in chunk
                if batch_segments:
                    batch_results = self._process_segment_batch(
                        batch_segments, batch_times, stable_id, stall_id, file_path
                    )
                    results.extend(batch_results)
                    processed_segments += len(batch_segments)

                # Overlap handling: seek back so the next chunk starts after the last processed window's hop
                if f.tell() < info.frames:
                    # Calculate how much of the original file we truly "consumed"
                    # based on how many windows we extracted.
                    # A safer approach for sliding windows across chunks:
                    # Seek back by (segment_duration * overlap) in original frames
                    overlap_frames = int(segment_duration * overlap * info.samplerate)
                    f.seek(max(0, f.tell() - overlap_frames))

        return results

    def _audio_to_mel_spectrogram(self, audio_segment):
        """Convert audio segment to mel spectrogram"""
        return self.audio_processor.audio_to_mel_spectrogram(audio_segment)

    def _process_segment_batch(
        self, segments, times, stable_id, stall_id, file_path
    ) -> list[MultiStallBatchResultReturn]:
        """Process a batch of audio segments (updated to use audio processor)"""
        # Convert segments to mel spectrograms using centralized method
        # mels = []
        processed_segments = []

        for segment in segments:
            # Ensure correct length and convert to mel spectrogram
            processed_segment = self.audio_processor._prepare_audio_segment(
                segment, int(self.audio_processor.sr * self.audio_processor.win_seconds)
            )
            # mel = self.audio_processor.audio_to_mel_spectrogram(processed_segment)

            processed_segments.append(processed_segment)
            # mels.append(mel[..., np.newaxis])

        # Use the same batch processing logic
        results = []
        self._process_batch(
            processed_segments, times, results, stable_id, stall_id, file_path
        )
        return results

    def _process_batch(
        self, audio_list, times_list, results, stable_id, stall_id, file_path
    ) -> MultiStallBatchResultReturn:
        """Process a batch of raw audio segments (works on both GPU/CPU)"""

        # Load configs
        # soundscape_config = self.model_registry.get_soundscape_classifier_config(
        #     stable_id, stall_id
        # )
        # REFACTOR: Use global config instead of per-stall config
        metamodel_config = load_metamodel_config()
        classifier_thresholds = metamodel_config.model_parameters.get(
            "classifier_thresholds", {}
        )
        silence_threshold = metamodel_config.model_parameters.get(
            "silence_threshold", -60.0
        )

        # New Anomaly detection using PANNs centroid
        clf = self.get_soundscape_classifier()

        # Load anomaly config
        anomaly_config = load_anomaly_config()
        anomaly_enabled = anomaly_config.enabled
        thr_window = anomaly_config.threshold

        logger.debug(
            f"Applied anomaly config: enabled={anomaly_enabled}, threshold={thr_window}"
        )

        # Pre-compute embeddings for the batch to save time (avoid double inference)
        # Note: audio_list contains raw audio segments; convert to numpy array first
        audio_batch = np.stack(audio_list)
        embeddings = clf.get_embeddings(audio_batch)

        # audio_list contains raw audio segments. score_anomaly handles list input.
        if anomaly_enabled:
            errs = clf.score_anomaly(embeddings=embeddings)
        else:
            errs = np.zeros(len(audio_list))

        # Soundscape classification (request labeled probabilities)
        probs = clf.infer_audio_segment(
            embeddings=embeddings, return_probabilities=True, file_path=file_path
        )

        # Clear GPU cache after PANNs inference
        if self.has_gpu:
            torch.cuda.empty_cache()

        # Try to get label mapping from multiple sources
        idx_to_label = {}
        if hasattr(clf, "idx_to_label") and clf.idx_to_label:
            idx_to_label = clf.idx_to_label
        if not idx_to_label:
            raise RuntimeError(
                "Model has no label mapping loaded. Cannot proceed with inference safely."
            )

        from smartstablemodel.labels import get_label_remapping

        label_remapping = get_label_remapping()
        background_label = metamodel_config.model_parameters.get(
            "background_label", "background"
        )
        ignore_labels = set(
            metamodel_config.model_parameters.get(
                "ignore_labels", ["unsure", "anomaly_normal", "anomaly_abnormal", "silence"]
            )
        )

        # Get audio features for metamodel analysis.
        loudness_values = [extract_loudness_features(a) for a in audio_list]
        spectral_features = [extract_spectral_features(a, sample_rate=SR) for a in audio_list]

        # Collect results
        for (start, end), score, prob, loudness, spec in zip(
            times_list, errs, probs, loudness_values, spectral_features
        ):
            spectral_centroid, high_freq_ratio = spec
            # Expect dict[label] -> prob; build a dict if array-like
            if isinstance(prob, dict):
                prob_dict = cast(
                    LabelProbabilities, {str(k): float(v) for k, v in prob.items()}
                )
            else:
                arr = np.asarray(prob).ravel().astype(float)
                if idx_to_label:
                    labels = [
                        idx_to_label.get(i, f"class_{i}") for i in range(len(arr))
                    ]
                else:
                    labels = [f"class_{i}" for i in range(len(arr))]

                prob_dict = cast(
                    LabelProbabilities,
                    {labels[i]: float(arr[i]) for i in range(len(arr))},
                )

            # Anomaly detection
            anomaly_score = float(score)
            is_anomaly = anomaly_score > thr_window

            # SILENCE HANDLING
            if loudness < silence_threshold:
                # If too quiet, force "silence" label and remove anomaly
                prob_dict = cast(LabelProbabilities, {"silence": 1.0})
                anomaly_score = 0.0
                is_anomaly = False
                logger.debug(
                    f"Segment {start:.2f}-{end:.2f}s: Silence detected (loudness {loudness:.3f})"
                )

            # Check if it is an anomaly
            # (we currently do not train on vocalizations so we need to check how
            # probable the anomaly is a "normal" vocalization)
            if is_anomaly:
                # Get the probabilities of the "normal" vocalizations
                # only compare to the most likely normal vocalization, warning! scream is similar to neigh!
                # so we need to choose threshold carefully!
                normal_vocalization_prob = max(
                    prob_dict.get(label, 0.0) for label in OTHER_VOCALIZATIONS
                )
                time_start = f"{start // 60}:{start % 60}"
                time_end = f"{end // 60}:{end % 60}"
                if (
                    normal_vocalization_prob
                    > anomaly_config.normal_vocalization_threshold
                ):
                    # get time start and end in min and seconds
                    best_normal_label = max(prob_dict.items(), key=lambda kv: kv[1])[0]
                    logger.info(
                        f"Anomaly detected (score {anomaly_score:.2f}) but likely normal vocalization of label {best_normal_label} (prob {normal_vocalization_prob:.2f}) (at {time_start}-{time_end})"
                    )
                    anomaly_score = 0.0  # Not sure if we should do this?
                    is_anomaly = False
                else:
                    best_normal_label = max(prob_dict.items(), key=lambda kv: kv[1])[0]
                    logger.warning(
                        f"Anomaly detected at {time_start}-{time_end} (score {anomaly_score:.2f}) and likely NOT a normal vocalization of label {best_normal_label} (prob {normal_vocalization_prob:.2f})"
                    )

            # Top-1 prediction
            predicted_label_str, predicted_prob = max(
                prob_dict.items(), key=lambda kv: kv[1]
            )

            # Apply remapping to top prediction
            predicted_label_str = label_remapping.get(
                predicted_label_str, predicted_label_str
            )

            # Filter out ignored labels
            if predicted_label_str in ignore_labels:
                filtered_probs = {}
                for k, v in prob_dict.items():
                    remapped_k = label_remapping.get(k, k)
                    if remapped_k not in ignore_labels:
                        filtered_probs[remapped_k] = filtered_probs.get(
                            remapped_k, 0.0
                        ) + float(v)

                if filtered_probs:
                    predicted_label_str, predicted_prob = max(
                        filtered_probs.items(), key=lambda kv: kv[1]
                    )
                else:
                    # No non-ignored labels left; default to background label
                    predicted_label_str, predicted_prob = background_label, 1.0

            # Log the prediction
            logger.debug(
                f"Segment {start:.2f}-{end:.2f}s: Predicted {predicted_label_str} ({predicted_prob:.2f}), Anomaly score: {score:.2f}"
            )

            # Apply per-label acceptance threshold; below threshold -> mark as 'uncertain'
            # label_thr = soundscape_config.get_threshold(predicted_label_str)

            # NEW LOGIC: Single Source of Truth
            # 1. Get architecture default
            model_arch = getattr(clf, "model_architecture", "panns")
            default_thr = float(classifier_thresholds.get(model_arch, 0.9))

            label_thr = default_thr

            if predicted_prob >= label_thr:
                accepted_label = predicted_label_str
                accepted_prob = predicted_prob
            else:
                accepted_label = "uncertain"
                # If predicted prob is below threshold
                # the certainty of uncertain is 1.0 - predicted_prob
                accepted_prob = 1.0 - predicted_prob

            # Never output 'unsure' as a label; collapse to uncertain
            if accepted_label == "unsure":
                accepted_label = "uncertain"

            results.append(
                MultiStallBatchResultReturn(
                    start_time=start,
                    end_time=end,
                    is_anomaly=is_anomaly,
                    anomaly_score=float(anomaly_score),
                    predicted_label=accepted_label,
                    probability=float(accepted_prob),
                    label_probabilities=prob_dict,
                    loudness=loudness,
                    spectral_centroid=spectral_centroid,
                    high_freq_ratio=high_freq_ratio,
                )
            )
        return results

    # ################## METAMODEL SECTION: ###############
    # def get_metamodel(self, stable_id: str) -> MetaModel:
    #     """Get or create metamodel for specific stall"""
    #     key = f"{stable_id}"

    #     if key not in self.metamodels:
    #         self.metamodels[key] = MetaModel(stable_id)

    #     return self.metamodels[key]
