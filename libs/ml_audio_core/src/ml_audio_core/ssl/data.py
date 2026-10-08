import numpy as np
import tensorflow as tf
from typing import List, Generator, Tuple
import logging
from ml_audio_core.audio_processing import AudioProcessor
from ml_audio_core.ssl.augmentations import apply_ssl_augmentations

logger = logging.getLogger("ml_audio_core.ssl.data")


class SSLDataGenerator(tf.keras.utils.Sequence):
    """
    Generates data for Self-Supervised Learning (e.g. SimCLR).
    Input: List of audio file paths.
    Output: Batches of (view1, view2) pairs.
    """

    def __init__(
        self,
        file_paths: List[str],
        audio_processor: AudioProcessor,
        batch_size: int = 32,
        duration: float = 2.0,
        shuffle: bool = True,
    ):
        """
        Args:
            file_paths: List of absolute paths to audio files.
            audio_processor: Instance of AudioProcessor for loading/resampling.
            batch_size: Number of pairs per batch.
            duration: Duration of each audio segment in seconds.
            shuffle: Whether to shuffle file order each epoch.
        """
        self.file_paths = file_paths
        self.audio_processor = audio_processor
        self.batch_size = batch_size
        self.duration = duration
        self.shuffle = shuffle

        self.sample_rate = self.audio_processor.sr
        self.target_length = int(self.sample_rate * self.duration)

        # Pre-calculate or lazy-load segments?
        # For large datasets, we should probably stream or index.
        # For now, let's load all segments into memory (assuming moderate dataset size for now).
        # Optimization: Move to lazy loading if memory becomes an issue.
        self.segments = self._load_all_segments()

        # Indices for batching
        self.indices = np.arange(len(self.segments))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _load_all_segments(self) -> List[np.ndarray]:
        """Load and segment all audio files."""
        all_segments = []
        logger.info(f"Loading {len(self.file_paths)} files for SSL training...")

        for fp in self.file_paths:
            try:
                # Load file with AudioProcessor
                # We use overlap=0.0 to maximize coverage without too much redundancy for SSL
                # But typical SSL often uses overlap. Let's use 0.5 default from AudioProcessor if appropriate.
                # Actually, standardizing on a dedicated overlap for SSL might be better.
                # Let's use overlap=0.0 for now to get distinct samples.
                segments = self.audio_processor.load_varying_length_audio(
                    fp, duration=self.duration, overlap=0.0, include_metadata=False
                )
                all_segments.extend(segments)
            except Exception as e:
                logger.warning(f"Failed to load {fp}: {e}")

        logger.info(f"Loaded {len(all_segments)} total audio segments.")
        return all_segments

    def __len__(self):
        """Number of batches per epoch."""
        return int(np.floor(len(self.segments) / self.batch_size))

    def on_epoch_end(self):
        """Shuffle indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        """Generate one batch of data."""
        batch_indices = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        # Get raw segments for this batch
        batch_segments = [self.segments[i] for i in batch_indices]

        # Create two augmented views for each segment
        view1_batch = []
        view2_batch = []

        for seg in batch_segments:
            # Augment view 1
            v1_audio = apply_ssl_augmentations(
                seg, self.sample_rate, self.target_length
            )
            # Convert to Mel Spectrogram (using AudioProcessor logic but need to handle simple array)
            # Note: AudioProcessor.audio_to_mel_spectrogram expects raw audio
            v1_mel = self.audio_processor.audio_to_mel_spectrogram(
                v1_audio, target_length=self.target_length
            )

            # Augment view 2
            v2_audio = apply_ssl_augmentations(
                seg, self.sample_rate, self.target_length
            )
            v2_mel = self.audio_processor.audio_to_mel_spectrogram(
                v2_audio, target_length=self.target_length
            )

            view1_batch.append(v1_mel)
            view2_batch.append(v2_mel)

        # Stack into arrays (batch, n_mels, t_bins, 1)
        # Using prepare_mel_batch ensures consistent shaping
        X1 = self.audio_processor.prepare_mel_batch(view1_batch)
        X2 = self.audio_processor.prepare_mel_batch(view2_batch)

        # SimCLR training usually expects inputs as a tuple (prob. handled by custom training loop)
        # But for standard Keras fit(), X should be inputs, y should be targets.
        # However, SimCLR loss is calculated based on embeddings, not ground truth labels.
        # So we usually pass dummy targets or handle loss in a custom Model.train_step.

        return (X1, X2), np.zeros((self.batch_size,))  # Dummy targets
