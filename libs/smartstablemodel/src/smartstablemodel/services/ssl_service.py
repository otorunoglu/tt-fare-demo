import logging
import tensorflow as tf
from pathlib import Path
from typing import List, Optional, Union
import numpy as np

from ml_audio_core.ssl.trainer import SSLTrainer
from ml_audio_core.audio_processing import AudioProcessor

logger = logging.getLogger("smartstablemodel.services.ssl")


class SSLService:
    def __init__(self):
        self.model = None
        # Default processor, will be re-initialized if model is loaded with specific shape
        self.audio_processor = AudioProcessor()

    def train_model(
        self,
        data_dir: str,
        output_dir: str,
        epochs: int = 10,
        batch_size: int = 32,
        input_shape: tuple = (64, 96, 1),
    ) -> str:
        """
        Train the SSL backbone on audio files found in data_dir.

        Args:
            data_dir: Directory containing audio files.
            output_dir: Directory to save model and checkpoints.
            epochs: Number of training epochs.
            batch_size: Batch size.
            input_shape: Input shape (n_mels, time_bins, channels).

        Returns:
            Path to the saved encoder model.
        """
        data_path = Path(data_dir)
        output_path = Path(output_dir)

        # Find all audio files
        extensions = ["*.wav", "*.flac", "*.mp3", "*.ogg"]
        file_paths = []
        for ext in extensions:
            file_paths.extend([str(p) for p in data_path.rglob(ext)])

        if not file_paths:
            raise ValueError(
                f"No audio files found in {data_dir} with extensions {extensions}"
            )

        logger.info(f"Found {len(file_paths)} audio files for training.")

        trainer = SSLTrainer(output_dir=str(output_path))
        trainer.train(
            file_paths=file_paths,
            epochs=epochs,
            batch_size=batch_size,
            input_shape=input_shape,
        )

        encoder_path = output_path / "ssl_encoder.keras"
        if not encoder_path.exists():
            raise RuntimeError(f"Expected model file missing at {encoder_path}")

        logger.info(f"Training complete. Model saved to {encoder_path}")
        return str(encoder_path)

    def load_model(self, model_path: str):
        """Load a saved Keras encoder."""
        logger.info(f"Loading model from {model_path}")
        self.model = tf.keras.models.load_model(model_path)

        # Re-configure AudioProcessor based on model input shape
        # Model input: (None, n_mels, t_bins, channels)
        input_shape = self.model.input_shape
        if input_shape and len(input_shape) == 4:
            n_mels = input_shape[1]
            fixed_tbins = input_shape[2]
            logger.info(
                f"Configuring AudioProcessor for model params: n_mels={n_mels}, fixed_tbins={fixed_tbins}"
            )
            self.audio_processor = AudioProcessor(
                n_mels=n_mels, fixed_tbins=fixed_tbins
            )

        return self.model

    def compute_embeddings(
        self, audio_file: str, model_path: str = None, overlap: float = 0.0
    ) -> np.ndarray:
        """
        Compute embeddings for an audio file.
        Returns array of shape (num_segments, embedding_dim).
        """
        if self.model is None:
            if model_path:
                self.load_model(model_path)
            else:
                raise ValueError("Model not loaded and no model_path provided")

        # Ensure processor matches model
        input_shape = self.model.input_shape
        target_tbins = input_shape[2] if input_shape and len(input_shape) == 4 else 96

        # Load and segment audio
        segments = self.audio_processor.load_varying_length_audio(
            audio_file,
            duration=self.audio_processor.win_seconds,
            overlap=overlap,
            include_metadata=False,
        )

        if not segments:
            logger.warning(f"No segments generated for {audio_file}")
            return np.array([])

        # Convert to Mel Spectrograms
        mels = [
            self.audio_processor.audio_to_mel_spectrogram(s, target_length=target_tbins)
            for s in segments
        ]

        # Batch and predict
        batch = self.audio_processor.prepare_mel_batch(mels)
        embeddings = self.model.predict(batch, verbose=0)

        return embeddings
