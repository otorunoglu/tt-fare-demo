import tensorflow as tf
import logging
from pathlib import Path
from typing import List, Optional
from ml_audio_core.audio_processing import AudioProcessor
from ml_audio_core.ssl.data import SSLDataGenerator
from ml_audio_core.ssl.models import SimCLR, create_default_encoder

logger = logging.getLogger("ml_audio_core.ssl.trainer")


class SSLTrainer:
    """
    Trainer for Self-Supervised Learning models.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    def train(
        self,
        file_paths: List[str],
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        input_shape: tuple = (64, 96, 1),
        encoder: Optional[tf.keras.Model] = None,
    ):
        """
        Train a SimCLR model.

        Args:
            file_paths: List of audio files to train on.
            epochs: Number of epochs.
            batch_size: Batch size.
            learning_rate: Learning rate.
            input_shape: Input shape for the encoder (n_mels, t_bins, channels).
            encoder: Optional custom encoder model. If None, uses default CNN.
        """

        # 1. Setup Data
        # Initialize AudioProcessor with settings matching input_shape roughly
        # n_mels = input_shape[0], fixed_tbins = input_shape[1]
        ap = AudioProcessor(n_mels=input_shape[0], fixed_tbins=input_shape[1])

        data_gen = SSLDataGenerator(
            file_paths, ap, batch_size=batch_size, duration=ap.win_seconds
        )

        # 2. Setup Model
        if encoder is None:
            logger.info("Creating default encoder...")
            encoder = create_default_encoder(input_shape=input_shape)

        self.model = SimCLR(base_encoder=encoder, temperature=0.1)

        # 3. Compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer)

        # 4. Train
        logger.info(f"Starting SSL training for {epochs} epochs...")
        history = self.model.fit(
            data_gen,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(self.output_dir / "simclr_checkpoint.weights.h5"),
                    save_best_only=False,  # No val set usually in pure SSL
                    save_weights_only=True,
                )
            ],
        )

        # 5. Save Encoder (Backbone) separately
        # We only want the encoder for downstream tasks
        encoder_path = self.output_dir / "ssl_encoder.keras"
        self.model.base_encoder.save(encoder_path)
        logger.info(f"Saved trained encoder to {encoder_path}")

        return history
