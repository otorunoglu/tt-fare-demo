import unittest
import tempfile
import soundfile as sf
import numpy as np
import shutil
from pathlib import Path
import os
import tensorflow as tf
from ml_audio_core.ssl.trainer import SSLTrainer
from ml_audio_core.ssl.models import SimCLR, create_default_encoder
from ml_audio_core.ssl.augmentations import apply_ssl_augmentations


class TestSSL(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.audio_dir = self.tmp_dir / "audio"
        self.audio_dir.mkdir()
        self.output_dir = self.tmp_dir / "output"
        self.output_dir.mkdir()

        # Create dummy wav files
        self.sr = 16000
        self.dummy_files = []
        for i in range(5):
            path = self.audio_dir / f"test_{i}.wav"
            # Random noise matching 3 seconds
            audio = np.random.uniform(-0.5, 0.5, self.sr * 3).astype(np.float32)
            sf.write(path, audio, self.sr)
            self.dummy_files.append(str(path))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_augmentations(self):
        """Verify augmentations return correct shape and type."""
        audio = np.random.uniform(-1, 1, 16000 * 2)
        aug = apply_ssl_augmentations(audio, 16000, 16000 * 2)
        self.assertEqual(aug.shape, (16000 * 2,))
        self.assertIsInstance(aug, np.ndarray)

    def test_default_encoder(self):
        """Verify default encoder creation."""
        input_shape = (64, 96, 1)
        model = create_default_encoder(input_shape)
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.output_shape, (None, 128))

    def test_trainer_integration(self):
        """Test the full training loop with SSLTrainer."""
        trainer = SSLTrainer(output_dir=str(self.output_dir))

        # Run 1 epoch
        history = trainer.train(
            self.dummy_files,
            epochs=1,
            batch_size=2,  # Small batch for small data
            input_shape=(64, 96, 1),
        )

        self.assertIn("loss", history.history)
        self.assertTrue((self.output_dir / "ssl_encoder.keras").exists())
        self.assertTrue((self.output_dir / "simclr_checkpoint.weights.h5").exists())


if __name__ == "__main__":
    unittest.main()
