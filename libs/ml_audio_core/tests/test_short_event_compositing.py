import unittest
import numpy as np
import sys
import types


if "audio_preprocessing" not in sys.modules:
    sys.modules["audio_preprocessing"] = types.SimpleNamespace(
        load_audio=lambda *_args, **_kwargs: (np.zeros(1, dtype=np.float32), 16000),
        resample_numpy=lambda audio, **_kwargs: audio,
    )

from ml_audio_core.audio_processing import AudioProcessor


class TestShortEventCompositing(unittest.TestCase):
    def setUp(self):
        self.processor = AudioProcessor(sr=100, win_seconds=2.0, win_overlap=0.5)

    def test_compose_returns_target_shape_and_dtype(self):
        short_event = np.ones(40, dtype=np.float32)
        background_pool = [np.zeros(200, dtype=np.float32)]

        out = self.processor.compose_short_event_with_background(
            short_event=short_event,
            background_pool=background_pool,
            target_duration_seconds=2.0,
            crossfade_ms=0,
            position_mode="center",
            rng=np.random.default_rng(7),
        )

        self.assertEqual(out.shape[0], 200)
        self.assertEqual(out.dtype, np.float32)

    def test_center_mode_places_event_in_middle_without_crossfade(self):
        short_event = np.ones(40, dtype=np.float32)
        background_pool = [np.zeros(200, dtype=np.float32)]

        out = self.processor.compose_short_event_with_background(
            short_event=short_event,
            background_pool=background_pool,
            target_duration_seconds=2.0,
            crossfade_ms=0,
            position_mode="center",
            rng=np.random.default_rng(1),
        )

        start = (200 - 40) // 2
        end = start + 40

        self.assertTrue(np.allclose(out[:start], 0.0))
        self.assertTrue(np.allclose(out[start:end], 1.0))
        self.assertTrue(np.allclose(out[end:], 0.0))

    def test_random_mode_is_deterministic_with_seed(self):
        short_event = np.linspace(-0.5, 0.5, 30, dtype=np.float32)
        background_pool = [np.zeros(200, dtype=np.float32), np.ones(200, dtype=np.float32) * 0.1]

        out_1 = self.processor.compose_short_event_with_background(
            short_event=short_event,
            background_pool=background_pool,
            target_duration_seconds=2.0,
            crossfade_ms=20,
            position_mode="random",
            rng=np.random.default_rng(1234),
        )
        out_2 = self.processor.compose_short_event_with_background(
            short_event=short_event,
            background_pool=background_pool,
            target_duration_seconds=2.0,
            crossfade_ms=20,
            position_mode="random",
            rng=np.random.default_rng(1234),
        )

        self.assertTrue(np.allclose(out_1, out_2))

    def test_crossfade_smooths_boundaries(self):
        short_event = np.ones(40, dtype=np.float32)
        background_pool = [np.zeros(200, dtype=np.float32)]

        out = self.processor.compose_short_event_with_background(
            short_event=short_event,
            background_pool=background_pool,
            target_duration_seconds=2.0,
            crossfade_ms=100,
            position_mode="center",
            rng=np.random.default_rng(1),
        )

        start = (200 - 40) // 2
        fade_samples = 10  # 100 ms at sr=100

        fade_in = out[start : start + fade_samples]
        fade_out = out[start + 40 - fade_samples : start + 40]

        self.assertAlmostEqual(float(fade_in[0]), 0.0, places=5)
        self.assertAlmostEqual(float(fade_in[-1]), 1.0, places=5)
        self.assertTrue(np.all(np.diff(fade_in) >= -1e-6))

        self.assertAlmostEqual(float(fade_out[0]), 1.0, places=5)
        self.assertAlmostEqual(float(fade_out[-1]), 0.0, places=5)
        self.assertTrue(np.all(np.diff(fade_out) <= 1e-6))


if __name__ == "__main__":
    unittest.main()
