import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Adjust path to find modules
sys.path.append("src")
sys.path.append("src/smartstablemodel")

from smartstablemodel.multi_stall_manager import MultiStallDetectorManager
from smartstablemodel.generated_types import LabelProbabilities


class TestThresholdLogic(unittest.TestCase):
    def setUp(self):
        # Mock dependencies to avoid loading real models
        with patch(
            "smartstablemodel.multi_stall_manager.SoundscapeClassifier"
        ) as MockClassifier:
            with patch(
                "smartstablemodel.multi_stall_manager.AudioProcessor"
            ) as MockAudio:
                self.manager = MultiStallDetectorManager()
                self.mock_clf = self.manager.soundscape_classifier

                # Mock label registry
                self.mock_registry = MagicMock()
                self.mock_clf.label_registry = self.mock_registry

    def test_panns_default_threshold(self):
        """Test PANNS uses 0.9 default"""
        self.mock_clf.model_architecture = "panns"
        # Mock label registry to return None for 'neigh' (no override)
        self.mock_registry.get_label.return_value = None

        # Mock inference result: neigh=0.85 (should be rejected)
        self.mock_clf.infer_audio_segment.return_value = [
            {"neigh": 0.85, "normal": 0.15}
        ]

        # Call internal batch processing
        # We need to mock _process_batch inputs
        # But _process_batch is what we want to test.
        # It's easier to mock the helper methods it uses, OR just call it directly if we mock inputs.

        # Mock inputs
        audio_list = [np.zeros(100)]
        times_list = [(0, 2)]
        results = []

        # Mock score_anomaly
        self.mock_clf.score_anomaly.return_value = np.zeros(1)

        # Run
        # We need to patch load_metamodel_config to ensure consistent 0.9/0.6 values
        with patch(
            "smartstablemodel.multi_stall_manager.load_metamodel_config"
        ) as mock_load:
            mock_load.return_value.model_parameters = {
                "classifier_thresholds": {"panns": 0.9, "yamnet": 0.6}
            }

            self.manager._process_batch(
                audio_list, times_list, results, "stable", "stall", "file"
            )

        # Check result
        self.assertEqual(len(results), 1)
        # Should be normal because 0.85 < 0.9
        self.assertEqual(results[0].predicted_label, "normal")
        self.assertEqual(results[0].probability, 0.85)

    def test_yamnet_default_threshold(self):
        """Test YAMNet uses 0.6 default"""
        self.mock_clf.model_architecture = "yamnet"
        self.mock_registry.get_label.return_value = None

        # Mock inference result: neigh=0.7 (should be accepted)
        self.mock_clf.infer_audio_segment.return_value = [{"neigh": 0.7, "normal": 0.3}]

        # Mock inputs
        audio_list = [np.zeros(100)]
        times_list = [(0, 2)]
        results = []
        self.mock_clf.score_anomaly.return_value = np.zeros(1)

        with patch(
            "smartstablemodel.multi_stall_manager.load_metamodel_config"
        ) as mock_load:
            mock_load.return_value.model_parameters = {
                "classifier_thresholds": {"panns": 0.9, "yamnet": 0.6}
            }
            self.manager._process_batch(
                audio_list, times_list, results, "stable", "stall", "file"
            )

        self.assertEqual(results[0].predicted_label, "neigh")


if __name__ == "__main__":
    unittest.main()
