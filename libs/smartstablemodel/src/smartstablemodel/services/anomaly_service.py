import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger("smartstablemodel.services.anomaly")


class AnomalyService:
    def __init__(self):
        self.detector = None

    def train_detector(
        self,
        embeddings: np.ndarray,
        n_estimators: int = 100,
        contamination: float = 0.01,
        random_state: int = 42,
    ):
        """
        Train an Isolation Forest on the provided embeddings.

        Args:
            embeddings: Numpy array of shape (n_samples, n_features).
            n_estimators: Number of trees in the forest.
            contamination: Expected proportion of outliers in the data.
            random_state: Seed for reproducibility.
        """
        logger.info(f"Training Isolation Forest on {len(embeddings)} samples...")
        self.detector = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,  # Use all cores
        )
        self.detector.fit(embeddings)
        logger.info("Training complete.")
        return self.detector

    def save_detector(self, path: str):
        """Save the trained detector to a pickle file."""
        if self.detector is None:
            raise ValueError("No detector to save.")

        with open(path, "wb") as f:
            pickle.dump(self.detector, f)
        logger.info(f"Detector saved to {path}")

    def load_detector(self, path: str):
        """Load a detector from a pickle file."""
        with open(path, "rb") as f:
            self.detector = pickle.load(f)
        logger.info(f"Detector loaded from {path}")
        return self.detector

    def predict_anomaly_score(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.
        Lower scores = more anomalous.
        """
        if self.detector is None:
            raise ValueError("Detector not loaded.")
        return self.detector.decision_function(embeddings)

    def detect_anomalies(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Detect anomalies.
        Returns -1 for outliers and 1 for inliers.
        """
        if self.detector is None:
            raise ValueError("Detector not loaded.")
        return self.detector.predict(embeddings)
