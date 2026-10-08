# ml_audio_core/embedders/base_embedder.py

from abc import ABC, abstractmethod
import numpy as np

class BaseEmbedder(ABC):
    """
    Abstract base class for any audio embedding extractor.

    Implementations must define:
    - embedding_dim property
    - embed(audio_batch)
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass

    @abstractmethod
    def embed(self, audio_batch: np.ndarray) -> np.ndarray:
        """
        Takes a batch of raw audio (shape: N x samples)
        Returns embeddings (shape: N x embedding_dim)
        """
        pass