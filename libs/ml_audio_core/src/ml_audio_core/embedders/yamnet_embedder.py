import logging
import numpy as np
import tensorflow_hub as hub

try:
    import tensorflow as tf
except ImportError:
    tf = None

from ml_audio_core.embedders.base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class YamNetEmbedder(BaseEmbedder):
    """
    Wrapper around YAMNet for embedding extraction.
    Loads the YAMNet model from TensorFlow Hub using tensorflow_hub library.
    """

    def __init__(self, model_dir: str = None):
        """
        Args:
            model_dir: Not strictly needed for hub.load as it handles caching,
                       but kept for interface compatibility.
                       TensorFlow Hub caches models in TFHUB_CACHE_DIR (env var).
        """
        if tf is None:
            raise RuntimeError("TensorFlow is required for YamNetEmbedder.")

        logger.info("Loading YAMNet model from TensorFlow Hub...")
        # Load the model from TFHub.
        # 'https://tfhub.dev/google/yamnet/1' is the handle.
        # hub.load downloads and caches it automatically.
        try:
            self.model = hub.load("https://tfhub.dev/google/yamnet/1")
            logger.info("YAMNet model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YAMNet from TFHub: {e}")
            raise

    @property
    def embedding_dim(self) -> int:
        return 1024  # YAMNet embedding dimension

    def embed(self, audio_batch: np.ndarray) -> np.ndarray:
        """
        Embed a batch of audio.
        YAMNet expects 16kHz audio.
        Input: 1D float32 tensor or batch of waveforms?
        TFHub YAMNet signature:
          Input: 'waveform' (1-D tensor of float32)

        We need to loop over the batch because the standard SavedModel usually accepts 1D input.
        """
        embeddings_list = []

        for waveform in audio_batch:
            # Ensure float32
            waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)

            # Run inference
            # model(waveform) returns (scores, embeddings, log_mel_spectrogram)
            scores, embeddings, log_mel_spectrogram = self.model(waveform_tf)

            # embeddings shape: (N_frames, 1024)
            # We average over frames to get a single vector per segment
            avg_emb = tf.reduce_mean(embeddings, axis=0)
            embeddings_list.append(avg_emb)

        return np.stack(embeddings_list)
