import numpy as np
import logging

logger = logging.getLogger("ml_audio_core.ssl.augmentations")


def random_crop(audio: np.ndarray, crop_size: int) -> np.ndarray:
    """Randomly crop the audio to the specified size."""
    if len(audio) <= crop_size:
        return audio

    start = np.random.randint(0, len(audio) - crop_size)
    return audio[start : start + crop_size]


def add_gaussian_noise(
    audio: np.ndarray, min_amplitude: float = 0.001, max_amplitude: float = 0.015
) -> np.ndarray:
    """Add random Gaussian noise to the audio."""
    noise_level = np.random.uniform(min_amplitude, max_amplitude)
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise


def random_gain(
    audio: np.ndarray, min_gain: float = 0.8, max_gain: float = 1.2
) -> np.ndarray:
    """Apply random gain (volume change)."""
    gain = np.random.uniform(min_gain, max_gain)
    return audio * gain


def polarity_inversion(audio: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Invert the polarity of the audio with probability p."""
    if np.random.random() < p:
        return -audio
    return audio


def time_mask(
    audio: np.ndarray, max_mask_pct: float = 0.1, n_masks: int = 1
) -> np.ndarray:
    """Apply time masking (zero out random chunks)."""
    L = len(audio)
    masked_audio = audio.copy()

    for _ in range(n_masks):
        mask_len = np.random.randint(0, int(L * max_mask_pct))
        mask_start = np.random.randint(0, L - mask_len)
        masked_audio[mask_start : mask_start + mask_len] = 0

    return masked_audio


def apply_ssl_augmentations(
    audio: np.ndarray, sample_rate: int, target_length: int
) -> np.ndarray:
    """
    Apply a stochastic pipeline of augmentations suitable for SSL (e.g. SimCLR).

    Args:
        audio: Input audio array
        sample_rate: Sample rate of the audio
        target_length: Desired output length in samples

    Returns:
        Augmented audio array of shape (target_length,)
    """
    # 1. Random Crop (if longer than target) or Pad (if shorter)
    if len(audio) > target_length:
        audio = random_crop(audio, target_length)
    elif len(audio) < target_length:
        pad_len = target_length - len(audio)
        audio = np.pad(audio, (0, pad_len))

    # 2. Polarity Inversion (50% chance)
    audio = polarity_inversion(audio, p=0.5)

    # 3. Add Gaussian Noise (randomly)
    if np.random.random() < 0.5:
        audio = add_gaussian_noise(audio)

    # 4. Random Gain
    audio = random_gain(audio)

    # 5. Time Masking (occasionally)
    if np.random.random() < 0.3:
        audio = time_mask(audio)

    return audio.astype(np.float32)
