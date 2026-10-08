import numpy as np
import logging

logger = logging.getLogger("ml_audio_core.augmentation")


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------


def _add_noise(segment: np.ndarray) -> np.ndarray:
    """Add Gaussian noise with a random amplitude."""
    noise_level = np.random.uniform(0.002, 0.04)
    noise = np.random.normal(0, noise_level, segment.shape)
    return segment + noise


def _time_shift(segment: np.ndarray) -> np.ndarray:
    """Shift audio forward/backward and zero-pad (no circular wrap)."""
    max_shift = min(2000, len(segment) // 8)
    if max_shift == 0:
        return segment
    shift = np.random.randint(-max_shift, max_shift)
    out = np.zeros_like(segment)
    if shift > 0:
        out[shift:] = segment[:-shift]
    elif shift < 0:
        out[:shift] = segment[-shift:]
    else:
        out = segment.copy()
    return out


def _gain(segment: np.ndarray) -> np.ndarray:
    """Random volume scaling."""
    factor = np.random.uniform(0.5, 1.5)
    return segment * factor


def _polarity_inversion(segment: np.ndarray) -> np.ndarray:
    """Flip the waveform polarity."""
    return -segment


def _time_mask(segment: np.ndarray) -> np.ndarray:
    """Zero out 1-2 random chunks (up to 15 % each)."""
    out = segment.copy()
    n_masks = np.random.randint(1, 3)
    for _ in range(n_masks):
        mask_len = np.random.randint(0, int(len(out) * 0.15))
        start = np.random.randint(0, max(1, len(out) - mask_len))
        out[start : start + mask_len] = 0
    return out


def _pitch_shift(segment: np.ndarray) -> np.ndarray:
    """Crude pitch shift via resampling: stretch/squeeze then crop/pad."""
    rate = np.random.uniform(0.9, 1.1)
    orig_len = len(segment)
    indices = np.arange(0, orig_len, rate)
    indices = indices[indices < orig_len - 1].astype(int)
    resampled = segment[indices]
    # Match original length
    if len(resampled) < orig_len:
        resampled = np.pad(resampled, (0, orig_len - len(resampled)))
    else:
        resampled = resampled[:orig_len]
    return resampled


# ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------

# Transforms that always apply (cheap, non-destructive)
_ALWAYS_POOL = [_gain]

# Transforms applied stochastically (each with its own probability)
_STOCHASTIC_POOL = [
    (_add_noise, 0.6),
    (_time_shift, 0.5),
    (_polarity_inversion, 0.3),
    (_time_mask, 0.4),
    (_pitch_shift, 0.3),
]


def _compose_augmentation(segment: np.ndarray) -> np.ndarray:
    """Apply a random *composition* of multiple transforms to one segment."""
    aug = segment.copy()

    # Always apply gain variation
    for fn in _ALWAYS_POOL:
        aug = fn(aug)

    # Stochastically layer additional transforms
    for fn, prob in _STOCHASTIC_POOL:
        if np.random.random() < prob:
            aug = fn(aug)

    return np.clip(aug, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Public API (unchanged signature)
# ---------------------------------------------------------------------------


def augment_audio_data(segments, augmentation_factor=2):
    """
    Enhanced audio augmentation for small datasets.

    Each augmented sample is created by *composing* multiple random transforms
    (noise, time-shift, gain, polarity flip, time-mask, pitch-shift) rather
    than applying a single one.  This produces genuinely diverse training
    examples even from very few originals.
    """
    if augmentation_factor <= 1:
        return segments

    augmented = []
    original_count = len(segments)
    target_count = original_count * augmentation_factor

    # Always include originals
    augmented.extend(segments)

    while len(augmented) < target_count:
        for segment in segments:
            if len(augmented) >= target_count:
                break
            augmented.append(_compose_augmentation(segment))

    logger.debug(f"Augmented {original_count} -> {len(augmented)} samples")
    return augmented[:target_count]
