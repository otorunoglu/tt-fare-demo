import numpy as np
import soundfile as sf
import logging
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path

logger = logging.getLogger("ml_audio_core.synthesizer")


class SoundscapeSynthesizer:
    """
    Mixes foreground sound events into a background audio track.
    Generates synthetic evaluation data for Sound Event Detection (SED).
    """

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def create_track(
        self,
        duration_seconds: float,
        foreground_events: List[Dict],
        background_audio: Optional[np.ndarray] = None,
        background_gain: float = 0.5,
        foreground_gain: float = 1.0,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Create a long audio track by mixing foreground events.

        Args:
            duration_seconds: Total duration of the track.
            foreground_events: List of events, each with:
                - 'audio': np.ndarray
                - 'label': str
                - 'start_time': float (seconds)
                - 'metadata': dict (optional, e.g. SampleID)
            background_audio: Optional background loop or noise.
            background_gain: Volume scaling for background.
            foreground_gain: Volume scaling for foreground events.
            normalize: If True, peak normalize the final track to 1.0 if it clips.

        Returns:
            Tuple of (mixed_audio, annotations)
        """
        total_samples = int(duration_seconds * self.sr)

        # Initialize canvas
        if background_audio is not None:
            # Loop background if shorter, or truncate if longer
            bg_samples = len(background_audio)
            num_repeats = (total_samples // bg_samples) + 1
            mixed = (
                np.tile(background_audio, num_repeats)[:total_samples].astype(
                    np.float32
                )
                * background_gain
            )
        else:
            mixed = np.zeros(total_samples, dtype=np.float32)

        annotations = []

        # Sort events by start time for consistent mixing/annotation
        foreground_events = sorted(foreground_events, key=lambda x: x["start_time"])

        for event in foreground_events:
            audio = event["audio"]
            label = event["label"]
            start_time = event["start_time"]

            start_idx = int(start_time * self.sr)
            end_idx = start_idx + len(audio)

            # Boundary check
            if start_idx >= total_samples:
                logger.warning(
                    f"Event {label} starts after track end ({start_time}s). Skipping."
                )
                continue

            if end_idx > total_samples:
                # Truncate audio to fit
                audio_mixed = audio[: total_samples - start_idx]
                end_idx = total_samples
            else:
                audio_mixed = audio

            # Mix (additive)
            mixed[start_idx:end_idx] += audio_mixed.astype(np.float32) * foreground_gain

            annotations.append(
                {
                    "label": label,
                    "start_time": float(start_time),
                    "end_time": float(end_idx / self.sr),
                    "metadata": event.get("metadata", {}),
                }
            )

        # Final peak normalization if it clips
        if normalize:
            max_val = np.max(np.abs(mixed))
            if max_val > 1.0:
                logger.info(f"Normalizing track (peak={max_val:.2f})")
                mixed = mixed / max_val

        return mixed, annotations

    def generate_random_soundscape(
        self,
        duration_seconds: float,
        sample_pool: Dict[str, List[Tuple[np.ndarray, str]]],
        events_per_minute: float = 10,
        background_audio: Optional[np.ndarray] = None,
        exclude_labels: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Randomly select events from a pool and distribute them along the timeline.
        Good for stress-testing evaluation.
        """
        num_events = int((duration_seconds / 60.0) * events_per_minute)
        labels = [
            l
            for l in sample_pool.keys()
            if not exclude_labels or l not in exclude_labels
        ]

        if not labels:
            raise ValueError("Sample pool is empty or all labels excluded")

        foreground_events = []
        for _ in range(num_events):
            label = np.random.choice(labels)
            pool = sample_pool[label]
            if not pool:
                continue

            # Pick a random sample from the pool
            idx = np.random.randint(len(pool))
            audio, sample_id = pool[idx]

            start_time = np.random.uniform(0, max(0, duration_seconds - 2.0))
            foreground_events.append(
                {
                    "audio": audio,
                    "label": label,
                    "start_time": start_time,
                    "metadata": {"sample_id": sample_id},
                }
            )

        return self.create_track(
            duration_seconds, foreground_events, background_audio=background_audio
        )

    def save_track(self, audio: np.ndarray, annotations: List[Dict], output_path: str):
        """Save WAV and JSON annotations."""
        output_path = Path(output_path)

        # Save WAV
        sf.write(str(output_path), audio, self.sr)

        # Save JSON
        annot_path = output_path.with_suffix(".json")
        with open(annot_path, "w") as f:
            json.dump(
                {
                    "sr": self.sr,
                    "duration": float(len(audio) / self.sr),
                    "annotations": annotations,
                },
                f,
                indent=2,
            )

        logger.info(
            f"Saved synthetic track to {output_path} and annotations to {annot_path}"
        )
