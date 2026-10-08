"""
Audio recorder component responsible for ingesting raw soundscape data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..domain import Event
from .data_store import DataStore


class Recorder:
    """
    High-level service orchestrating ingestion, enrichment, and storage of events.
    """

    def __init__(self, data_store: DataStore) -> None:
        self._data_store = data_store

    def record_audio(self, duration_seconds: float) -> Path:
        """
        Capture raw audio for the requested duration and return the file path.
        """

        raise NotImplementedError("record_audio must interface with the recorder hardware")

    def compute_loudness(self, audio_path: Path) -> float:
        """
        Compute loudness from a recorded audio file.
        """

        raise NotImplementedError("compute_loudness must analyse the audio signal")

    def classify_sound(self, audio_path: Path) -> Iterable[Event]:
        """
        Run the ML classifier and return a stream of events.
        """

        raise NotImplementedError("classify_sound must invoke the classifier pipeline")

    def save_to_store(self, events: Iterable[Event]) -> None:
        """
        Persist the generated events into the backing data store.
        """

        for event in events:
            self._data_store.append_event(event)
