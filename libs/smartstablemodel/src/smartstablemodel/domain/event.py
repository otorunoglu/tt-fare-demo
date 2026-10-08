"""
Event domain model used across the monitoring pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


@dataclass(slots=True)
class Event:
    """
    Normalized representation of a single soundscape classification event.
    """

    timestamp: datetime
    label: str
    confidence: float
    loudness: float
    spectral_centroid: Optional[float] = None
    high_freq_ratio: Optional[float] = None
    original_label: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        """
        Serialize the event into primitives for logging or storage.
        """

        return {
            "timestamp": self.timestamp.isoformat(),
            "label": self.label,
            "confidence": self.confidence,
            "loudness": self.loudness,
            "spectral_centroid": self.spectral_centroid,
            "high_freq_ratio": self.high_freq_ratio,
            "original_label": self.original_label,
            "metadata": dict(self.metadata),
        }
