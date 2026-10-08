"""
Warning domain model emitted by the meta-model decider.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict


class WarningType(str, Enum):
    """Severity categories for generated warnings."""

    ALERT = "alert"
    CLUSTER = "cluster"
    LOUDNESS = "loudness"
    INFORMATIONAL = "informational"


@dataclass(slots=True)
class DomainWarning:
    """
    Alerting payload that can be handed off to user-facing systems.
    """

    label: str
    timestamp: datetime
    confidence: float
    warning_type: WarningType
    severity: float

    def as_dict(self) -> Dict[str, object]:
        """
        Serialize the warning into primitives for messaging or storage.
        """

        return {
            "label": self.label,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "warning_type": self.warning_type.value,
            "severity": self.severity,
        }


