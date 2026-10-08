"""
User actor receiving alerts from the monitoring pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .warning import DomainWarning


class AlertSink(Protocol):
    """Protocol to allow plugging in notification backends."""

    def send(self, warning: DomainWarning) -> None:  # pragma: no cover - protocol definition
        ...


@dataclass(slots=True)
class User:
    """
    Representation of a user who is notified about warnings.
    """

    name: str
    alert_sink: AlertSink

    def receive_alert(self, warning: DomainWarning) -> None:
        """
        Forward a warning to the configured alert sink.
        """

        self.alert_sink.send(warning)
