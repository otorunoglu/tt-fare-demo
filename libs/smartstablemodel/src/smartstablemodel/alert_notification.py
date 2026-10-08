"""Notification aggregation layer.

Converts a stream of already *filtered* alert records (output of AlertFilter)
into higher-level user-facing notifications. The aim is to avoid spamming the
end user with many mid‑level alerts (e.g. repeated `possible_burst` or
`kick_concern`) while still surfacing clearly critical events immediately.

Pipeline layering now becomes:

    raw segment event -> ContextualMetaModelEvaluator -> AlertFilter ->
    NotificationAggregator -> (user notification / UI)

Design principles:
  1. Immediate pass-through for high severity alerts (scream, whimmering,
     confirmed_burst) – these are rare and important.
  2. Episode grouping for mid/low severity alerts. Multiple similar alerts
     within a short time window become a *single* notification summarizing
     intensity (count, duration, max confidence).
  3. Rate limiting of mid-level notifications per hour to prevent fatigue.
  4. Escalation path: a single mid-level alert with *very* high confidence
     can still generate a notification sooner (high_confidence_escalation).
  5. Extensible severity mapping; adding a new alert type only requires
     editing SEVERITY_RANK / policy sets.

Assumptions:
  - Incoming records are dicts produced by AlertFilter.process():
        { 'timestamp': str, 'alert': str, 'confidence': float, ... }
  - Timestamp format matches one of the formats in _parse_ts().

"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Iterable


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class NotificationPolicy:
    # Alerts that should always generate a notification immediately.
    high_severity_immediate: set[str] = field(
        default_factory=lambda: {"critical_scream", "critical_whimmering", "confirmed_burst"}
    )
    # Max gap (silence between alerts of same type) to keep accumulating in an existing episode.
    episode_gap_seconds: int = 300  # 5 minutes
    # Window to consider for meeting the count-based escalation criterion.
    window_seconds: int = 600  # 10 minutes
    # Minimum number of mid-level alerts in the window to trigger an episode notification.
    min_mid_severity_count: int = 3
    # Confidence threshold to escalate a single mid-level alert immediately.
    high_confidence_escalation: float = 0.88
    # Rate limit: maximum emitted mid-level notifications per rolling hour.
    max_mid_notifications_per_hour: int = 4
    # Severity ranking (higher = worse). Modify to tune alert taxonomy.
    severity_rank: Dict[str, int] = field(
        default_factory=lambda: {
            "critical_scream": 3,
            "critical_whimmering": 3,
            "confirmed_burst": 2,
            "kick_concern": 1,
            "distress_concern": 1,
            "restlessness": 0,
            "possible_burst": 0,
        }
    )


# ---------------------------------------------------------------------------
# Internal models
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    alert_type: str
    first_time: datetime
    last_time: datetime
    count: int = 1
    max_confidence: float = 0.0
    notified: bool = False

    def update(self, ts: datetime, conf: float):
        self.last_time = ts
        self.count += 1
        if conf > self.max_confidence:
            self.max_confidence = conf

    def duration(self) -> float:
        return (self.last_time - self.first_time).total_seconds()


# ---------------------------------------------------------------------------
# Aggregator implementation
# ---------------------------------------------------------------------------


class NotificationAggregator:
    def __init__(self, policy: NotificationPolicy | None = None):
        self.policy = policy or NotificationPolicy()
        self._episodes: Dict[str, Episode] = {}
        self._mid_notification_times: List[datetime] = []

    # --- utilities -------------------------------------------------------
    @staticmethod
    def _parse_ts(value: Any) -> Optional[datetime]:  # mirror parsing logic used elsewhere
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value))
            except Exception:
                return None
        if isinstance(value, str):
            for fmt in (
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
            ):
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
        return None

    def _severity(self, alert_type: str) -> int:
        return self.policy.severity_rank.get(alert_type, 0)

    def _prune_mid_rate_window(self, now: datetime):
        cutoff = now - timedelta(hours=1)
        self._mid_notification_times = [t for t in self._mid_notification_times if t >= cutoff]

    # --- core API --------------------------------------------------------
    def process(self, alert_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Ingest a filtered alert record.

        Returns a notification dict OR None if aggregation chooses to suppress
        (or wait for more evidence).
        """
        alert_type = alert_record.get("alert")
        ts = self._parse_ts(alert_record.get("timestamp"))
        if not alert_type or ts is None:
            return None

        conf = float(alert_record.get("confidence", 0.0))
        severity = self._severity(alert_type)
        p = self.policy

        # Immediate high severity pass-through
        if alert_type in p.high_severity_immediate:
            return self._build_notification(
                reason="immediate_high_severity",
                alert_type=alert_type,
                severity=severity,
                ts=ts,
                episode=None,
                base_record=alert_record,
            )

        # Mid / low severity episode handling
        episode = self._episodes.get(alert_type)
        gap_thresh = timedelta(seconds=p.episode_gap_seconds)
        if episode is None or ts - episode.last_time > gap_thresh:
            # Start new episode
            episode = Episode(alert_type=alert_type, first_time=ts, last_time=ts, max_confidence=conf)
            self._episodes[alert_type] = episode
        else:
            episode.update(ts, conf)

        # If already notified, do nothing (unless we want escalation logic later)
        if episode.notified:
            return None

        # Escalate on high confidence single event
        if conf >= p.high_confidence_escalation:
            episode.notified = True
            return self._emit_episode_notification(episode, alert_record, ts, severity, reason="high_confidence")

        # Count based escalation within window
        if episode.count >= p.min_mid_severity_count:
            if (ts - episode.first_time).total_seconds() <= p.window_seconds:
                # Rate limiting for mid-level notifications
                self._prune_mid_rate_window(ts)
                if len(self._mid_notification_times) >= p.max_mid_notifications_per_hour:
                    return None  # silently suppress – could also return a 'digest' later
                self._mid_notification_times.append(ts)
                episode.notified = True
                return self._emit_episode_notification(episode, alert_record, ts, severity, reason="count_threshold")

        return None

    # --- helpers ---------------------------------------------------------
    def _emit_episode_notification(
        self,
        episode: Episode,
        base_record: Dict[str, Any],
        ts: datetime,
        severity: int,
        reason: str,
    ) -> Dict[str, Any]:
        return self._build_notification(
            reason=reason,
            alert_type=episode.alert_type,
            severity=severity,
            ts=ts,
            episode=episode,
            base_record=base_record,
        )

    @staticmethod
    def _build_notification(
        reason: str,
        alert_type: str,
        severity: int,
        ts: datetime,
        episode: Episode | None,
        base_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "notification_timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "notification_type": alert_type,
            "severity": severity,
            "reason": reason,
            "confidence": base_record.get("confidence"),
            "alert_record": base_record,
        }
        if episode:
            payload["episode"] = {
                "count": episode.count,
                "first_timestamp": episode.first_time.strftime("%Y-%m-%d %H:%M:%S"),
                "last_timestamp": episode.last_time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": episode.duration(),
                "max_confidence": round(episode.max_confidence, 3),
            }
        return payload

    # --- batch helper ----------------------------------------------------
    def process_many(self, records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in records:
            n = self.process(r)
            if n:
                out.append(n)
        return out


__all__ = [
    "NotificationPolicy",
    "NotificationAggregator",
]
