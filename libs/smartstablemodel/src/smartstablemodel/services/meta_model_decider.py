"""
Decision layer combining heuristic and statistical rules.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Optional

from ..domain import Event, DomainWarning, DecisionResult, WarningType
from .data_store import DataStore
from smartstablemodel.config import MetaModelConfig

import math
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
from statistics import mean
from typing import (
    Dict,
    Any,
    Optional,
    List,
    Deque,
    Iterable,
    Tuple,
    Union,
    Iterable,
    Mapping,
)
from enum import Enum


class MetaModelDecider:
    """Context-aware MetaModel evaluator"""

    @property
    def model_params(self) -> Dict[str, Any]:
        return self.config.model_parameters

    @property
    def loudness_params(self) -> Dict[str, Any]:
        return self.config.loudness_parameters

    def __init__(
        self,
        config: MetaModelConfig,
        datastore: DataStore,
        enable_group_severity: bool = True,
    ):
        self.config = config
        self.datastore = datastore
        # Core parameters (with safe defaults when not present in config)
        self.burst_window_seconds = self.model_params.get("burst_window_seconds", 120)
        # Night definition; default to wrapping night 20:00 -> 07:00 if unspecified
        # Read as separate keys to match the shared TOML config (also used by Rust edge monitor)
        self.night_hours = (
            int(self.model_params.get("night_hour_start", 20)),
            int(self.model_params.get("night_hour_end", 7)),
        )
        self.early_morning = self.model_params.get("early_morning", None)
        # Segment duration used when converting covered-seconds counts
        self.default_snippet_duration_seconds = self.model_params.get(
            "default_snippet_duration_seconds", 1.0
        )
        self.cluster_same_label_only = self.model_params.get(
            "cluster_same_label_only", True
        )
        self.dominant_label_floor = self.model_params["dominant_label_floor"]
        self.background_label = self.model_params["background_label"]
        self.drop_below_floor = self.model_params["drop_below_floor"]
        self.severity_step = self.model_params["severity_step"]
        self.group_cluster_warning_min_severity = self.model_params.get(
            "group_cluster_warning_min_severity", 1.0
        )
        self.merge_gap_seconds = self.model_params["merge_gap_seconds"]
        # Track last emitted cluster severity "level" (integer floor of severity / step)
        self.last_cluster_level: Dict[str, int] = {}
        self.current_night_start: Optional[datetime] = (
            None  # precise night anchor (start_h of night)
        )
        self.current_night_id = None
        self.enable_group_severity = enable_group_severity
        self.last_group_level: Dict[str, int] = {}
        # Adaptive loudness detection
        self.enable_adaptive_loudness = self.loudness_params["enable_adaptive_loudness"]
        self.loudness_warmup_seconds = self.loudness_params["loudness_warmup_seconds"]
        self.loudness_sigma_multiplier = self.loudness_params[
            "loudness_sigma_multiplier"
        ]
        self.loudness_min_window_seconds = self.loudness_params[
            "loudness_min_window_seconds"
        ]
        self.loudness_min_std = self.loudness_params["loudness_min_std"]
        self.loudness_severity_scale = self.loudness_params["loudness_severity_scale"]
        # Adaptive loudness state (Welford)
        self._loud_base_n = 0
        self._loud_base_mean = 0.0
        self._loud_base_M2 = 0.0
        self.last_loudness_level: int = -1
        self.all_warnings: List[
            DomainWarning
        ] = []  # accumulate all warnings across nights
        # Fast-path recent events buffer for batch import (avoid DB queries per event)
        self._use_memory_window: bool = False
        self._recent_events: Deque[Dict[str, Any]] = deque()
        # Toggle indicating we're inside a batch ingest (control DB autocommit)
        self._in_batch: bool = False

    # Back-compat: expose events as a list of dicts for plotting utilities that expect evaluator.events
    @property
    def events(self) -> List[Dict[str, Any]]:
        return [e.as_dict() for e in self.datastore.query()]

    # Back-compat: expose grouped spans across all events for plotting utilities
    @property
    def grouped_events(self) -> List[Dict[str, Any]]:
        return self.get_grouped_events()

    def _coerce_ts(self, value: Any) -> Optional[datetime]:
        """Best-effort conversion to datetime from datetime or ISO-like string; returns None on failure."""
        if isinstance(value, datetime):
            return value
        if value is None:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return None

    def _trim_recent(self, now: datetime) -> None:
        """Trim the in-memory deque to only keep events within the burst window."""
        if not self._recent_events:
            return
        cutoff = now - timedelta(seconds=self.burst_window_seconds)
        while self._recent_events:
            head_ts = self._coerce_ts(self._recent_events[0].get("timestamp"))
            if head_ts is None or head_ts < cutoff:
                self._recent_events.popleft()
                continue
            break

    def get_grouped_events(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        *,
        merge_gap_seconds: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build contiguous spans per label by merging events that are within merge_gap_seconds.
        Returns a list of dicts: {"label": str, "start": datetime, "end": datetime, "seconds": float}

        Notes:
        - Uses DataStore to query raw events in the [start, end] window (or all if None).
        - Uses self.default_snippet_duration_seconds for per-event duration when extending spans.
        - Does not filter by confidence; it reflects what was persisted (including background if stored).
        """
        merge_gap = (
            merge_gap_seconds
            if merge_gap_seconds is not None
            else float(self.merge_gap_seconds)
        )

        # Query events ordered by time
        events = self.datastore.query(start=start, end=end)
        if not events:
            return []

        # Keep an open span per label; when a gap exceeds threshold, close the span
        open_spans: Dict[str, Tuple[datetime, datetime]] = {}
        closed: List[Tuple[str, datetime, datetime]] = []
        seg_dur = float(self.default_snippet_duration_seconds or 1.0)

        for ev in events:
            lab = getattr(ev, "label", None) or ""
            if not lab:
                continue
            ts = (
                ev.timestamp
                if isinstance(ev.timestamp, datetime)
                else datetime.fromisoformat(str(ev.timestamp))
            )
            ev_start = ts
            ev_end = ts + timedelta(seconds=seg_dur)

            if lab not in open_spans:
                open_spans[lab] = (ev_start, ev_end)
                continue

            cur_start, cur_end = open_spans[lab]
            gap = (ev_start - cur_end).total_seconds()
            if gap <= merge_gap:
                # Extend the current span
                new_end = ev_end if ev_end > cur_end else cur_end
                open_spans[lab] = (cur_start, new_end)
            else:
                # Close current span and start a new one
                if cur_end > cur_start:
                    closed.append((lab, cur_start, cur_end))
                open_spans[lab] = (ev_start, ev_end)

        # Close remaining open spans
        for lab, (s, e) in open_spans.items():
            if e > s:
                closed.append((lab, s, e))

        # Map to dicts with seconds
        grouped: List[Dict[str, Any]] = []
        for lab, s, e in closed:
            seconds = (e - s).total_seconds()
            if seconds <= 0:
                continue
            grouped.append({"label": lab, "start": s, "end": e, "seconds": seconds})

        # Stable sort by start time
        grouped.sort(key=lambda d: d["start"])  # type: ignore[index]
        return grouped

    def iter_warnings(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        source: str = "all",
    ) -> Iterable[DomainWarning]:
        """Yield warnings filtered by [start,end]. source: 'all' | 'nightly'."""
        src = self.all_warnings if source == "all" else self.nightly_warnings()
        for w in src:
            if start and w.timestamp < start:
                continue
            if end and w.timestamp > end:
                continue
            yield w

    def _update_loudness_baseline(self, loud: float) -> None:
        """Incremental Welford update."""
        n = self._loud_base_n + 1
        delta = loud - self._loud_base_mean
        mean_new = self._loud_base_mean + delta / n
        delta2 = loud - mean_new
        self._loud_base_M2 += delta * delta2
        self._loud_base_mean = mean_new
        self._loud_base_n = n

    def _loudness_baseline_stats(self) -> Tuple[int, float, float]:
        if self._loud_base_n < 2:
            return self._loud_base_n, self._loud_base_mean, 0.0
        var = self._loud_base_M2 / (self._loud_base_n - 1)
        return (
            self._loud_base_n,
            self._loud_base_mean,
            math.sqrt(var) if var > 0 else 0.0,
        )

    def _window_loudness_mean(self, now: datetime) -> Tuple[int, float]:
        """Return (sampled_seconds, mean_loudness) over burst window (distinct seconds).
        Pulls events from DataStore in [now - window, now] and averages by distinct seconds.
        """
        window_start = now - timedelta(seconds=self.burst_window_seconds)
        if self._use_memory_window:
            iter_ev = reversed(self._recent_events)
            use_db = False
        else:
            recent_events = self.datastore.query(start=window_start, end=now)
            iter_ev = recent_events
            use_db = True
        total = 0.0
        cnt = 0
        seen_secs: set[int] = set()
        for ev in iter_ev:
            if isinstance(ev, dict):
                ts = self._coerce_ts(ev.get("timestamp"))
                v = ev.get("loudness")
            else:
                ts = self._coerce_ts(getattr(ev, "timestamp", None))
                v = getattr(ev, "loudness", None)
            if ts is None:
                continue
            if ts < window_start or ts > now:
                continue
            if v is None:
                continue
            s_key = int(ts.timestamp())
            if s_key in seen_secs:
                continue
            seen_secs.add(s_key)
            total += float(v)
            cnt += 1
        return cnt, (total / cnt if cnt else 0.0)

    def _compute_night_start(self, ts: datetime) -> datetime:
        """
        Return the datetime representing the start of the 'night' that ts belongs to.
        For wrapping nights (e.g. 20 -> 7):
          - Hours >= start_h belong to the night starting that calendar day at start_h.
          - Hours < end_h belong to the night that started the previous day at start_h.
        For non‑wrapping nights (start_h < end_h):
          - If hour >= start_h and < end_h -> same day start_h.
          - Else if hour < start_h -> previous day start_h (we treat continuity).
          - Else (hour >= end_h) -> next upcoming start (same day at start_h already passed; we start a new one).
        (Your use case is wrapping, so simplified path is most relevant.)
        """
        start_h, end_h = self.night_hours
        base = ts.replace(minute=0, second=0, microsecond=0)
        if start_h == end_h:
            # Degenerate: treat every day same anchor
            return base.replace(hour=start_h)

        if start_h > end_h:  # wrapping (e.g. 20 -> 7)
            if ts.hour >= start_h:
                return base.replace(hour=start_h)
            # early morning < end_h -> previous day
            prev = (base - timedelta(days=1)).replace(hour=start_h)
            return prev
        else:
            # non-wrapping (not primary case here)
            if start_h <= ts.hour < end_h:
                return base.replace(hour=start_h)
            if ts.hour < start_h:
                # belongs to previous night's span
                prev = (base - timedelta(days=1)).replace(hour=start_h)
                return prev
            # hour >= end_h -> this is AFTER the defined night window; start a new night at next start
            return base.replace(hour=start_h)

    def _rollover_if_needed(self, ts: datetime):
        """
        New logic: Only roll over when the computed night_start strictly changes.
        Daytime events (between end_h and next start_h) no longer trigger an
        unwanted flush mid 'file'; we only flush when we enter a NEW night start anchor.
        """
        night_start = self._compute_night_start(ts)
        if self.current_night_start is None:
            self.current_night_start = night_start
            self.current_night_id = night_start.strftime("%Y-%m-%d")
            return
        if night_start > self.current_night_start:
            # Reset severity levels for the new night window
            self.last_cluster_level.clear()
            self.last_group_level.clear()
            self.last_loudness_level = -1
            self.current_night_start = night_start
            self.current_night_id = night_start.strftime("%Y-%m-%d")

    def _window_coverages(self, now: datetime) -> Dict[str, float]:
        window_start = now - timedelta(seconds=self.burst_window_seconds)
        sec_sets: Dict[str, set[int]] = {
            lab: set() for lab in self.config.cluster_seconds_per_severity_level
        }
        if self._use_memory_window:
            iterable = reversed(self._recent_events)
            use_db = False
        else:
            iterable = self.datastore.query(end=now, start=window_start)
            use_db = True
        for ev in iterable:
            if isinstance(ev, dict):
                ts = self._coerce_ts(ev.get("timestamp"))
                lab = ev.get("label")
                conf = float(ev.get("confidence") or 0.0)
            else:
                ts = self._coerce_ts(getattr(ev, "timestamp", None))
                lab = getattr(ev, "label", None)
                conf = getattr(ev, "confidence", 0.0) or 0.0
            if ts is None:
                continue
            if ts < window_start:
                continue
            if not lab or lab not in self.config.cluster_seconds_per_severity_level:
                continue
            if conf < self._cluster_confidence_floor(lab):
                continue
            sec_sets[lab].add(int(ts.timestamp()))
        dur = self.default_snippet_duration_seconds
        return {lab: len(secs) * dur for lab, secs in sec_sets.items()}

    def _window_group_coverages(self, now: datetime) -> Dict[str, float]:
        """Union of per-second coverage across labels in each group (no double counting).
        Includes all observed labels (not just cluster labels) so 'other' and 'normal' show coverage.
        """
        window_start = now - timedelta(seconds=self.burst_window_seconds)
        if self._use_memory_window:
            iterable = reversed(self._recent_events)
            use_db = False
        else:
            iterable = self.datastore.query(end=now, start=window_start)
            use_db = True
        per_label_secs: Dict[str, set[int]] = {}
        for ev in iterable:
            if isinstance(ev, dict):
                ts = self._coerce_ts(ev.get("timestamp"))
                lab = ev.get("label")
                conf = float(ev.get("confidence") or 0.0)
            else:
                ts = self._coerce_ts(getattr(ev, "timestamp", None))
                lab = getattr(ev, "label", None)
                conf = getattr(ev, "confidence", 0.0) or 0.0
            if ts is None:
                continue
            if ts < window_start:
                continue
            if not lab:
                continue
            # still enforce confidence floor for cluster labels
            if lab in self.config.cluster_seconds_per_severity_level:
                floor = self._cluster_confidence_floor(lab)
                if conf < floor:
                    continue
            per_label_secs.setdefault(lab, set()).add(int(ts.timestamp()))
        dur = self.default_snippet_duration_seconds
        group_cover: Dict[str, float] = {}
        for gname, members in self.config.label_groups.items():
            union_set: set[int] = set()
            for m in members:
                if m in per_label_secs:
                    union_set.update(per_label_secs[m])
            if not union_set:
                continue
            group_cover[gname] = len(union_set) * dur
        return group_cover

    # Helper: get confidence floor for a cluster label
    def _cluster_confidence_floor(self, label: str) -> float:
        return self.config.cluster_confidence_floor_overrides.get(
            label, self.config.default_cluster_confidence_floor
        )

    def _cluster_warning_min_severity(self, label: str) -> float:
        return self.config.cluster_warning_min_severity_overrides.get(label, 1.0)

    # --- core evaluation --------------------------------------------------
    def evaluate(self, current_event: Dict[str, Any]) -> DecisionResult:
        """Evaluate a single event in context of recent history."""
        ts_raw = current_event.get("timestamp")
        if isinstance(ts_raw, datetime):
            ts = ts_raw
        else:
            ts = datetime.fromisoformat(str(ts_raw))

        # Night rollover
        self._rollover_if_needed(ts)

        label = ""
        conf = 0.0
        # if no label_distribution, assume we have dominant_label directly (csv export case)
        if "label_distribution" in current_event:
            labels = current_event.get("label_distribution", {})

            # Determine dominant label if any
            if labels:
                label, conf = max(labels.items(), key=lambda x: x[1])
        else:
            # assume dominant_label is present
            label = current_event.get("dominant_label", "")
            conf = float(current_event.get("dominant_label_confidence", 0.0))
        # Apply dominant label floor: always persist as background (no hard drop),
        # so DB keeps one record per second even if the model confidence is low.
        if conf < self.dominant_label_floor:
            label = self.background_label
            conf = 1.0  # normalize to background

        # Normalize loudness (extract from metrics if needed) before persisting
        loud = current_event.get("loudness")
        if loud is None:
            loud = (
                current_event.get("loudness_metrics", {})
                .get("segment", {})
                .get("value")
            )
        if isinstance(loud, (int, float)):
            current_event["loudness"] = float(loud)
        else:
            current_event["loudness"] = None

        # Ensure timestamp is ISO string for storage
        current_event["timestamp"] = ts.isoformat()

        current_event["label"] = label
        current_event["confidence"] = conf
        # Store event using DataStore (defer commit handled by batch when applicable)
        self.datastore.append_event(current_event, autocommit=(not self._in_batch))
        # Append to memory window for fast coverage calculations (normalized record)
        if self._use_memory_window:
            self._recent_events.append(
                {
                    "timestamp": ts,
                    "label": label,
                    "confidence": conf,
                    "loudness": current_event.get("loudness"),
                }
            )
            self._trim_recent(ts)

        # Decide if current event is an alert label and trigger alert
        alert_threshold = self.config.alert_confidence_thresholds.get(
            label, self.config.alert_labels_confidence_floor
        )
        if label in self.config.alert_labels and conf >= alert_threshold:
            # Immediate alert
            warning = DomainWarning(
                label=label,
                timestamp=ts,
                confidence=conf,
                warning_type=WarningType.ALERT,
                severity=self.config.alert_labels[label],
            )
            self.all_warnings.append(warning)
            try:
                self.datastore.append_warning(warning)
            except Exception:
                pass
            return DecisionResult(
                warning_type=WarningType.ALERT,
                label=label,
                severity=self.config.alert_labels[label],
                confidence=conf,
                contributing_factors=[
                    f"Immediate alert for label '{label}' with confidence {conf:.2f}"
                ],
                debug={"reason": "immediate_alert", "event": current_event},
            )

        # Decide if there was a burst in the past burst_window_seconds
        if (label in self.config.cluster_seconds_per_severity_level.keys()) and (
            conf >= self._cluster_confidence_floor(label)
        ):
            coverages = self._window_coverages(ts)
            covered = coverages.get(label, 0.0)
            seconds_per_level = self.config.cluster_seconds_per_severity_level[label]
            severity_raw = covered / seconds_per_level if seconds_per_level > 0 else 0.0
            level = int(math.floor(severity_raw / self.severity_step))
            last_level = self.last_cluster_level.get(label, -1)
            min_severity = self._cluster_warning_min_severity(label)
            if severity_raw >= min_severity and level > last_level:
                # Emit escalating alert
                self.last_cluster_level[label] = level
                contributing_factors = [
                    f"{covered:.1f}s '{label}' in last {self.burst_window_seconds}s",
                    f"seconds_per_severity_level={seconds_per_level}",
                    f"min_severity={min_severity}",
                    f"severity={severity_raw:.2f} (level {level})",
                ]
                warning = DomainWarning(
                    label=label,
                    timestamp=ts,
                    confidence=conf,
                    warning_type=WarningType.CLUSTER,
                    severity=severity_raw,
                )
                self.all_warnings.append(warning)
                try:
                    self.datastore.append_warning(warning)
                except Exception:
                    pass
                return DecisionResult(
                    warning_type=WarningType.CLUSTER,
                    severity=severity_raw,
                    label=label,
                    confidence=conf,
                    contributing_factors=contributing_factors,
                    debug={
                        "reason": "severity_threshold_cross",
                        "covered_seconds": covered,
                        "seconds_per_severity_level": seconds_per_level,
                        "min_severity": min_severity,
                        "severity_raw": severity_raw,
                        "level": level,
                    },
                )

        if self.enable_group_severity:
            group_coverages = self._window_group_coverages(ts)
            for gname, covered in group_coverages.items():
                seconds_per_level = self.config.group_seconds_per_severity_level.get(
                    gname, 0.0
                )
                if seconds_per_level <= 0:
                    continue
                severity_raw = covered / seconds_per_level
                level = int(math.floor(severity_raw / self.severity_step))
                last_level = self.last_group_level.get(gname, -1)
                if (
                    severity_raw >= self.group_cluster_warning_min_severity
                    and level > last_level
                ):
                    self.last_group_level[gname] = level
                    warning = DomainWarning(
                        label=f"group:{gname}",
                        timestamp=ts,
                        confidence=1.0,
                        warning_type=WarningType.CLUSTER,
                        severity=severity_raw,
                    )
                    self.all_warnings.append(warning)
                    try:
                        self.datastore.append_warning(warning)
                    except Exception:
                        pass
                    return DecisionResult(
                        warning_type=WarningType.CLUSTER,
                        severity=severity_raw,
                        label=f"group:{gname}",
                        confidence=1.0,
                        contributing_factors=[
                            f"{covered:.1f}s group '{gname}' (union) in {self.burst_window_seconds}s",
                            f"group_seconds_per_severity_level={seconds_per_level}",
                            f"severity={severity_raw:.2f} (level {level})",
                        ],
                        debug={
                            "reason": "group_severity_threshold_cross",
                            "group": gname,
                            "covered_seconds": covered,
                            "seconds_per_severity_level": seconds_per_level,
                            "severity_raw": severity_raw,
                            "level": level,
                        },
                    )

        if self.enable_adaptive_loudness:
            # Capture current sample loudness (if present) then update baseline AFTER decision
            current_loud = current_event.get("loudness", None)
            if current_loud is None:
                current_loud = (
                    current_event.get("loudness_metrics", {})
                    .get("segment", {})
                    .get("value")
                )
            samples, win_mean = self._window_loudness_mean(ts)
            base_n, base_mean, base_std = self._loudness_baseline_stats()
            # Determine night elapsed for warmup
            night_elapsed = (
                (ts - self.current_night_start).total_seconds()
                if self.current_night_start
                else 0
            )
            # Conditions to evaluate anomaly
            if (
                base_n >= 30
                and night_elapsed >= self.loudness_warmup_seconds
                and samples >= self.loudness_min_window_seconds
            ):
                eff_std = max(base_std, self.loudness_min_std)
                z = (win_mean - base_mean) / eff_std if eff_std > 0 else 0.0
                if z >= self.loudness_sigma_multiplier:
                    severity_raw = (
                        z / self.loudness_sigma_multiplier
                    ) * self.loudness_severity_scale
                    level = int(math.floor(severity_raw / self.severity_step))
                    if severity_raw >= 1.0 and level > self.last_loudness_level:
                        self.last_loudness_level = level
                        warning = DomainWarning(
                            label="loudness_adaptive",
                            timestamp=ts,
                            confidence=1.0,
                            warning_type=WarningType.LOUDNESS,
                            severity=severity_raw,
                        )
                        self.all_warnings.append(warning)
                        try:
                            self.datastore.append_warning(warning)
                        except Exception:
                            pass
                        # Update baseline after adding warning (still include this sample)
                        if isinstance(current_loud, (int, float)):
                            self._update_loudness_baseline(float(current_loud))
                        return DecisionResult(
                            warning_type=WarningType.LOUDNESS,
                            severity=severity_raw,
                            label="loudness_adaptive",
                            confidence=1.0,
                            contributing_factors=[
                                f"win_mean={win_mean:.3f} > μ+{self.loudness_sigma_multiplier}σ",
                                f"μ={base_mean:.3f}, σ={base_std:.3f}, z={z:.2f}",
                                f"samples={samples}, severity={severity_raw:.2f} (level {level})",
                            ],
                            debug={
                                "reason": "adaptive_loudness_alert",
                                "window_mean": win_mean,
                                "baseline_mean": base_mean,
                                "baseline_std": base_std,
                                "z": z,
                                "baseline_n": base_n,
                                "samples_secs": samples,
                            },
                        )
            # Always update baseline if we had a numeric loudness sample
            if isinstance(current_loud, (int, float)):
                self._update_loudness_baseline(float(current_loud))

        # return the decision result
        return DecisionResult(
            warning_type=None,
            label="",
            confidence=0.0,
            contributing_factors=[],
            debug={"reason": "no_alert", "event": current_event},
        )

    def _ingest_event_for_plot(self, ev: Dict[str, Any]) -> None:
        """
        Fast path to ingest an event for plotting:
        - determine dominant label + confidence
        - apply floor/background logic
        - append to self.events and update grouped_events
        - update loudness baseline
        Skips all warning/coverage computations for speed.
        """
        ts_str = str(ev.get("timestamp"))
        # Ensure ISO-parseable (fromisoformat handles "YYYY-mm-dd HH:MM:SS")
        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception:
            return  # skip malformed

        # Night rollover
        self._rollover_if_needed(ts)

        # Determine dominant label/conf
        label = ""
        conf = 0.0
        labels = ev.get("label_distribution", {})
        if labels:
            label, conf = max(labels.items(), key=lambda x: x[1])
        else:
            label = ev.get("dominant_label", "") or ""
            conf = float(ev.get("dominant_label_confidence", 0.0) or 0.0)

        # Apply dominant label floor: do not drop; persist as background for completeness
        if conf < self.dominant_label_floor:
            label = self.background_label
            conf = 1.0

        ev["label"] = label
        ev["confidence"] = conf

        # Normalize loudness before store
        loud = ev.get("loudness", None)
        if loud is None:
            loud = ev.get("loudness_metrics", {}).get("segment", {}).get("value")
        if isinstance(loud, (int, float)):
            ev["loudness"] = float(loud)
        else:
            ev["loudness"] = None

        # Ensure timestamp is ISO string for storage
        ev_ts = ev.get("timestamp")
        if isinstance(ev_ts, datetime):
            ev["timestamp"] = ev_ts.isoformat()

        # Store (grouping hook is a no-op); defer commit for batch
        self.datastore.append_event(ev, autocommit=(not self._in_batch))
        if self._use_memory_window:
            self._recent_events.append(
                {
                    "timestamp": ts,
                    "label": label,
                    "confidence": conf,
                    "loudness": ev.get("loudness"),
                }
            )
            self._trim_recent(ts)

        # Update loudness baseline (if numeric available)
        loud = ev.get("loudness", None)
        if loud is None:
            loud = ev.get("loudness_metrics", {}).get("segment", {}).get("value")
        if isinstance(loud, (int, float)):
            self._update_loudness_baseline(float(loud))

    def nightly_warnings(self) -> List[DomainWarning]:
        """Return the list of warnings for the current night."""
        return self.all_warnings

    def evaluate_batch(
        self,
        events: Iterable[Dict[str, Any]],
        *,
        fast_plot_only: bool = True,
        disable_warnings: bool = True,
        disable_group_severity: Optional[bool] = None,
        disable_adaptive_loudness: Optional[bool] = None,
    ) -> tuple[int, int]:
        """
        DEPRECATED: The decider is a realtime, per-event evaluator. For offline imports,
        use scripts/import_logs.py to populate the DB and services/visualizer.py for plots.

        This method remains for temporary compatibility with metamodel_replay, but it will be
        removed. Prefer decoupled ingestion and visualization.
        """
        processed = 0
        warnings_before = len(self.nightly_warnings())

        # Save toggles and temporarily disable expensive features if requested
        orig_group_sev = self.enable_group_severity
        orig_adapt_loud = self.enable_adaptive_loudness
        try:
            # Begin a transaction for faster bulk insert
            self.datastore.begin()
            # Enable memory window for fast intra-batch windows
            self._use_memory_window = True
            self._recent_events.clear()
            self._in_batch = True
            if disable_group_severity is not None:
                self.enable_group_severity = not disable_group_severity
            if disable_adaptive_loudness is not None:
                self.enable_adaptive_loudness = not disable_adaptive_loudness

            if fast_plot_only or disable_warnings:
                # Pure ingest for plots (fast)
                for ev in events:
                    self._ingest_event_for_plot(ev)
                    processed += 1
            else:
                # Full decisions (still sequential)
                for ev in events:
                    res = self.evaluate(ev)
                    processed += 1
        finally:
            # Restore flags
            self.enable_group_severity = orig_group_sev
            self.enable_adaptive_loudness = orig_adapt_loud
            # Commit all appended events
            self.datastore.commit()
            # Disable memory window
            self._use_memory_window = False
            self._in_batch = False

        warnings_after = len(self.nightly_warnings())
        return processed, (warnings_after - warnings_before)
