"""
Central persistence abstraction for recorder events and generated warnings.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Iterable, Dict, Any

import pandas as pd

from ..domain import Event
from ..domain.warning import DomainWarning as DomainWarning, WarningType


class DataStore:
    """
    Adapter that persists events and exposes query helpers for downstream services.
    """

    def __init__(self, db_path: str | Path = "soundscape.db") -> None:
        self._path = Path(db_path)
        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._has_epoch: bool = False
        self._init_schema()

    # -------------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------------
    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                timestamp TEXT PRIMARY KEY,
                label TEXT,
                confidence REAL,
                loudness REAL,
                spectral_centroid REAL,
                high_freq_ratio REAL,
                original_label TEXT
            );
            """
        )
        # Warnings table for overlays
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS warnings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                label TEXT,
                confidence REAL,
                warning_type TEXT NOT NULL,
                severity REAL
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_warnings_timestamp ON warnings(timestamp)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_warnings_type ON warnings(warning_type)"
        )
        # Helpful indexes for visualization queries
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_events_label ON events(label);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
            """
        )
        # Composite index for range+label scans
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_events_ts_label ON events(timestamp, label);
            """
        )
        # Try to add a generated epoch column for faster binning (SQLite 3.31+)
        try:
            cur.execute(
                """
                ALTER TABLE events ADD COLUMN epoch INTEGER
                GENERATED ALWAYS AS (CAST(strftime('%s', timestamp) AS INTEGER)) STORED
                """
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            # Column exists already or SQLite too old; ignore
            pass

        # Schema migration: check for original_label
        try:
            info = cur.execute("PRAGMA table_info(events)").fetchall()
            cols = {row[1] for row in info}
            if "original_label" not in cols:
                cur.execute("ALTER TABLE events ADD COLUMN original_label TEXT")
            if "spectral_centroid" not in cols:
                cur.execute("ALTER TABLE events ADD COLUMN spectral_centroid REAL")
            if "high_freq_ratio" not in cols:
                cur.execute("ALTER TABLE events ADD COLUMN high_freq_ratio REAL")
                self._conn.commit()
        except sqlite3.OperationalError:
            pass

        # Detect presence of epoch column
        try:
            info = cur.execute("PRAGMA table_info(events)").fetchall()
            cols = {row[1] for row in info}  # row[1] is name
            self._has_epoch = "epoch" in cols
        except sqlite3.OperationalError:
            self._has_epoch = False
        # Index epoch if present
        if self._has_epoch:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_epoch ON events(epoch)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_epoch_label ON events(epoch, label)"
            )
        self._conn.commit()

    # -------------------------------------------------------------------------
    # Core Interface
    # -------------------------------------------------------------------------
    def append_event(
        self, event: Event | Dict[str, Any], *, autocommit: bool = True
    ) -> None:
        """
        Persist an event into the underlying storage.
        """
        if isinstance(event, Event):
            data = event.as_dict()
        else:
            data = dict(event)

        # Normalize timestamp to ISO string
        ts = data.get("timestamp")
        if isinstance(ts, datetime):
            data["timestamp"] = ts.isoformat()

        # Ensure numeric fields are properly converted
        conf_val = data.get("confidence")
        try:
            conf_out = float(conf_val) if conf_val is not None else 0.0  # type: ignore[arg-type]
        except (TypeError, ValueError):
            conf_out = 0.0

        loud_val = data.get("loudness")
        if loud_val is None:
            loud_out = None
        else:
            try:
                loud_out = float(loud_val)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                loud_out = None

        centroid_val = data.get("spectral_centroid")
        if centroid_val is None:
            centroid_out = None
        else:
            try:
                centroid_out = float(centroid_val)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                centroid_out = None

        hf_ratio_val = data.get("high_freq_ratio")
        if hf_ratio_val is None:
            hf_ratio_out = None
        else:
            try:
                hf_ratio_out = float(hf_ratio_val)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                hf_ratio_out = None

        self._conn.execute(
            """
            INSERT OR REPLACE INTO events (
                timestamp,
                label,
                confidence,
                loudness,
                spectral_centroid,
                high_freq_ratio,
                original_label
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.get("timestamp"),
                data.get("label"),
                conf_out,
                loud_out,
                centroid_out,
                hf_ratio_out,
                data.get("original_label"),
            ),
        )
        if autocommit:
            self._conn.commit()

    # Transaction helpers for bulk operations
    def begin(self) -> None:
        self._conn.execute("BEGIN")

    def commit(self) -> None:
        self._conn.commit()

    def query(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        last_seconds: Optional[int] = None,
    ) -> Sequence[Event]:
        """
        Return a time-ordered sequence of events in the given window.
        """
        query = "SELECT * FROM events"
        params: list = []
        conditions: list[str] = []

        if last_seconds is not None:
            query += " WHERE timestamp >= datetime('now', ?)"
            params.append(f"-{int(last_seconds)} seconds")
        else:
            if start:
                conditions.append("timestamp >= ?")
                params.append(start.isoformat())
            if end:
                conditions.append("timestamp <= ?")
                params.append(end.isoformat())
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp ASC"

        df = pd.read_sql_query(query, self._conn, params=params)
        return [
            Event(
                timestamp=datetime.fromisoformat(row["timestamp"]),
                label=row["label"],
                confidence=row["confidence"],
                loudness=row["loudness"],
                spectral_centroid=row["spectral_centroid"]
                if "spectral_centroid" in row
                else None,
                high_freq_ratio=row["high_freq_ratio"]
                if "high_freq_ratio" in row
                else None,
                original_label=row["original_label"]
                if "original_label" in row
                else None,
            )
            for _, row in df.iterrows()
        ]

    def bin_avg_loudness(
        self,
        *,
        start: datetime,
        end: datetime,
        bin_seconds: int,
    ) -> Dict[int, float]:
        """
        Average loudness per bin in [start, end). Ignores NULL loudness.
        """
        t0_sec = int(start.timestamp())
        if self._has_epoch:
            # Compute t0 epoch inside SQLite to avoid timezone discrepancies
            sql = (
                "SELECT CAST((epoch - strftime('%s', ?)) / ? AS INTEGER) AS bin, AVG(loudness) as avg_l "
                "FROM events WHERE timestamp >= ? AND timestamp < ? AND loudness IS NOT NULL "
                "GROUP BY bin ORDER BY bin"
            )
            params = (
                start.isoformat(),
                int(bin_seconds),
                start.isoformat(),
                end.isoformat(),
            )
        else:
            sql = (
                "SELECT CAST((strftime('%s', timestamp) - strftime('%s', ?)) / ? AS INTEGER) AS bin, AVG(loudness) as avg_l "
                "FROM events WHERE timestamp >= ? AND timestamp < ? AND loudness IS NOT NULL "
                "GROUP BY bin ORDER BY bin"
            )
            params = (
                start.isoformat(),
                int(bin_seconds),
                start.isoformat(),
                end.isoformat(),
            )
        cur = self._conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        return {
            int(b): float(avg) for (b, avg) in rows if b is not None and avg is not None
        }

    def _bin_counts_by_label_epoch(
        self,
        *,
        start: datetime,
        end: datetime,
        bin_seconds: int,
    ) -> Dict[str, Dict[int, int]]:
        t0_sec = int(start.timestamp())
        sql = (
            "SELECT label, CAST((epoch - ?) / ? AS INTEGER) AS bin, COUNT(*) as c "
            "FROM events WHERE epoch >= ? AND epoch < ? "
            "GROUP BY label, bin ORDER BY label, bin"
        )
        params = (
            t0_sec,
            int(bin_seconds),
            int(start.timestamp()),
            int(end.timestamp()),
        )
        cur = self._conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        out: Dict[str, Dict[int, int]] = {}
        for label, bin_idx, c in rows:
            if label is None or bin_idx is None:
                continue
            out.setdefault(label, {})[int(bin_idx)] = int(c)
        return out

    def bin_counts_by_label(
        self,
        *,
        start: datetime,
        end: datetime,
        bin_seconds: int,
        confidence_threshold: float = 0.0,
        background_label: str = "background",
        reassign_low_confidence_to_background: bool = True,
    ) -> Dict[str, Dict[int, int]]:
        """
        Return counts per (label, bin) where bin is computed from start at bin_seconds resolution.
        Prefers the epoch column if available to avoid per-row strftime costs.
        """
        threshold = float(confidence_threshold)
        if self._has_epoch:
            # Use epoch for event times but compute t0 epoch in SQLite for consistency
            if threshold > 0.0:
                if reassign_low_confidence_to_background:
                    sql = (
                        "SELECT CASE WHEN COALESCE(confidence, 0.0) >= ? THEN label ELSE ? END AS effective_label, "
                        "CAST((epoch - strftime('%s', ?)) / ? AS INTEGER) AS bin, COUNT(*) as c "
                        "FROM events WHERE timestamp >= ? AND timestamp < ? "
                        "GROUP BY effective_label, bin ORDER BY effective_label, bin"
                    )
                    params = (
                        threshold,
                        background_label,
                        start.isoformat(),
                        int(bin_seconds),
                        start.isoformat(),
                        end.isoformat(),
                    )
                else:
                    sql = (
                        "SELECT label, CAST((epoch - strftime('%s', ?)) / ? AS INTEGER) AS bin, COUNT(*) as c "
                        "FROM events WHERE timestamp >= ? AND timestamp < ? AND COALESCE(confidence, 0.0) >= ? "
                        "GROUP BY label, bin ORDER BY label, bin"
                    )
                    params = (
                        start.isoformat(),
                        int(bin_seconds),
                        start.isoformat(),
                        end.isoformat(),
                        threshold,
                    )
            else:
                sql = (
                    "SELECT label, CAST((epoch - strftime('%s', ?)) / ? AS INTEGER) AS bin, COUNT(*) as c "
                    "FROM events WHERE timestamp >= ? AND timestamp < ? "
                    "GROUP BY label, bin ORDER BY label, bin"
                )
                params = (
                    start.isoformat(),
                    int(bin_seconds),
                    start.isoformat(),
                    end.isoformat(),
                )
        else:
            if threshold > 0.0:
                if reassign_low_confidence_to_background:
                    sql = (
                        "SELECT CASE WHEN COALESCE(confidence, 0.0) >= ? THEN label ELSE ? END AS effective_label, "
                        "CAST((strftime('%s', timestamp) - strftime('%s', ?)) / ? AS INTEGER) AS bin, COUNT(*) as c "
                        "FROM events WHERE timestamp >= ? AND timestamp < ? "
                        "GROUP BY effective_label, bin ORDER BY effective_label, bin"
                    )
                    params = (
                        threshold,
                        background_label,
                        start.isoformat(),
                        int(bin_seconds),
                        start.isoformat(),
                        end.isoformat(),
                    )
                else:
                    sql = (
                        "SELECT label, CAST((strftime('%s', timestamp) - strftime('%s', ?)) / ? AS INTEGER) AS bin, COUNT(*) as c "
                        "FROM events WHERE timestamp >= ? AND timestamp < ? AND COALESCE(confidence, 0.0) >= ? "
                        "GROUP BY label, bin ORDER BY label, bin"
                    )
                    params = (
                        start.isoformat(),
                        int(bin_seconds),
                        start.isoformat(),
                        end.isoformat(),
                        threshold,
                    )
            else:
                sql = (
                    "SELECT label, CAST((strftime('%s', timestamp) - strftime('%s', ?)) / ? AS INTEGER) AS bin, COUNT(*) as c "
                    "FROM events WHERE timestamp >= ? AND timestamp < ? "
                    "GROUP BY label, bin ORDER BY label, bin"
                )
                params = (
                    start.isoformat(),
                    int(bin_seconds),
                    start.isoformat(),
                    end.isoformat(),
                )
        cur = self._conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        out: Dict[str, Dict[int, int]] = {}
        for label, bin_idx, c in rows:
            if label is None or bin_idx is None:
                continue
            out.setdefault(label, {})[int(bin_idx)] = int(c)
        return out

    def anomaly_details_by_bin(
        self,
        *,
        start: datetime,
        end: datetime,
        bin_seconds: int,
    ) -> Dict[str, Dict[int, str]]:
        """
        Return per-bin 'original_label' aggregation for 'anomaly_abnormal' label.
        Returns: {'anomaly_abnormal': {bin_idx: 'label1, label2...'}}
        """
        # We only care about anomaly_abnormal events that have an original_label
        where_clause = "label = 'anomaly_abnormal' AND original_label IS NOT NULL"

        if self._has_epoch:
            sql = (
                f"SELECT CAST((epoch - strftime('%s', ?)) / ? AS INTEGER) AS bin, "
                f"GROUP_CONCAT(original_label, ', ') as details "
                f"FROM events WHERE timestamp >= ? AND timestamp < ? AND {where_clause} "
                f"GROUP BY bin"
            )
            params = (
                start.isoformat(),
                int(bin_seconds),
                start.isoformat(),
                end.isoformat(),
            )
        else:
            sql = (
                f"SELECT CAST((strftime('%s', timestamp) - strftime('%s', ?)) / ? AS INTEGER) AS bin, "
                f"GROUP_CONCAT(original_label, ', ') as details "
                f"FROM events WHERE timestamp >= ? AND timestamp < ? AND {where_clause} "
                f"GROUP BY bin"
            )
            params = (
                start.isoformat(),
                int(bin_seconds),
                start.isoformat(),
                end.isoformat(),
            )

        cur = self._conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()

        out_bins: Dict[int, str] = {}
        for bin_idx, details in rows:
            if bin_idx is not None and details:
                # Deduplicate if many identical labels (optional, but cleaner)
                # details is "neigh, neigh, neigh" -> "neigh (x3)" or just uniq list
                # For simplicity, let's just split, uniq, and join
                raw_labels = [s.strip() for s in details.split(",")]
                counts = {}
                for l in raw_labels:
                    counts[l] = counts.get(l, 0) + 1

                # Format: "neigh (5), kick (2)"
                formatted = ", ".join(
                    [f"{l} ({c})" if c > 1 else l for l, c in counts.items()]
                )
                out_bins[int(bin_idx)] = formatted

        return {"anomaly_abnormal": out_bins} if out_bins else {}

    def import_from_csv(self, file_path: Path) -> Iterable[Event]:
        """
        Bulk-import events from a labelled CSV dataset (LabelStudio export format).
        """
        df = pd.read_csv(file_path)
        events: list[Event] = []
        for _, row in df.iterrows():
            conf_val = row.get("dominant_label_confidence", 1.0)
            confidence = float(conf_val) if not pd.isna(conf_val) else 1.0

            loud_val = row.get("loudness", None)
            loudness = (
                float(loud_val)
                if loud_val is not None and not pd.isna(loud_val)
                else 0.0
            )

            # Parse timestamp string into datetime
            ts_raw = row["timestamp"]
            ts = datetime.fromisoformat(ts_raw) if isinstance(ts_raw, str) else ts_raw
            e = Event(
                timestamp=ts,
                label=row["dominant_label"],
                confidence=confidence,
                loudness=loudness,
            )
            self.append_event(e)
            events.append(e)
        return events

    # -------------------------------------------------------------------------
    # Convenience
    # -------------------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self._path

    def clear(self) -> None:
        """Remove all events."""
        self._conn.execute("DELETE FROM events")
        self._conn.execute("DELETE FROM warnings")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---------------------------------------------------------------------
    # Fast metadata queries
    # ---------------------------------------------------------------------
    def time_bounds(self) -> tuple[Optional[datetime], Optional[datetime]]:
        """Return (min_timestamp, max_timestamp) from the events table."""
        cur = self._conn.cursor()
        cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM events")
        row = cur.fetchone()
        if not row:
            return None, None
        min_ts, max_ts = row
        return (
            datetime.fromisoformat(min_ts) if min_ts else None,
            datetime.fromisoformat(max_ts) if max_ts else None,
        )

    def dates_with_recordings(self) -> list:
        """Return list of dates (as date objects) that have at least one event."""
        from datetime import date as date_type

        cur = self._conn.cursor()
        cur.execute(
            "SELECT DISTINCT DATE(timestamp) FROM events ORDER BY DATE(timestamp)"
        )
        rows = cur.fetchall()
        result = []
        for (d,) in rows:
            if d:
                try:
                    result.append(date_type.fromisoformat(d))
                except ValueError:
                    pass
        return result

    def count_in_range(self, *, start: datetime, end: datetime) -> int:
        """Return number of events in [start, end)."""
        cur = self._conn.cursor()
        if self._has_epoch:
            cur.execute(
                "SELECT COUNT(*) FROM events WHERE epoch >= ? AND epoch < ?",
                (int(start.timestamp()), int(end.timestamp())),
            )
        else:
            cur.execute(
                "SELECT COUNT(*) FROM events WHERE timestamp >= ? AND timestamp < ?",
                (start.isoformat(), end.isoformat()),
            )
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def loudness_stats(
        self, *, start: datetime, end: datetime
    ) -> tuple[int, float, float]:
        """
        Compute (n, mean, std) for loudness values in [start, end).
        Uses an online algorithm to avoid loading everything into memory.
        Returns (0, 0.0, 0.0) if no samples.
        """
        cur = self._conn.cursor()
        if self._has_epoch:
            cur.execute(
                "SELECT loudness FROM events WHERE epoch >= ? AND epoch < ? AND loudness IS NOT NULL",
                (int(start.timestamp()), int(end.timestamp())),
            )
        else:
            cur.execute(
                "SELECT loudness FROM events WHERE timestamp >= ? AND timestamp < ? AND loudness IS NOT NULL",
                (start.isoformat(), end.isoformat()),
            )
        n = 0
        mean = 0.0
        M2 = 0.0
        for (lv,) in cur:
            try:
                x = float(lv)
            except (TypeError, ValueError):
                continue
            n += 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            M2 += delta * delta2
        if n < 2:
            return n, mean if n else 0.0, 0.0
        variance = M2 / (n - 1)
        std = (variance**0.5) if variance > 0 else 0.0
        return n, mean, std

    # ---------------------------------------------------------------------
    # Warnings persistence and queries
    # ---------------------------------------------------------------------
    def append_warning(
        self, warning: DomainWarning, *, autocommit: bool = True
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO warnings (timestamp, label, confidence, warning_type, severity)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                warning.timestamp.isoformat(),
                warning.label,
                float(warning.confidence),
                warning.warning_type.value,
                float(warning.severity),
            ),
        )
        if autocommit:
            self._conn.commit()

    def warnings(
        self,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        types: Optional[Sequence[WarningType]] = None,
    ) -> Sequence[DomainWarning]:
        query = (
            "SELECT timestamp, label, confidence, warning_type, severity FROM warnings"
        )
        params: list = []
        conds: list[str] = []
        if start is not None:
            conds.append("timestamp >= ?")
            params.append(start.isoformat())
        if end is not None:
            conds.append("timestamp <= ?")
            params.append(end.isoformat())
        if types:
            placeholders = ",".join(["?"] * len(types))
            conds.append(f"warning_type IN ({placeholders})")
            params.extend([t.value for t in types])
        if conds:
            query += " WHERE " + " AND ".join(conds)
        query += " ORDER BY timestamp ASC"
        cur = self._conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        out: list[DomainWarning] = []
        for ts, label, conf, wt, sev in rows:
            try:
                out.append(
                    DomainWarning(
                        label=label or "",
                        timestamp=datetime.fromisoformat(ts),
                        confidence=float(conf or 0.0),
                        warning_type=WarningType(wt),
                        severity=float(sev or 0.0),
                    )
                )
            except Exception:
                continue
        return out
