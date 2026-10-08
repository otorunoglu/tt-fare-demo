"""
Import labeled CSV logs into the SQLite database.

This script decouples data ingestion from the MetaModelDecider. It reads one or more
CSV files and persists rows into the `events` table:

  - timestamp -> events.timestamp (ISO string is preferred; other ISO-like strings accepted)
  - dominant_label -> events.label
  - dominant_label_confidence -> events.confidence
  - loudness -> events.loudness

No decisions or plotting are performed here. Use the Visualizer (DB-based) for plots.
"""

from __future__ import annotations
import csv
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Iterable

from smartstablemodel.services.data_store import DataStore
from smartstablemodel.services.meta_model_decider import MetaModelDecider
from smartstablemodel.config import load_metamodel_config

# Configuration constants
SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = "soundscape.db"
CLEAR_DB_ON_START = True
RECURSIVE = True
# When True, use MetaModelDecider to evaluate events while importing and persist warnings
GENERATE_WARNINGS = True
# Toggle decider features (only used if GENERATE_WARNINGS=True)
ENABLE_GROUP_SEVERITY = True
ENABLE_ADAPTIVE_LOUDNESS = True
PATHS: list[Path] = [
    SCRIPT_DIR.parent / "smart-stable-logs" / "segments" / "stable01" / "2025-08-20.csv",
    SCRIPT_DIR.parent / "smart-stable-logs" / "segments" / "stable01" / "2025-08-21.csv",
    SCRIPT_DIR.parent / "smart-stable-logs" / "segments" / "stable01" / "2025-08-22.csv",
    SCRIPT_DIR.parent / "smart-stable-logs" / "segments" / "stable01" / "2025-08-23.csv",
    SCRIPT_DIR.parent / "smart-stable-logs" / "segments" / "stable01" / "2025-08-24.csv",
    SCRIPT_DIR.parent / "smart-stable-logs" / "segments" / "stable01" / "2025-08-25.csv",
    SCRIPT_DIR.parent / "smart-stable-logs" / "segments" / "stable01" / "2025-08-26.csv",
    SCRIPT_DIR.parent / "smart-stable-logs" / "segments" / "stable01" / "2025-08-27.csv",
    SCRIPT_DIR.parent / "smart-stable-logs" / "segments" / "stable01" / "2025-08-28.csv",
    SCRIPT_DIR.parent / "smart-stable-logs" / "segments" / "stable01" / "2025-08-29.csv",
    SCRIPT_DIR.parent / "smart-stable-logs" / "segments" / "stable01" / "2025-08-30.csv",
]


def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x) if x not in (None, "") else None
    except Exception:
        return None


def _row_to_ds_event(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ts = row.get("timestamp")
    if not ts:
        return None
    ev: Dict[str, Any] = {"timestamp": ts}
    dom = row.get("dominant_label")
    if dom:
        ev["label"] = dom
    conf = _to_float(row.get("dominant_label_confidence"))
    if conf is not None:
        ev["confidence"] = conf
    loud = _to_float(row.get("loudness"))
    if loud is not None:
        ev["loudness"] = loud
    return ev


def _row_to_decider_event(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ts = row.get("timestamp")
    if not ts:
        return None
    ev: Dict[str, Any] = {"timestamp": ts}
    dom = row.get("dominant_label")
    if dom:
        ev["dominant_label"] = dom
    conf = _to_float(row.get("dominant_label_confidence"))
    if conf is not None:
        ev["dominant_label_confidence"] = conf
    loud = _to_float(row.get("loudness"))
    if loud is not None:
        ev["loudness"] = loud
    return ev


def iter_csv(path: Path, *, for_decider: bool = False) -> Iterable[Dict[str, Any]]:
    with open(path, "r", newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ev = _row_to_decider_event(row) if for_decider else _row_to_ds_event(row)
            if ev is None:
                continue
            # Normalize timestamp if it's a datetime-like string that fromisoformat can parse
            ts = ev.get("timestamp")
            try:
                ts_dt = datetime.fromisoformat(str(ts))
                ev["timestamp"] = ts_dt.isoformat()
            except Exception:
                # leave as-is; DataStore will attempt storage
                pass
            yield ev


def main() -> int:
    # Resolve input list from constants
    inputs: list[Path] = []
    for p in PATHS:
        pth = Path(p)
        if pth.is_dir():
            if RECURSIVE:
                inputs.extend(sorted(pth.rglob("*.csv")))
            else:
                inputs.extend(sorted(pth.glob("*.csv")))
        else:
            inputs.append(pth)
    if not inputs:
        print("No input CSV files found.")
        return 1

    ds = DataStore(DB_PATH)
    if CLEAR_DB_ON_START:
        ds.clear()
        print("[import] Cleared events and warnings tables.")

    total = 0
    stored = 0

    if GENERATE_WARNINGS:
        # Route through the decider in batch mode (fast path with memory window and a single transaction inside)
        cfg = load_metamodel_config()
        decider = MetaModelDecider(config=cfg, datastore=ds, enable_group_severity=ENABLE_GROUP_SEVERITY)
        decider.enable_adaptive_loudness = ENABLE_ADAPTIVE_LOUDNESS
        for fp in inputs:
            if not fp.exists():
                print(f"[import] Missing file: {fp}")
                continue
            if fp.suffix.lower() != ".csv":
                print(f"[import] Skipping non-CSV: {fp}")
                continue
            # Stream directly to decider.evaluate_batch
            events_iter = iter_csv(fp, for_decider=True)
            processed, _ = decider.evaluate_batch(
                events_iter,
                fast_plot_only=False,
                disable_warnings=False,
                disable_group_severity=None,
                disable_adaptive_loudness=None,
            )
            total += processed
        stored = ds.count_in_range(start=datetime.min.replace(year=1970), end=datetime.max.replace(year=9999))
        print(f"[import] Processed={total} via decider; events in DB now={stored}")
        # Summarize warnings
        try:
            warns = ds.warnings()
            by_type: dict[str, int] = {}
            for w in warns:
                by_type[w.warning_type.value] = by_type.get(w.warning_type.value, 0) + 1
            if by_type:
                print("[import] Warnings persisted:")
                for k, v in by_type.items():
                    print(f"  {k}: {v}")
            else:
                print("[import] No warnings persisted.")
        except Exception:
            pass
    else:
        # Pure append to events table (fastest path, no warnings)
        ds.begin()
        try:
            for fp in inputs:
                if not fp.exists():
                    print(f"[import] Missing file: {fp}")
                    continue
                if fp.suffix.lower() != ".csv":
                    print(f"[import] Skipping non-CSV: {fp}")
                    continue
                for ev in iter_csv(fp, for_decider=False):
                    total += 1
                    ds.append_event(ev, autocommit=False)
                    stored += 1
        finally:
            ds.commit()
        print(f"[import] Processed={total} stored={stored} into DB={DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
