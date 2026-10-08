from __future__ import annotations

import csv
import json
import gzip
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from time import sleep

from smartstablemodel import MetaModelDecider, WarningType
from smartstablemodel.config import load_metamodel_config
from smartstablemodel.services.data_store import DataStore

# --- Configuration ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILES: List[Path] = [
    # Adjust or extend as needed; relative paths are resolved from this script's folder
    SCRIPT_DIR.parent / "smart-stable-logs" / "segments" / "stable01" / "2025-08-29.csv",
]
CLEAR_DB_ON_START = False  # If True, clears the events table before replay
SLEEP_SECONDS: float = 0.0  # >0 to emulate realtime pacing
MAX_EVENTS: Optional[int] = None  # Stop early after N events (None = no cap)


# --- Helpers ---------------------------------------------------------------

def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x) if x not in (None, "") else None
    except Exception:
        return None


def _row_to_event(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize a CSV/JSONL row to the decider's expected event shape."""
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


essupported_json_gz_exts = {".jsonl", ".json", ".gz"}

def iter_events(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield normalized events from .csv, .jsonl, or .jsonl.gz files."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ev = _row_to_event(row)
                if ev:
                    yield ev
        return

    opener = gzip.open if suffix == ".gz" or path.name.endswith(".jsonl.gz") else open
    with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[arg-type]
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ev = _row_to_event(obj)
            if ev:
                yield ev


# --- Main ------------------------------------------------------------------

def main() -> int:
    cfg = load_metamodel_config()
    datastore = DataStore()
    if CLEAR_DB_ON_START:
        datastore.clear()
    decider = MetaModelDecider(config=cfg, datastore=datastore)

    # Resolve and validate files
    files: List[Path] = [p if p.is_absolute() else (SCRIPT_DIR / p) for p in INPUT_FILES]
    files = [p for p in files if p.exists()]
    if not files:
        print("No input files; nothing to simulate.")
        return 0

    processed = 0
    for fp in files:
        for ev in iter_events(fp):
            decider.evaluate(ev)  # realtime per-event evaluation and persistence
            processed += 1
            if SLEEP_SECONDS and SLEEP_SECONDS > 0:
                sleep(SLEEP_SECONDS)
            if MAX_EVENTS and processed >= MAX_EVENTS:
                break
        if MAX_EVENTS and processed >= MAX_EVENTS:
            break

    # Summary
    warnings_list = list(decider.nightly_warnings())
    by_type: Dict[WarningType, int] = {}
    for w in warnings_list:
        by_type[w.warning_type] = by_type.get(w.warning_type, 0) + 1  # type: ignore[index]

    print(f"Processed events: {processed}")
    if warnings_list:
        print("Warnings summary:")
        for wt, cnt in by_type.items():
            print(f"  {wt.value}: {cnt}")
    else:
        print("Warnings: none")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())