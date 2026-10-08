"""Synthetic event generator & injector for metamodel testing.

Features added:
 - Multiple injections defined in a single INJECTIONS list (clusters, ramps, single alerts)
 - Target severity helper (compute cluster length needed to reach or exceed severity level)
 - Adaptive loudness test support (ramp or plateau) ensuring warmup baseline exists
 - Injection into existing real night log with automatic timestamp collision handling
 - Summary metadata JSON for each generated synthetic scenario & injection file

You can still generate legacy standalone scenario files (set GENERATE_LEGACY_SCENARIOS=True)
but the recommended workflow is to configure INJECTIONS below and run the script.

Severity reminder (cluster): severity = covered_seconds / seconds_per_severity_level
Covered seconds for desired severity S => S * seconds_per_severity_level.
"""

from __future__ import annotations

from pathlib import Path
import json
import random
import datetime as dt
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Any

# ---------------------------------------------------------------------------
# CONFIGURATION SECTION
# ---------------------------------------------------------------------------

GENERATE_LEGACY_SCENARIOS = False  # keep False to skip old single-scenario files

# If injecting into an existing file, define multiple injections here.
# Each injection is relative to the first timestamp of the source night file.
# Types: 'cluster', 'single', 'loudness_ramp', 'loudness_plateau'
#   cluster: repeated dominant label to reach target severity OR explicit length
#   single: one event (e.g., scream)
#   loudness_ramp: gradually ramp loudness value (start->end) across duration
#   loudness_plateau: sustained loudness above baseline for duration
# Common fields:
#   start_offset: seconds after first_ts
#   label: (cluster/single) label name
#   confidence: probability for dominant label (default 0.95)
#   loudness: override loudness per event (cluster/single) or start/end for ramps
#   target_severity: (cluster) compute needed length automatically (overrides length if provided)
#   length: explicit number of seconds for cluster if no target_severity

INJECTIONS: list[dict[str, Any]] = [
    # Example cluster reaching severity 2.0 for horse_kick (weight=2.0, threshold default 10s => need 10s)
    {
        "type": "cluster",
        "label": "horse_kick",
        "start_offset": 3 * 3600 + 180,  # 03:00:00 + 180s
        "target_severity": 2.0,
        "confidence": 0.97,
        "loudness": 0.55
    },
    # A single scream alert
    {
        "type": "single",
        "label": "scream",
        "start_offset": 3 * 3600 + 600,
        "confidence": 0.96,
        "loudness": 0.62
    },
    # Loudness ramp to trigger adaptive loudness after warmup
    {
        "type": "loudness_ramp",
        "start_offset": 5 * 3600,  # after warmup
        "duration": 600,           # 10 minutes
        "start_loudness": 0.18,
        "end_loudness": 1.55,
        "kick_every": 120,         # optional mild kicks every N seconds
        "kick_confidence": 0.75
    },
    # Plateau sustained loudness for 5 minutes
    {
        "type": "loudness_plateau",
        "start_offset": 6 * 3600 + 900,
        "duration": 300,
        "loudness": 1.52,
    },
]

# Path to an existing real log to inject synthetic events into (JSONL)
INJECT_SOURCE_FILE: Path | None = (
    Path(__file__).resolve().parent.parent / "smart-stable-logs" / "segments" / "stable01" / "2025-08-28.jsonl"
)
# Output file with ALL injections applied (single combined file)
COMBINED_INJECTION_OUTPUT: Path = (
    Path(__file__).resolve().parent.parent / "smart-stable-logs" / "segments" / "artificial" / "stable01_2025-08-28_multi_injected.jsonl"
)

# Model / rule parameters (used for computing target cluster length)
CLUSTER_SECONDS_PER_SEVERITY_LEVEL = {
    "horse_kick": 10.0,
    "rattle": 20.0,
    "pawing": 10.0,
    "coughing": 5.0,
    "panting": 100.0,
    "walking": 100.0,
    "change_stance": 100.0,
    "rolling": 20.0,
    "squeal": 1.0,
}

# Random seed (set for reproducibility if desired)
RANDOM_SEED: Optional[int] = 42
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

SCRIPT_PATH = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_PATH.parent / "smart-stable-logs" / "segments" / "artificial"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Base timeline length (seconds) per legacy standalone scenario file
BASE_DURATION = 600

def base_event(ts: dt.datetime) -> dict:
    return {
        "timestamp": ts.strftime('%Y-%m-%d %H:%M:%S'),
        "time_context": {"hour": ts.hour, "is_night": True},
        "loudness_metrics": {"segment": {"value": round(random.uniform(0.05, 0.18), 4)}},
        "label_distribution": {
            "horse_kick": 0.05,
            "rattle": 0.02,
            "other_impact": 0.02,
            "scream": 0.01,
            "whimmering": 0.01,
            "pawing": 0.05,
            "groan": 0.04,
        },
    }

def write_events(name: str, events: list[dict]):
    path = LOGS_DIR / f"synthetic_{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    print(f"Wrote {len(events)} events -> {path}")

def gen_kick_cluster():
    start = dt.datetime(2025, 8, 28, 22, 0, 0)
    events: list[dict] = []
    cluster_start = 200
    cluster_len = 15  # ensures cluster_intensity reaches 1.0
    for i in range(BASE_DURATION):
        t = start + dt.timedelta(seconds=i)
        ev = base_event(t)
        if cluster_start <= i < cluster_start + cluster_len:
            ev["label_distribution"]["horse_kick"] = 0.85  # high prob
            ev["loudness_metrics"]["segment"]["value"] = 0.52
        events.append(ev)
    write_events("kick_cluster", events)

def gen_critical_scream():
    start = dt.datetime(2025, 8, 28, 22, 30, 0)
    events: list[dict] = []
    scream_index = 120
    for i in range(BASE_DURATION):
        t = start + dt.timedelta(seconds=i)
        ev = base_event(t)
        if i == scream_index:
            ev["label_distribution"]["scream"] = 0.95
            ev["loudness_metrics"]["segment"]["value"] = 0.6
        events.append(ev)
    write_events("critical_scream", events)

def gen_groan_cluster():
    start = dt.datetime(2025, 8, 28, 23, 0, 0)
    events: list[dict] = []
    groan_window = range(250, 250 + 6)
    for i in range(BASE_DURATION):
        t = start + dt.timedelta(seconds=i)
        ev = base_event(t)
        if i in groan_window:
            ev["label_distribution"]["groan"] = 0.65  # high groan prob
            ev["loudness_metrics"]["segment"]["value"] = 0.35
        events.append(ev)
    write_events("groan_cluster", events)

def gen_pawing_restlessness():
    start = dt.datetime(2025, 8, 29, 0, 0, 0)
    events: list[dict] = []
    paw_window = range(300, 300 + 10)
    for i in range(BASE_DURATION):
        t = start + dt.timedelta(seconds=i)
        ev = base_event(t)
        if i in paw_window:
            ev["label_distribution"]["pawing"] = 0.75
            ev["loudness_metrics"]["segment"]["value"] = 0.42
        events.append(ev)
    write_events("pawing_restlessness", events)

def gen_mixed_impact_burst():
    start = dt.datetime(2025, 8, 29, 1, 0, 0)
    events: list[dict] = []
    burst_indices = range(180, 180 + 14)
    for i in range(BASE_DURATION):
        t = start + dt.timedelta(seconds=i)
        ev = base_event(t)
        if i in burst_indices:
            # Cycle label focus to create diversity above thresholds
            phase = (i - 180) % 3
            if phase == 0:
                ev["label_distribution"]["horse_kick"] = 0.8
            elif phase == 1:
                ev["label_distribution"]["rattle"] = 0.7
            else:
                ev["label_distribution"]["other_impact"] = 0.82
            ev["loudness_metrics"]["segment"]["value"] = 0.55
        events.append(ev)
    write_events("mixed_impact_burst", events)

def gen_sustained_loudness():
    start = dt.datetime(2025, 8, 29, 2, 0, 0)
    events: list[dict] = []
    ramp_start = 250
    ramp_end = 400
    for i in range(BASE_DURATION):
        t = start + dt.timedelta(seconds=i)
        ev = base_event(t)
        if ramp_start <= i <= ramp_end:
            # Linear ramp loudness 0.2 -> 0.65
            frac = (i - ramp_start) / max(1, (ramp_end - ramp_start))
            ev["loudness_metrics"]["segment"]["value"] = round(0.2 + 0.45 * frac, 4)
            if i % 25 == 0:
                # sporadic moderate kicks
                ev["label_distribution"]["horse_kick"] = 0.7
        events.append(ev)
    write_events("sustained_loudness", events)

def _cluster_length_for_target_severity(label: str, target_severity: float) -> int:
    seconds_per_level = CLUSTER_SECONDS_PER_SEVERITY_LEVEL.get(label, 10.0)
    needed = target_severity * seconds_per_level
    return max(1, int(round(needed)))


def _make_dominant_event(ts: dt.datetime, label: str, confidence: float, loudness: float | None) -> dict:
    ev = base_event(ts)
    # Zero out others (optional: dampen) then set dominant
    for k in list(ev["label_distribution"].keys()):
        ev["label_distribution"][k] = round(ev["label_distribution"][k] * 0.05, 4)
    ev["label_distribution"][label] = confidence
    if loudness is not None:
        ev["loudness_metrics"]["segment"]["value"] = float(round(loudness, 4))
    ev["synthetic"] = True
    ev["synthetic_type"] = "cluster" if label != "scream" else "single"
    ev["dominant_label"] = label
    ev["dominant_confidence"] = confidence
    return ev


def _build_cluster(start_ts: dt.datetime, label: str, length: int, confidence: float, loudness: float | None) -> list[dict]:
    return [_make_dominant_event(start_ts + dt.timedelta(seconds=i), label, confidence, loudness) for i in range(length)]


def _build_single(start_ts: dt.datetime, label: str, confidence: float, loudness: float | None) -> list[dict]:
    return [_make_dominant_event(start_ts, label, confidence, loudness)]


def _build_loudness_ramp(start_ts: dt.datetime, duration: int, start_loudness: float, end_loudness: float,
                         kick_every: int | None, kick_confidence: float) -> list[dict]:
    events: list[dict] = []
    for i in range(duration):
        ts = start_ts + dt.timedelta(seconds=i)
        frac = i / max(1, duration - 1)
        loud = start_loudness + (end_loudness - start_loudness) * frac
        if kick_every and kick_every > 0 and i % kick_every == 0:
            events.extend(_build_single(ts, "horse_kick", kick_confidence, loud))
        else:
            # background breathing/panting
            events.extend(_build_single(ts, "panting", 0.55, loud))
    for ev in events:
        ev["synthetic_type"] = "loudness_ramp"
    return events


def _build_loudness_plateau(start_ts: dt.datetime, duration: int, loudness: float) -> list[dict]:
    events = []
    for i in range(duration):
        ts = start_ts + dt.timedelta(seconds=i)
        events.extend(_build_single(ts, "panting", 0.5, loudness))
    for ev in events:
        ev["synthetic_type"] = "loudness_plateau"
    return events


def perform_multi_injection():
    if INJECT_SOURCE_FILE is None or not INJECT_SOURCE_FILE.exists():
        print(f"Multi-injection skipped (source missing: {INJECT_SOURCE_FILE})")
        return
    with INJECT_SOURCE_FILE.open("r", encoding="utf-8") as f:
        source_events = [json.loads(l) for l in f if l.strip()]
    if not source_events:
        print("Source file empty; aborting multi-injection.")
        return
    # Parse first timestamp (try several formats)
    def _parse_any(ts_str: str) -> dt.datetime:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return dt.datetime.strptime(ts_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unparseable timestamp: {ts_str}")

    first_ts = _parse_any(source_events[0]["timestamp"])  # assume chronological
    injected: list[dict] = []
    meta: list[dict] = []
    for spec in INJECTIONS:
        start_offset = int(spec["start_offset"]) if "start_offset" in spec else 0
        start_ts = first_ts + dt.timedelta(seconds=start_offset)
        typ = spec["type"]
        if typ == "cluster":
            label = spec["label"]
            conf = spec.get("confidence", 0.95)
            loud = spec.get("loudness")
            if spec.get("target_severity") is not None:
                length = _cluster_length_for_target_severity(label, float(spec["target_severity"]))
            else:
                length = int(spec.get("length", 10))
            chunk = _build_cluster(start_ts, label, length, conf, loud)
            meta.append({"type": typ, "label": label, "length": length, "start": start_ts.isoformat()})
        elif typ == "single":
            chunk = _build_single(start_ts, spec["label"], spec.get("confidence", 0.95), spec.get("loudness"))
            meta.append({"type": typ, "label": spec["label"], "start": start_ts.isoformat()})
        elif typ == "loudness_ramp":
            duration = int(spec.get("duration", 300))
            chunk = _build_loudness_ramp(start_ts, duration, spec.get("start_loudness", 0.15), spec.get("end_loudness", 0.55),
                                         spec.get("kick_every"), spec.get("kick_confidence", 0.8))
            meta.append({"type": typ, "duration": duration, "start": start_ts.isoformat()})
        elif typ == "loudness_plateau":
            duration = int(spec.get("duration", 300))
            chunk = _build_loudness_plateau(start_ts, duration, spec.get("loudness", 0.5))
            meta.append({"type": typ, "duration": duration, "start": start_ts.isoformat()})
        else:
            print(f"Unknown injection type '{typ}', skipping.")
            continue
        injected.extend(chunk)

    merged = source_events + injected
    merged.sort(key=lambda e: e["timestamp"])  # stable ordering
    COMBINED_INJECTION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with COMBINED_INJECTION_OUTPUT.open("w", encoding="utf-8") as f:
        for ev in merged:
            f.write(json.dumps(ev) + "\n")
    meta_path = COMBINED_INJECTION_OUTPUT.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as mf:
        json.dump({
            "source_file": str(INJECT_SOURCE_FILE),
            "output_file": str(COMBINED_INJECTION_OUTPUT),
            "total_injected": len(injected),
            "injection_specs": INJECTIONS,
            "meta_summary": meta
        }, mf, indent=2)
    print(f"Multi-injection wrote {len(merged)} events (added {len(injected)}) -> {COMBINED_INJECTION_OUTPUT}")
    print(f"Metadata summary -> {meta_path}")


def main():
    if GENERATE_LEGACY_SCENARIOS:
        gen_kick_cluster()
        gen_critical_scream()
        gen_groan_cluster()
        gen_pawing_restlessness()
        gen_mixed_impact_burst()
        gen_sustained_loudness()
        print("Legacy standalone scenario files generated.")
    perform_multi_injection()

# Remove obsolete single-injection helpers (_parse_timestamp, _scenario_events, inject_into_existing)

if __name__ == "__main__":  # pragma: no cover
    main()