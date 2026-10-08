#!/usr/bin/env python3
"""
Filter the FSD50K dataset down to a chosen set of AudioSet labels and licenses,
and lay the matching clips out as `<out>/<label>/<uploader>/<fname>.wav`.

The per-uploader subfolder is deliberate: it becomes the group_id when these clips
are ingested by `_ingest_external_audio_dir` (training.py), so the same uploader's
clips can't leak across the train/test split.

Stdlib only. Works on the raw FSD50K layout from Zenodo (record 4060432):

    FSD50K/
      FSD50K.ground_truth/{dev.csv,eval.csv}
      FSD50K.metadata/{dev,eval}_clips_info_FSD50K.json
      FSD50K.dev_audio/<fname>.wav
      FSD50K.eval_audio/<fname>.wav

Examples
--------
# Discover which AudioSet labels exist (and how many clips), e.g. to find the
# right names for tractors/airplanes before filtering:
    python filter_fsd50k.py --list-labels | grep -i -E "vehicle|engine|aircraft|tractor"

# Default: extract CC0 dog sounds into <root>/dog_barking/<uploader>/...
    python filter_fsd50k.py

# Add a different negative source later:
    python filter_fsd50k.py --label tractor --match Tractor "Medium_engine_(mid_frequency)"
"""

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

# Default raw-dataset root: apps/ml-backend/data/datasets/FSD50K, resolved relative
# to this script so it works regardless of the current working directory.
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "datasets" / "FSD50K"

# AudioSet label tokens that correspond to a barking/vocalising dog.
DEFAULT_DOG_LABELS = [
    "Bark",
    "Dog",
    "Bow-wow",
    "Growling",
    "Yip",
    "Howl",
    "Whimper_(dog)",
]

def classify_license(value: str) -> str:
    """
    Map a license string to a canonical token. Robust to both Freesound URL forms
    (e.g. http://creativecommons.org/licenses/by/3.0/) and human-readable display
    names (e.g. "Attribution", "Creative Commons 0"), since FSD50K metadata has
    used both over time. Order matters: check the restrictive variants first so
    "Attribution Noncommercial" isn't mistaken for plain "Attribution".
    """
    s = value.strip().lower()
    if not s:
        return ""
    if "publicdomain/zero" in s or "cc0" in s or s in ("creative commons 0", "creativecommons0"):
        return "cc0"
    if "by-nc" in s or "noncommercial" in s or "non-commercial" in s:
        return "by-nc"
    if "sampling" in s:
        return "sampling+"
    if "by-sa" in s or "sharealike" in s or "share alike" in s:
        return "by-sa"
    if "by-nd" in s or "noderiv" in s:
        return "by-nd"
    if "/licenses/by/" in s or s == "attribution":
        return "by"
    return "other"


# Which canonical license tokens each --license mode accepts.
LICENSE_MODES = {
    "cc0": lambda c: c == "cc0",                 # public domain, no attribution
    "ccby": lambda c: c == "by",                 # attribution only
    "commercial": lambda c: c in ("cc0", "by"),  # CC0 or CC-BY
    "any": lambda c: True,
}


def _load_ground_truth(root: Path):
    """Yield (fname, label_tokens, split_name) over dev + eval ground truth."""
    for csv_name, split in (("dev.csv", "dev"), ("eval.csv", "eval")):
        path = root / "FSD50K.ground_truth" / csv_name
        if not path.is_file():
            print(f"WARNING: ground truth not found: {path}", file=sys.stderr)
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                labels = [t for t in str(row.get("labels", "")).split(",") if t]
                yield str(row["fname"]), labels, split


def _load_metadata(root: Path):
    """Return {split: {fname: info_dict}} from the *_clips_info_FSD50K.json files."""
    meta = {}
    for split in ("dev", "eval"):
        path = root / "FSD50K.metadata" / f"{split}_clips_info_FSD50K.json"
        if not path.is_file():
            print(f"WARNING: metadata not found: {path}", file=sys.stderr)
            meta[split] = {}
            continue
        with open(path) as f:
            # keys are fname strings -> {title, description, tags, license, uploader, ...}
            meta[split] = {str(k): v for k, v in json.load(f).items()}
    return meta


def _uploader(info: dict) -> str:
    return info.get("uploader") or info.get("username") or "unknown"


def list_labels(root: Path) -> int:
    counts = Counter()
    for _, labels, _ in _load_ground_truth(root):
        counts.update(labels)
    if not counts:
        print("No ground-truth rows found — is --fsd50k-root correct?", file=sys.stderr)
        return 1
    for label, n in counts.most_common():
        print(f"{n:6d}  {label}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fsd50k-root", type=Path, default=DEFAULT_ROOT,
                   help=f"Raw FSD50K dataset root (default: {DEFAULT_ROOT})")
    p.add_argument("--out", type=Path, default=None,
                   help="Output root; clips go to <out>/<label>/<uploader>/ "
                        "(default: same as --fsd50k-root)")
    p.add_argument("--label", default="dog_barking",
                   help="Target class folder name (default: dog_barking). "
                        "Point DOG_BARK_DIR (or the label's external dir) at <out>/<label>.")
    p.add_argument("--match", nargs="+", default=DEFAULT_DOG_LABELS,
                   help="AudioSet label tokens to include (any-match). "
                        f"Default: {' '.join(DEFAULT_DOG_LABELS)}")
    p.add_argument("--license", choices=sorted(LICENSE_MODES), default="cc0",
                   help="License filter (default: cc0)")
    p.add_argument("--move", action="store_true",
                   help="Move files instead of copying (default: copy)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be extracted without writing anything")
    p.add_argument("--list-labels", action="store_true",
                   help="Print all AudioSet labels with clip counts, then exit")
    p.add_argument("--list-licenses", action="store_true",
                   help="For the matched labels, print the raw license values seen "
                        "(and a sample metadata record), then exit — use this to "
                        "diagnose why a license filter extracts nothing")
    args = p.parse_args()

    root = args.fsd50k_root
    if not root.is_dir():
        print(f"ERROR: --fsd50k-root does not exist: {root}", file=sys.stderr)
        return 2

    if args.list_labels:
        return list_labels(root)

    out_root = args.out or root
    want = set(args.match)
    lic_ok = LICENSE_MODES[args.license]
    audio_dir = {"dev": root / "FSD50K.dev_audio", "eval": root / "FSD50K.eval_audio"}
    meta = _load_metadata(root)

    # Diagnostic mode: show the raw license strings (and the metadata schema) for the
    # matched clips so the cause of an empty result is obvious — a join failure shows
    # up as "<no-metadata-record>", a format surprise shows up as the raw value.
    if args.list_licenses:
        lic_counts: Counter = Counter()
        sample = None
        for fname, labels, split in _load_ground_truth(root):
            if not (want & set(labels)):
                continue
            info = meta.get(split, {}).get(fname, {})
            if sample is None and info:
                sample = info
            raw = str(info.get("license", "")) if info else None
            lic_counts[raw if info else "<no-metadata-record>"] += 1
        print(f"License values among matched clips (labels {sorted(want)}):")
        for val, n in lic_counts.most_common():
            shown = "<empty-license-field>" if val == "" else val
            print(f"{n:6d}  [{classify_license(val or '')}]  {shown}")
        if sample is not None:
            print(f"\nSample metadata record keys: {sorted(sample.keys())}")
        else:
            print("\nNo metadata records joined — check FSD50K.metadata/*_clips_info_FSD50K.json")
        return 0

    n_matched = n_written = 0
    skipped = Counter()
    groups: set[str] = set()

    for fname, labels, split in _load_ground_truth(root):
        if not (want & set(labels)):
            continue
        n_matched += 1

        info = meta.get(split, {}).get(fname, {})
        if not info:
            skipped["no_metadata"] += 1
            continue
        if not lic_ok(classify_license(str(info.get("license", "")))):
            skipped["license"] += 1
            continue

        src = audio_dir[split] / f"{fname}.wav"
        if not src.is_file():
            skipped["audio_missing"] += 1
            continue

        uploader = _uploader(info)
        groups.add(uploader)
        dst_dir = out_root / args.label / uploader
        dst = dst_dir / f"{fname}.wav"

        if args.dry_run:
            n_written += 1
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        if args.move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(src, dst)
        n_written += 1

    verb = "would extract" if args.dry_run else "extracted"
    print(
        f"\nMatched {n_matched} clips for labels {sorted(want)}; "
        f"{verb} {n_written} after license='{args.license}' "
        f"({len(groups)} uploaders/groups)."
    )
    if skipped:
        print("Skipped: " + ", ".join(f"{k}={v}" for k, v in skipped.items()))
    if n_written and not args.dry_run:
        print(f"Output: {out_root / args.label}")
        print(f"Point the ingestion dir at it, e.g.  export DOG_BARK_DIR={out_root / args.label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
