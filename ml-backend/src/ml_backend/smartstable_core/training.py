import os
import logging
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any

from ml_backend.smartstable_core.human_speech_dataset import load_human_speech
from ml_backend.util.label_studio_helper import (
    list_tasks_by_project,
    ls_url_to_local_path,
)
from ml_backend.util.blob_store import BlobStore
from ml_backend.smartstable_core.mlflow_support import (
    SoundscapeTrainingTracker,
    register_model,
    promote_model_to_production,
    evaluate_segments,
)
from smartstablemodel.config import load_metamodel_config
from smartstablemodel.labels import (
    get_label_registry,
    get_training_importance_weights,
    get_label_remapping,
    get_max_samples_per_class_overrides,
)
from smartstablemodel.multi_stall_manager import MultiStallDetectorManager

import hashlib, json, os, tempfile, datetime
from collections import Counter, defaultdict
from pathlib import Path
import mlflow
import pandas as pd
from mlflow.data.pandas_dataset import from_pandas

logger = logging.getLogger("SmartStableTraining")

# Global training state
training_state = {
    "is_training": False,
    "last_training_time": None,
    "training_history": [],
}


def _split_segment_and_id(item: Any) -> tuple[np.ndarray, str | None]:
    """Normalize training segment item to (audio, sample_id)."""
    if (
        isinstance(item, (tuple, list))
        and len(item) == 2
        and isinstance(item[0], np.ndarray)
    ):
        sample_id = item[1] if isinstance(item[1], str) else None
        return item[0], sample_id
    return np.asarray(item, dtype=np.float32), None


def _rebuild_segment_item(
    audio: np.ndarray, sample_id: str | None, original: Any
) -> Any:
    """Return item in same shape as original input item."""
    if isinstance(original, (tuple, list)) and len(original) == 2:
        return (audio, sample_id or "")
    return audio


def _apply_short_event_padding(
    training_segments_by_label: dict[str, list],
    audio_processor,
    background_label: str,
    enabled: bool,
    threshold_seconds: float,
    crossfade_ms: int,
    position_mode: str,
    seed: int | None,
) -> dict[str, int]:
    """
    Compose short non-background events into full windows using background segments.

    Returns stats summary counters.
    """
    stats = {
        "eligible_short": 0,
        "composited": 0,
        "skipped_no_background": 0,
    }

    if not enabled:
        return stats

    if threshold_seconds <= 0:
        logger.warning(
            f"short_event_threshold_seconds={threshold_seconds} is invalid, skipping short-event padding"
        )
        return stats

    threshold_samples = int(threshold_seconds * audio_processor.sr)
    if threshold_samples <= 0:
        logger.warning(
            f"short_event_threshold_seconds={threshold_seconds} produces 0 samples, skipping"
        )
        return stats

    rng = np.random.default_rng(seed)

    background_pool = []
    for bg_item in training_segments_by_label.get(background_label, []):
        bg_audio, _ = _split_segment_and_id(bg_item)
        background_pool.append(bg_audio)

    if not background_pool:
        stats["skipped_no_background"] = sum(
            len(items)
            for label, items in training_segments_by_label.items()
            if label != background_label
        )
        logger.warning(
            f"Short-event padding enabled but no '{background_label}' background segments found. Skipping compositing."
        )
        return stats

    for label, items in training_segments_by_label.items():
        if label == background_label:
            continue

        transformed_items = []
        label_eligible_short = 0
        label_composited = 0
        for item in items:
            audio, sample_id = _split_segment_and_id(item)
            from_short_coverage = bool(sample_id and "#shortcov" in sample_id)
            is_short_by_length = len(audio) < threshold_samples
            if from_short_coverage or is_short_by_length:
                stats["eligible_short"] += 1
                label_eligible_short += 1
                composited = audio_processor.compose_short_event_with_background(
                    short_event=audio,
                    background_pool=background_pool,
                    target_duration_seconds=audio_processor.win_seconds,
                    crossfade_ms=crossfade_ms,
                    position_mode=position_mode,
                    rng=rng,
                )
                new_id = f"{sample_id}#sepad" if sample_id else None
                transformed_items.append(
                    _rebuild_segment_item(composited, new_id, item)
                )
                stats["composited"] += 1
                label_composited += 1
            else:
                transformed_items.append(item)

        training_segments_by_label[label] = transformed_items
        if label_eligible_short > 0:
            logger.info(
                f"Short-event padding label='{label}': eligible={label_eligible_short}, composited={label_composited}"
            )

    return stats


## ---------- Dataset creation utilities ----------


def get_group_id_and_provenance(task):
    data = task.data if hasattr(task, "data") else task
    tid = str(task.id if hasattr(task, "id") else data.get("id", "unknown"))

    source = (data.get("source") or "").strip()
    if (
        source and source.lower() != "youtube"
    ):  # a SPECIFIC source, e.g. "youtube:vidABC"
        provenance = source.split(":")[0] if ":" in source else "external"
        return source, provenance  # each distinct source string = its own group
    if source.lower() == "youtube":
        # flat "youtube" can't distinguish videos — fall back to task so they
        # don't all collapse into one group (and log it so you fix the data)
        logger.warning(
            f"Task {tid}: source='youtube' is not per-video; "
            f"grouping by task. Set source to 'youtube:<video_id>'."
        )
        return f"task:{tid}", "youtube"
    # stable recording: group by the recording session = the task
    return f"task:{tid}", "stable_mic"


def determine_blob_metadata(
    task, annotation: List[Dict[str, Any]], accepted_labels: List[str]
):
    blobs_metadata = []
    group_id, provenance = get_group_id_and_provenance(task)
    source_file = task.data.get("audio", "") if hasattr(task, "data") else ""
    source_file = ls_url_to_local_path(source_file)

    for result_item in annotation or []:
        val = result_item.get("value", {})
        labels = val.get("labels", []) or []
        start_time = val.get("start")
        end_time = val.get("end")
        if start_time is None or end_time is None:
            logger.warning(f"Skipping annotation with missing start/end: {result_item}")
            continue

        for raw_label in labels:
            if accepted_labels and raw_label not in accepted_labels:
                logger.info(
                    f"Skipping label '{raw_label}' not in accepted labels: {accepted_labels}"
                )
                continue
            blobs_metadata.append(
                {
                    "group_id": group_id,
                    "provenance": provenance,
                    "source_file": source_file,
                    "start": start_time,
                    "end": end_time,
                    "label": raw_label,
                }
            )
    return blobs_metadata


def _interval_key(start, end, sr):
    # quantize to samples so float jitter can't fork the key
    return (round(start * sr), round(end * sr))


def save_blobmap(blobmap_path, blobmap):
    d = os.path.dirname(blobmap_path)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(blobmap, f, indent=2)
    os.replace(tmp, blobmap_path)  # atomic; never a half-written index


def _ingest_external_audio_dir(
    new_blobmap,
    blob_store,
    manager: "MultiStallDetectorManager",
    extract_labels,
    *,
    label: str,
    env_var: str,
    default_dir: str,
    provenance: str,
    group_prefix: str,
):
    """
    Ingest a bulk public dataset (one label, one folder) as normal content-addressed
    blobs so it joins the dataset exactly like Label-Studio segments. Used for classes
    that have no GDPR constraint and are sourced in bulk rather than annotated in LS
    (e.g. dog_barking). Mutates `new_blobmap` in place; does nothing if the class is
    not trainable in the current config or the folder is absent.

    Each audio file under the folder becomes one full-clip segment. group_id is taken
    from the file's immediate subdirectory under the root, so you can keep clips from
    the same recording/source in one subfolder to prevent train/test leakage; files
    placed directly in the root are grouped per-file.
    """
    # `extract_labels` already includes merge sources whose target is trainable, so a
    # class like dog_barking (merge_into="other") is present here even though it's
    # excluded from the raw trainable set.
    if extract_labels and label not in extract_labels:
        logger.info(f"{label} ingestion skipped: not a trainable/merged label")
        return

    root = Path(os.getenv(env_var, default_dir))
    if not root.is_dir():
        logger.info(f"{label} ingestion skipped: {root} not found (set {env_var})")
        return

    sr = manager.audio_processor.sr
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".aac"}
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)
    if not files:
        logger.warning(f"{label} ingestion: no audio files under {root}")
        return

    bucket = []
    n_ok = n_fail = 0
    for path in files:
        try:
            full_audio, got_sr = manager.audio_processor.load_full(str(path))
            if got_sr != sr:
                raise RuntimeError(f"SR mismatch: got {got_sr}, expected {sr}")
            duration = len(full_audio) / float(sr)
            if duration <= 0:
                raise RuntimeError("empty audio")
            blob_hash_val, audio = manager.audio_processor.slice_and_hash(
                full_audio, sr, 0.0, duration
            )
            blob_store.write(blob_hash_val, audio)  # no-op if bytes already present

            rel = path.relative_to(root)
            group_key = rel.parts[0] if len(rel.parts) > 1 else path.stem
            bucket.append(
                {
                    "group_id": f"{group_prefix}:{group_key}",
                    "provenance": provenance,
                    "source_file": str(path),
                    "start": 0.0,
                    "end": duration,
                    "label": label,
                    "hash": blob_hash_val,
                }
            )
            n_ok += 1
        except Exception as e:
            logger.debug(f"{label} decode failed {path.name}: {e}")
            n_fail += 1

    if bucket:
        # one synthetic task bucket; the split is driven by group_id, not task id.
        new_blobmap[f"external:{label}"] = bucket
        n_groups = len({b["group_id"] for b in bucket})
        logger.info(
            f"{label} ingested: {n_ok} clips ({n_groups} groups) from {root}"
            + (f", {n_fail} failed to decode" if n_fail else "")
        )


def update_dataset_blobs(dataset_tasks, manager: "MultiStallDetectorManager"):
    dataset_path = os.getenv("DATASETS_PATH", "/datasets")
    blobmap_path = os.path.join(dataset_path, "blobmap.json")
    blob_store = BlobStore(os.path.join(dataset_path, "blobs"))
    sr = manager.audio_processor.sr

    # --- load OLD blobmap as a hash CACHE only (not as the base to mutate) ----
    # keyed by (task_id, interval_key) -> existing entry (carries its hash).
    # Lets us skip decoding for unchanged segments without carrying deletions forward.
    hash_cache = {}
    if os.path.isfile(blobmap_path):
        try:
            with open(blobmap_path, "r") as f:
                old = json.load(f)
            for tid, segs in old.items():
                for b in segs:
                    if "hash" in b:
                        hash_cache[(tid, _interval_key(b["start"], b["end"], sr))] = b
            logger.info(f"Loaded {len(hash_cache)} cached segment hashes")
        except Exception as e:
            logger.warning(f"Failed to load blobmap cache from {blobmap_path}: {e}")

    trainable = set(get_label_registry().get_trainable_label_values_for_classifier())
    remap = get_label_registry().get_label_remapping()
    # extract anything trainable OR anything that merges into a trainable label
    extract_labels = trainable | {src for src, dst in remap.items() if dst in trainable}
    logger.info(
        f"Accepted labels for blob creation (includes merged labels): {extract_labels}"
    )

    # --- build a FRESH blobmap from current LS state --------------------------
    new_blobmap = {}  # <- the rebuild; deletions vanish because we never copy them in

    tasks_processed = 0
    tasks_skipped_no_audio = 0
    tasks_skipped_no_annotations = 0
    total_segments_extracted = 0
    segments_reused = 0
    total_tasks = len(dataset_tasks)

    for task in dataset_tasks:
        audio_url = task.data.get("audio", "")
        if not audio_url:
            tasks_skipped_no_audio += 1
            continue
        audio_path = ls_url_to_local_path(audio_url)
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            tasks_skipped_no_audio += 1
            continue

        task_annotations = []
        if hasattr(task, "annotations") and task.annotations:
            for ann in task.annotations:
                if isinstance(ann, dict) and "result" in ann:
                    task_annotations.extend(ann.get("result", []))
        if not task_annotations:
            tasks_skipped_no_annotations += 1
            continue

        task_id = str(task.id)
        segments = determine_blob_metadata(task, task_annotations, extract_labels)

        bucket = []  # fresh list for this task — NOT setdefault on old map
        full_audio = None

        for seg in segments:
            key = _interval_key(seg["start"], seg["end"], sr)
            cached = hash_cache.get((task_id, key))

            if cached is not None:
                # segment unchanged interval -> reuse hash, no decode.
                # take the CURRENT label from seg (handles relabels), cached hash.
                seg["hash"] = cached["hash"]
                bucket.append(seg)
                segments_reused += 1
                continue

            # genuinely new segment (or new interval) -> decode once per task
            if full_audio is None:
                full_audio, got_sr = manager.audio_processor.load_full(
                    seg["source_file"]
                )
                if got_sr != sr:
                    raise RuntimeError(
                        f"SR mismatch: processor returned {got_sr}, expected {sr}."
                    )
            blob_hash_val, audio = manager.audio_processor.slice_and_hash(
                full_audio, sr, seg["start"], seg["end"]
            )
            seg["hash"] = blob_hash_val
            blob_store.write(blob_hash_val, audio)  # no-op if bytes already on disk
            bucket.append(seg)
            total_segments_extracted += 1

        if bucket:  # only keep tasks that actually have segments
            new_blobmap[task_id] = bucket
        tasks_processed += 1
        if tasks_processed % 10 == 0 or tasks_processed == total_tasks:
            logger.info(
                f"Processed {tasks_processed}/{total_tasks} tasks "
                f"({total_segments_extracted} new, {segments_reused} reused)"
            )

    # Fold in bulk external-folder classes (no GDPR constraint, so they're stored as
    # normal blobs) AFTER the LS rebuild, so they survive the from-scratch rebuild and
    # flow through the same manifest / split / training path as everything else.
    _ingest_external_audio_dir(
        new_blobmap,
        blob_store,
        manager,
        extract_labels,
        label="dog_barking",
        env_var="DOG_BARK_DIR",
        default_dir="/datasets/FSD50K/dog_barking",
        provenance="dog_bark",
        group_prefix="dog",
    )

    save_blobmap(blobmap_path, new_blobmap)  # atomically overwrite with the fresh map

    logger.info(
        f"Done: {tasks_processed} tasks processed, "
        f"{tasks_skipped_no_audio} skipped (no audio), "
        f"{tasks_skipped_no_annotations} skipped (no annotations), "
        f"{total_segments_extracted} new segments, {segments_reused} reused. "
        f"Blobmap now has {len(new_blobmap)} tasks "
        f"(was {len(hash_cache)} cached segments)."
    )
    return new_blobmap


cfg = load_metamodel_config().from_toml()
SCHEMA_VERSION = cfg.version  # bump when label schema changes; recorded per line
logger.info(f"Using schema version: {SCHEMA_VERSION}")
if not SCHEMA_VERSION or SCHEMA_VERSION == "ERROR":
    raise ValueError("Invalid schema version")


def _group_in_test(group_id, seed, test_pct):
    h = hashlib.md5(f"{seed}:{group_id}".encode()).hexdigest()
    return (int(h, 16) % 100) < test_pct


def _auto_allocate_rare_groups(blobmap, rare_labels, seed, test_split, min_test=1):
    """
    For scarce/external classes, guarantee >=min_test carrier-groups land in test,
    deterministically. Returns {group_id: "test"|"train"} to feed as overrides.
    Logs every assignment so a 7-group split is fully inspectable.
    """
    # collect the distinct groups that carry each rare label
    groups_for_label = defaultdict(set)
    for segments in blobmap.values():
        for seg in segments:
            if seg.get("label") in rare_labels and "hash" in seg:
                groups_for_label[seg["label"]].add(seg["group_id"])

    allocation = {}
    for label, groups in groups_for_label.items():
        groups = sorted(groups)  # stable order
        n = len(groups)
        n_test = max(min_test, round(n * test_split))
        n_test = min(n_test, n - 1) if n > 1 else n  # leave >=1 in train if possible
        # deterministic pick: rank by hash(seed:group), take the first n_test for test
        ranked = sorted(
            groups, key=lambda g: hashlib.md5(f"{seed}:{g}".encode()).hexdigest()
        )
        for i, g in enumerate(ranked):
            allocation[g] = "test" if i < n_test else "train"
        logger.info(
            f"Rare label '{label}': {n} carrier groups -> "
            f"{n_test} test, {n - n_test} train. "
            f"test={[g for g in ranked[:n_test]]}"
        )
    return allocation


def write_manifest(
    blobmap,
    test_split,
    seed,
    version,
    test_group_overrides=None,
    dataset_path=None,
    comment=None,
):
    """
    Freeze a versioned split. blobmap is mutable+growing; the manifest is immutable.
    test_group_overrides: {"group_id": "test"|"train"} — rare-class carriers you place
                          by hand; these win over the hash.
    """
    dataset_path = dataset_path or os.getenv("DATASETS_PATH", "/datasets")
    overrides = test_group_overrides or {}
    test_pct = int(round(test_split * 100))

    manifest_dir = Path(dataset_path) / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    if version == "new" or version is None:
        existing = sorted(
            int(p.stem[1:])
            for p in manifest_dir.glob("v*.jsonl")
            if p.stem[1:].isdigit()
        )
        version = f"v{(existing[-1] + 1) if existing else 1}"

    min_label_count = 80  # if a label appears in fewer than this many segments, consider it "rare" for group-level allocation purposes
    counts = Counter(
        seg["label"] for segs in blobmap.values() for seg in segs if "hash" in seg
    )
    rare_labels = {lab for lab, n in counts.items() if 0 < n < min_label_count}

    logger.info(f"Label counts: {dict(counts)}")
    logger.info(f"Rare labels: {rare_labels}")

    auto = _auto_allocate_rare_groups(blobmap, rare_labels, seed, test_split)
    overrides = {**auto, **(test_group_overrides or {})}  # manual wins on conflict

    eval_only_labels = set(get_label_registry().get_eval_only_labels())
    logger.info(f"Evaluation-only labels (excluded from training): {eval_only_labels}")

    # --- registry-driven carrier allocation (per-label split policy) ----------
    # Reads min_train_carrier_fraction from the label config (smart_stable_model_config.toml).
    # For such a label, force >= that fraction of its CARRIER GROUPS into train,
    # deterministically. This OVERRIDES the generic rare-group auto-allocation,
    # because it's an explicit, recorded policy rather than a count heuristic.
    registry = get_label_registry()
    policy_overrides = {}
    for label in counts:  # only labels that actually have extracted blobs
        frac = registry.get_min_train_carrier_fraction(label)  # None if unset
        if not frac:
            continue
        # distinct groups carrying this label
        carrier_groups = sorted(
            {
                seg["group_id"]
                for segs in blobmap.values()
                for seg in segs
                if seg.get("label") == label and "hash" in seg
            }
        )
        n = len(carrier_groups)
        if n == 0:
            continue
        n_train = max(1, int(round(n * frac)))
        n_train = min(n_train, n)  # never more than we have
        # deterministic pick: rank by hash(seed:group), first n_train -> train
        ranked = sorted(
            carrier_groups,
            key=lambda g: hashlib.md5(f"{seed}:{g}".encode()).hexdigest(),
        )
        for i, g in enumerate(ranked):
            policy_overrides[g] = "train" if i < n_train else "test"
        logger.info(
            f"Carrier policy '{label}': {n} groups, frac={frac} -> "
            f"{n_train} train / {n - n_train} test. "
            f"train_groups={ranked[:n_train]}"
        )

    # precedence: generic auto (weakest) < registry policy < manual override (strongest)
    overrides = {**auto, **policy_overrides, **(test_group_overrides or {})}

    # 1. assign every group to a split (override first, else deterministic hash)
    group_split = {}
    for task_id, segments in blobmap.items():
        for seg in segments:
            g = seg["group_id"]
            if g in group_split:
                continue
            if g in overrides:
                group_split[g] = overrides[g]
            else:
                group_split[g] = (
                    "test" if _group_in_test(g, seed, test_pct) else "train"
                )

    # 2. emit one line per blob, carrying its group's split + provenance
    lines = []
    per_class = defaultdict(lambda: {"train": 0, "test": 0})
    groups_per_class = defaultdict(lambda: {"train": set(), "test": set()})
    for task_id, segments in blobmap.items():
        for seg in segments:
            if "hash" not in seg:  # never extracted (skipped label, error) — exclude
                continue
            is_eval_only = seg["label"] in eval_only_labels
            split = "test" if is_eval_only else group_split[seg["group_id"]]
            per_class[seg["label"]][split] += 1
            groups_per_class[seg["label"]][split].add(seg["group_id"])
            lines.append(
                {
                    "hash": seg["hash"],
                    "label": seg["label"],  # raw label — merge is a train-time view
                    "group_id": seg["group_id"],
                    "provenance": seg["provenance"],
                    "split": split,
                    "eval_only": is_eval_only,  # proxy sample — report separately, never train
                    "source_task": task_id,
                    "start": seg["start"],
                    "end": seg["end"],
                    "schema_version": SCHEMA_VERSION,
                }
            )

    # 3. write manifest (atomic) + a sidecar header describing how it was cut
    manifest_path = manifest_dir / f"{version}.jsonl"
    fd, tmp = tempfile.mkstemp(dir=manifest_dir, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")
    os.replace(tmp, manifest_path)

    header = {
        "version": version,
        "seed": seed,
        "test_split": test_split,
        "schema_version": SCHEMA_VERSION,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "comment": comment,
        "overrides": overrides,
        "n_blobs": len(lines),
        "per_class_counts": dict(per_class),  # thin classes visible at a glance
        "groups_per_class": {
            k: {"train": len(v["train"]), "test": len(v["test"])}
            for k, v in groups_per_class.items()
        },
    }
    with open(manifest_dir / f"{version}.header.json", "w") as f:
        json.dump(header, f, indent=2)

    logger.info(
        f"Wrote manifest {version}: {len(lines)} blobs. "
        f"Per-class train/test: {dict(per_class)}"
    )
    return manifest_path


def create_dataset(
    project_id,
    manager,
    test_split=0.2,
    seed=42,
    version="new",
    test_group_overrides=None,
    comment=None,
):
    try:
        dataset_tasks = list_tasks_by_project(
            project_id, only_annotated=True, exclude_test_tasks=True
        )
        logger.info(f"Fetched {len(dataset_tasks)} annotated tasks")

        # Step 1: idempotent accumulation — slow, safe to repeat
        blobmap = update_dataset_blobs(dataset_tasks, manager)

        # Step 2: deliberate frozen cut — fast, named, versioned
        manifest_path = write_manifest(
            blobmap,
            test_split=test_split,
            seed=seed,
            version=version,
            test_group_overrides=test_group_overrides,
            comment=comment,
        )
        return {"manifest": str(manifest_path)}
    except Exception as e:
        logger.error(f"Dataset creation failed for {project_id}: {e}", exc_info=True)
        return False


def _window_segment(audio, target_len, overlap=0.0):
    """
    Turn one clip into a list of exactly-target_len windows.
      - shorter than a window  -> one right-padded window
      - exactly a window       -> itself
      - longer                 -> tiled windows; `overlap` in [0,1).
    overlap=0.0 => non-overlapping tiles (independent samples, the default).
    """
    import numpy as np

    n = audio.shape[0]
    if n <= target_len:
        if n == target_len:
            return [audio]
        out = np.zeros(target_len, dtype=np.float32)
        out[:n] = audio
        return [out]

    step = max(1, int(round(target_len * (1.0 - overlap))))
    windows = []
    start = 0
    while start + target_len <= n:
        windows.append(audio[start : start + target_len])
        start += step
    # tail: catch a remainder >= half a window so you don't silently drop the
    # end of an event (right-align it so it ends where the audio ends)
    if start < n and (n - start) >= target_len // 2:
        windows.append(audio[n - target_len : n])
    return windows


def _fit_to_window(audio, target_len):
    """Make a clip exactly target_len samples. Deterministic input-contract
    normalization (NOT augmentation) — applies to train, test, and eval alike.
    Long -> center-crop; short -> right-pad with silence."""
    import numpy as np

    n = audio.shape[0]
    if n == target_len:
        return audio
    if n > target_len:
        start = (n - target_len) // 2
        return audio[start : start + target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[:n] = audio
    return out


def _load_manifest_segments(
    manifest_path, blob_store, accepted_labels, label_remapping, eval_only_labels
):
    """
    Build train/test/eval_only segment dicts FROM the frozen manifest.
    Split comes from the manifest's `split` field — NOT random. That's the
    reproducibility guarantee: same manifest version -> byte-identical split.

    Merge (raw -> merged label) is applied HERE at load time, so the manifest
    stays raw and one manifest serves both merged and unmerged experiments.
    """
    remap = label_remapping or {}
    eval_only = set(eval_only_labels or [])

    train_by_label = defaultdict(list)
    test_by_label = defaultdict(list)
    eval_only_by_label = defaultdict(list)

    lines = []
    with open(manifest_path) as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                lines.append(json.loads(raw))

    missing = 0
    for ln in lines:
        raw = ln["label"]
        label = remap.get(raw, raw)  # apply merge view
        if accepted_labels and label not in accepted_labels and label not in eval_only:
            continue

        try:
            audio = blob_store.load(ln["hash"])
        except FileNotFoundError:
            missing += 1
            continue

        # segment carries the blob hash as its sample id — content-addressed,
        # stable across runs, traceable straight back to the manifest line
        seg = (audio, ln["hash"])

        if ln.get("eval_only"):
            eval_only_by_label[label].append(seg)
        elif ln["split"] == "test":
            test_by_label[label].append(seg)
        else:
            train_by_label[label].append(seg)

    blob_group = {ln["hash"]: ln["group_id"] for ln in lines}

    if missing:
        logger.warning(
            f"{missing} manifest blobs missing on disk — re-run blob creation?"
        )

    return (
        dict(train_by_label),
        dict(test_by_label),
        dict(eval_only_by_label),
        lines,
        blob_group,
    )


def _blob_store_training(
    manifest_version: str,
    multi_stall_manager: MultiStallDetectorManager,
    mlflow_tracker=None,
    model_architecture="panns",
    seed=42,
):
    """
    Train from a frozen manifest version + content-addressed blobs.
    Parallel to _label_studio_training, but: no re-extraction (loads blobs),
    no random split (reads manifest `split`), and logs the dataset digest to
    MLflow (fills the `Datasets = None` field). Augmentation stays TRAIN-only.
    """
    global training_state
    if multi_stall_manager is None:
        return [], 500

    datasets_path = os.getenv("DATASETS_PATH", "/datasets")
    manifest_dir = os.path.join(datasets_path, "manifests")
    manifest_path = os.path.join(manifest_dir, f"{manifest_version}.jsonl")
    if not os.path.isfile(manifest_path):
        logger.error(f"Manifest not found: {manifest_path}")
        return [], 404
    blob_store = BlobStore(os.path.join(datasets_path, "blobs"))

    tracker_created_here = False

    try:
        training_state["is_training"] = True
        training_state["last_training_time"] = datetime.datetime.now()
        logger.info(f"Starting blob-store training from manifest {manifest_version}")

        accepted_labels = (
            get_label_registry().get_trainable_label_values_for_classifier()
        )
        excluded_labels = (
            get_label_registry().get_labels_to_exclude_from_training_for_classifier()
        )
        eval_only_labels = set(
            getattr(get_label_registry(), "get_eval_only_labels", lambda: [])()
        )

        from smartstablemodel.labels import get_label_remapping

        label_remapping = get_label_remapping()

        # ---- DATA: from manifest, not from re-extraction; split is frozen -------
        # blob_group: {hash -> group_id}, the side map the invariant uses to read
        # each blob segment's group without stuffing group_id into the audio tuple.
        (
            training_set_by_label,
            test_set_by_label,
            eval_only_by_label,
            manifest_lines,
            blob_group,
        ) = _load_manifest_segments(
            manifest_path,
            blob_store,
            accepted_labels,
            label_remapping,
            eval_only_labels,
        )

        # --- external-source class: human_talking --------------------------------
        # GDPR: stable human voices are never stored as blobs. human_talking is
        # sourced entirely from public Common Voice and split speaker-disjoint, so
        # it flows through the same invariant / train / eval path as blob classes.
        if "human_talking" in accepted_labels:
            ht_train, ht_test = load_human_speech(
                data_dir=os.getenv("HUMAN_SPEECH_DIR", "/datasets/people"),
                win_seconds=multi_stall_manager.audio_processor.win_seconds,
                languages=(
                    "fi",
                ),  # Finnish only: full corpus w/ official split (en is delta-only)
                max_per_language=400,
                seed=seed,
            )
            if ht_train and ht_test:
                training_set_by_label["human_talking"] = ht_train
                test_set_by_label["human_talking"] = ht_test
                logger.info(
                    f"human_talking injected: {len(ht_train)} train / {len(ht_test)} test"
                )
            else:
                logger.warning(
                    "human_talking requested but no speech data loaded — "
                    "invariant will flag it if it's a trained class"
                )

        for label, segs in training_set_by_label.items():
            logger.info(f"  train {label}: {len(segs)}")
        for label, segs in test_set_by_label.items():
            logger.info(f"  test  {label}: {len(segs)}")
        for label, segs in eval_only_by_label.items():
            logger.info(f"  eval-only(proxy) {label}: {len(segs)}")

        if not training_set_by_label:
            logger.warning("No training segments in manifest")
            return [], 400

        # ---- config + hyperparams -----------------------------------------------
        CONFIG_BASE_DIR = os.path.abspath(
            os.getenv(
                "SMARTSTABLE_CONFIG_DIR",
                os.path.join(
                    os.path.dirname(__file__), "..", "smartstablemodel", "config"
                ),
            )
        )
        with open(os.path.join(CONFIG_BASE_DIR, "training_settings.toml"), "rb") as f:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            training_config = tomllib.load(f)
        sd_config = training_config.get("soundscape_detector", {})
        learning_rate = float(sd_config.get("learning_rate", 1e-4))
        epochs = sd_config.get("num_epochs", 10)
        batch_size = sd_config.get("batch_size", 32)
        dropout_rates = sd_config.get("dropout_rates", None)
        max_samples = sd_config.get("max_samples_per_class", None)
        max_samples_overrides = get_max_samples_per_class_overrides()
        augmentation_factor = sd_config.get("augmentation_factor", 2)
        training_metric = sd_config.get("training_metric", "accuracy")
        label_weights = get_training_importance_weights()
        min_samples = sd_config.get("min_samples_per_class", 20)

        if max_samples_overrides:
            logger.info(
                f"Using per-label max_samples_per_class overrides: {max_samples_overrides}"
            )

        # ---- short-event padding: TRAIN ONLY (never touches test) ---------------
        target_len = int(
            round(
                multi_stall_manager.audio_processor.win_seconds
                * multi_stall_manager.audio_processor.sr
            )
        )
        bg_label = next(
            (
                l
                for l in [
                    sd_config.get("short_event_background_label"),
                    "background",
                    "normal",
                ]
                if l and l in training_set_by_label
            ),
            "background",
        )
        _apply_short_event_padding(
            training_set_by_label,
            audio_processor=multi_stall_manager.audio_processor,
            background_label=bg_label,
            enabled=bool(sd_config.get("short_event_padding_enabled", True)),
            threshold_seconds=float(
                sd_config.get("short_event_threshold_seconds", 2.0)
            ),
            crossfade_ms=int(sd_config.get("short_event_crossfade_ms", 40)),
            position_mode=str(
                sd_config.get("short_event_position_mode", "random")
            ).lower(),
            seed=sd_config.get("short_event_seed", None),
        )

        def _normalize_split(d, target_len, overlap=0.0, multi_window=False):
            for label, segs in d.items():
                out = []
                for s in segs:
                    audio, sid = (s[0], s[1]) if isinstance(s, tuple) else (s, None)
                    if multi_window:
                        wins = _window_segment(audio, target_len, overlap=overlap)
                        for j, w in enumerate(wins):
                            new_id = (
                                f"{sid}#w{j}" if sid else None
                            )  # keep ids unique per window
                            out.append((w, new_id) if sid is not None else w)
                    else:
                        w = _fit_to_window(audio, target_len)  # one window, center-crop
                        out.append((w, sid) if sid is not None else w)
                d[label] = out

        # training: tile long events into independent windows (inflates sample counts)
        _normalize_split(
            training_set_by_label, target_len, overlap=0.5, multi_window=True
        )
        # test / eval: one window per event -> honest per-event metric
        _normalize_split(test_set_by_label, target_len, multi_window=False)
        _normalize_split(eval_only_by_label, target_len, multi_window=False)

        # --- group-aware class invariant -----------------------------------------
        # MUST run AFTER normalization: training counts WINDOWS (post-tiling), so the
        # min_samples gate has to see the same post-tiling counts. The group check is
        # tiling-invariant (windows inherit their event's group), so test-group counts
        # are unaffected by the move. Shares the split's source of truth: a class with
        # min_train_carrier_fraction set is allowed to have 0 test groups (measured by
        # proxy/LORO); any other class with 0 test groups is a hard error.
        registry = get_label_registry()

        def _seg_group(seg, blob_group):
            sid = seg[1] if isinstance(seg, tuple) and len(seg) >= 2 else None
            if isinstance(sid, str) and sid.startswith("cv:"):
                return sid  # speech: id IS the speaker-group
            if isinstance(sid, str):
                base = sid.split("#")[0]  # strip #w{j}/#sepad suffixes -> original hash
                return blob_group.get(base)
            return None

        def _groups(split_dict, label, blob_group):
            return {
                g
                for s in split_dict.get(label, [])
                if (g := _seg_group(s, blob_group)) is not None
            }

        trained_classes = {
            lab
            for lab, segs in training_set_by_label.items()
            if len(segs) >= min_samples
        }
        eval_only_present = set(eval_only_by_label.keys())

        errors, warnings_ = [], []
        for c in sorted(trained_classes):
            test_groups = _groups(test_set_by_label, c, blob_group)
            train_groups = _groups(training_set_by_label, c, blob_group)
            frac = registry.get_min_train_carrier_fraction(c)  # None unless policy set

            if len(test_groups) >= 1:
                continue  # measurable — fine

            # 0 test groups: policy-permitted, or a real coverage gap?
            if frac is not None and frac >= 1.0:
                warnings_.append(
                    f"  '{c}': 0 test groups BY POLICY (min_train_carrier_fraction={frac}). "
                    f"No frozen-test metric; relies on eval_only proxy / LORO-CV."
                )
            elif frac is not None:
                warnings_.append(
                    f"  '{c}': policy frac={frac} but only {len(train_groups)} group(s) total, "
                    f"none left for test. Add source recordings or lower the fraction."
                )
            else:
                errors.append(
                    f"  '{c}': trained on {len(train_groups)} group(s) but 0 test groups. "
                    f"-> reserve a test group, or set use_for_classifier_training=false."
                )

        # tested-but-not-trained (excluding eval_only proxies) is always an error
        for c in sorted(set(test_set_by_label) - trained_classes - eval_only_present):
            errors.append(
                f"  '{c}': has test data but isn't trained (min_samples={min_samples} "
                f"or excluded). -> remove from test or add training data."
            )

        if warnings_:
            logger.warning(
                "Class coverage warnings (policy-permitted, no frozen metric):\n"
                + "\n".join(warnings_)
            )
        if errors:
            msg = (
                "Train/test class invariant violated — refusing to train.\n"
                + "\n".join(errors)
            )
            logger.error(msg)
            raise ValueError(msg)
        logger.info(
            f"Class invariant OK: {len(trained_classes)} trained classes, "
            f"all measurable or policy-exempted."
        )

        # ---- tracker ------------------------------------------------------------
        if mlflow_tracker is None:
            mlflow_tracker = SoundscapeTrainingTracker(
                experiment_name="soundscape_classifier"
            )
            tracker_created_here = True
        tags = {
            "training_source": "blob_store",
            "manifest_version": manifest_version,
            "model_architecture": model_architecture,
            "evaluation_type": "frozen_manifest",  # real held-out split, unlike webhook path
        }
        mlflow_tracker.start_training_run(
            training_segments_by_label=training_set_by_label,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            tags=tags,
            dropout_rates=dropout_rates,
            min_samples_per_class=min_samples,
            max_samples_per_class=max_samples,
            augmentation_factor=augmentation_factor,
            label_remapping=label_remapping,
            exclude_labels=excluded_labels,
            training_metric=training_metric,
            label_weights=label_weights,
            run_name=f"train_{model_architecture}_{manifest_version}",
            audio_backend=getattr(
                multi_stall_manager.audio_processor, "backend", "unknown"
            ),
        )

        # ---- fills `Datasets = None`: log dataset refs + digests ----------------
        try:
            df = pd.DataFrame(manifest_lines)
            if "split" in df.columns:
                mlflow.log_input(
                    from_pandas(
                        df[df["split"] == "test"],
                        source=manifest_path,
                        name=f"tt_{manifest_version}_test",
                    ),
                    context="testing",
                )
                mlflow.log_input(
                    from_pandas(
                        df[df["split"] == "train"],
                        source=manifest_path,
                        name=f"tt_{manifest_version}_train",
                    ),
                    context="training",
                )
                logger.info(
                    "Logged dataset inputs to MLflow (digest enables run comparison)"
                )
            else:
                logger.warning(
                    "Manifest has no 'split' column — skipping dataset logging"
                )
        except Exception as e:
            logger.warning(f"log_input failed (non-fatal): {e}")

        # ---- train (augmentation happens HERE, train set only) ------------------
        try:
            history = multi_stall_manager.train_soundscape_classifier_multiclass(
                training_set_by_label,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                dropout_rates=dropout_rates,
                min_samples_per_class=min_samples,
                max_samples_per_class=max_samples,
                max_samples_per_class_overrides=max_samples_overrides,
                augmentation_factor=augmentation_factor,
                training_metric=training_metric,
                label_weights=label_weights,
                model_architecture=model_architecture,
            )
            mlflow_tracker.log_history(history)

            classifier = multi_stall_manager.get_soundscape_classifier()
            if hasattr(classifier, "model") and classifier.model is not None:
                mo = classifier.model.output_shape[-1]
                lm = len(classifier.valid_labels or [])
                if mo != lm:
                    raise RuntimeError(
                        f"Model/label size mismatch: {mo} vs {lm}; refusing to log"
                    )
            if getattr(classifier, "model_path", None):
                mlflow_tracker.log_model(str(classifier.model_path))
            if getattr(classifier, "label_to_idx", None):
                mlflow_tracker.log_labels(
                    {
                        "label_to_idx": classifier.label_to_idx,
                        "idx_to_label": classifier.idx_to_label,
                        "valid_labels": classifier.valid_labels,
                    }
                )

            if getattr(classifier, "model_path", None):
                mlflow_tracker.log_model(str(classifier.model_path))

                # --- register in MLflow Model Registry (enables @champion tagging) ---
                should_register = sd_config.get("register_model", False)
                auto_promote = sd_config.get("auto_promote_to_production", False)
                if should_register:
                    model_path_str = str(classifier.model_path)
                    label_mapping_path = model_path_str.replace(
                        ".keras", "_label_mapping.json"
                    )
                    num_classes = (
                        len(classifier.valid_labels) if classifier.valid_labels else 0
                    )
                    total_samples = sum(len(s) for s in training_set_by_label.values())
                    description = (
                        f"manifest={manifest_version}, classes={num_classes}, "
                        f"samples={total_samples}"
                    )
                    version = register_model(
                        model_path=model_path_str,
                        label_mapping_path=label_mapping_path,
                        description=description,
                        model_architecture=model_architecture,
                    )
                    if version:
                        logger.info(f"Registered model as version {version}")
                        if auto_promote:
                            if promote_model_to_production(
                                version=version, model_architecture=model_architecture
                            ):
                                logger.info(f"Promoted version {version} to Production")
                            else:
                                logger.warning(f"Failed to promote version {version}")
                    else:
                        logger.warning("Failed to register model in MLflow registry")

            # ---- eval: real frozen test set (headline) — NOT augmented ----------
            if test_set_by_label:
                trained = multi_stall_manager.get_soundscape_classifier()
                logger.info(f"Evaluating on {len(test_set_by_label)} classes")
                eval_results = evaluate_segments(
                    test_set_by_label, batch_size=32, classifier=trained
                )
                mlflow_tracker.log_evaluation(
                    eval_results.get("metrics", {}),
                    training_samples={
                        l: len(s) for l, s in training_set_by_label.items()
                    },
                    test_samples={l: len(s) for l, s in test_set_by_label.items()},
                )

            # ---- eval: proxy/eval-only — reported SEPARATELY, never averaged in --
            if eval_only_by_label:
                proxy = evaluate_segments(
                    eval_only_by_label,
                    batch_size=32,
                    classifier=multi_stall_manager.get_soundscape_classifier(),
                )
                pm = proxy.get("metrics", {})
                for k, v in pm.items() if isinstance(pm, dict) else []:
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"proxy_{k}", v)
                logger.info(
                    f"Proxy (eval-only) metrics, NOT comparable to real test: {pm}"
                )

            if tracker_created_here:
                mlflow_tracker.end_run(status="FINISHED")
        except Exception as train_error:
            if tracker_created_here:
                mlflow_tracker.end_run(status="FAILED")
            raise

        logger.info(f"Blob-store training complete (manifest {manifest_version})")
        return history, 200

    except Exception as e:
        logger.error(f"Blob-store training failed: {e}", exc_info=True)
        return [], 500
    finally:
        training_state["is_training"] = False


def smartstable_train(manager, project_id: int, ls, model_architecture: str = "panns"):
    """
    Training entry point called from webhook.
    Fetches tasks from Label Studio and runs the training pipeline.
    Automatically excludes tasks marked as 'test' (via task_purpose choice).
    """
    logger.info(f"Starting SmartStable training for project {project_id}")

    try:
        # Fetch only annotated tasks from Label Studio (much faster than fetching all)
        # exclude_test_tasks=True filters out tasks marked with task_purpose='test'
        training_tasks = list_tasks_by_project(
            project_id, ls_client=ls, only_annotated=True, exclude_test_tasks=True
        )

        if not training_tasks:
            logger.warning(
                f"No annotated training tasks found for project {project_id}"
            )
            return False

        logger.info(
            f"Fetched {len(training_tasks)} annotated training tasks (test tasks excluded)"
        )

        _label_studio_training(
            training_tasks,
            multi_stall_manager=manager,
            split_test_set=0.15,
            model_architecture=model_architecture,
        )

        # for task in training_tasks:
        #     annotations_count = len(task.annotations) if hasattr(task, 'annotations') else 0
        #     logger.info(f"Task {task.id}: {annotations_count} annotations")

        # logger.info("Training pipeline placeholder - implement full training here")
        return True

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return False


def _resolve_version(manifest_dir: Path, version: str) -> str | None:
    files = list(manifest_dir.glob("v*.jsonl"))
    if not files:
        return None
    if version and version != "latest":
        return version if (manifest_dir / f"{version}.jsonl").is_file() else None
    return max(
        files, key=lambda p: p.stat().st_mtime
    ).stem  # robust to v5 / v2.3 naming


def get_dataset_status(version="latest", datasets_path=None):
    datasets_path = datasets_path or os.getenv("DATASETS_PATH", "/datasets")
    manifest_dir = Path(datasets_path) / "manifests"
    ver = _resolve_version(manifest_dir, version)
    if not ver:
        return None

    lines = [json.loads(l) for l in open(manifest_dir / f"{ver}.jsonl") if l.strip()]
    header_path = manifest_dir / f"{ver}.header.json"
    header = json.load(open(header_path)) if header_path.is_file() else {}

    registry = get_label_registry()
    remap = get_label_remapping()  # {raw -> merged target}
    eval_only = set(registry.get_eval_only_labels())

    # min_samples (the trainability threshold the invariant uses)
    try:
        CONFIG_BASE_DIR = os.path.abspath(
            os.getenv(
                "SMARTSTABLE_CONFIG_DIR",
                os.path.join(
                    os.path.dirname(__file__), "..", "smartstablemodel", "config"
                ),
            )
        )
        with open(os.path.join(CONFIG_BASE_DIR, "training_settings.toml"), "rb") as f:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            min_samples = (
                tomllib.load(f)
                .get("soundscape_detector", {})
                .get("min_samples_per_class", 20)
            )
    except Exception:
        min_samples = 20

    # --- aggregate RAW labels: counts + group SETS (groups union on merge) -----
    raw = defaultdict(lambda: {"train": 0, "test": 0, "tg": set(), "eg": set()})
    for ln in lines:
        r = raw[ln["label"]]
        g = ln["group_id"]
        if ln.get("eval_only") or ln["split"] == "test":
            r["test"] += 1
            r["eg"].add(g)
        else:
            r["train"] += 1
            r["tg"].add(g)

    # --- fold raw -> merged (the classes the model actually trains on) ---------
    merged = defaultdict(
        lambda: {"train": 0, "test": 0, "tg": set(), "eg": set(), "from": {}}
    )
    eval_rows = []
    for lab, r in raw.items():
        if lab in eval_only:  # proxies: reported separately, never merged/trained
            eval_rows.append(
                {"label": lab, "test": r["test"], "test_groups": len(r["eg"])}
            )
            continue
        tgt = remap.get(lab, lab)
        m = merged[tgt]
        m["train"] += r["train"]
        m["test"] += r["test"]
        m["tg"] |= r["tg"]
        m["eg"] |= r["eg"]
        m["from"][lab] = r["train"]  # show each constituent's train contribution

    # --- per merged class: readiness status + blocking detection --------------
    trained, blocking = [], []
    for lab, m in sorted(merged.items()):
        train_g, test_g = len(m["tg"]), len(m["eg"])
        frac = registry.get_min_train_carrier_fraction(lab)
        if m["train"] < min_samples:
            status = "under_support"  # below threshold -> not trained
        elif test_g >= 1:
            status = "ok"
        elif frac is not None:
            status = "policy_train_only"  # 0 test groups, but by explicit policy
        else:
            status = "NO_TEST_COVERAGE"  # trained, 0 test groups, no policy -> BLOCKS training
            blocking.append(lab)
        row = {
            "label": lab,
            "train": m["train"],
            "test": m["test"],
            "train_groups": train_g,
            "test_groups": test_g,
            "status": status,
        }
        if len(m["from"]) > 1:  # only show composition when a merge actually happened
            row["merged_from"] = m["from"]
        if frac is not None:
            row["carrier_fraction"] = frac
        trained.append(row)

    return {
        "version": ver,
        "created": header.get("created"),
        "comment": header.get("comment"),
        "n_blobs": len(lines),
        "test_split": header.get("test_split"),
        "min_samples": min_samples,
        "blocking": blocking,  # <- fix these before training will run
        "trained_classes": trained,
        "eval_only": eval_rows,
    }


def get_untrained_classes(project_id, ls_client=None):
    registry = get_label_registry()
    trained = set(registry.get_trainable_label_values_for_classifier())
    remap = get_label_remapping()
    eval_only = set(registry.get_eval_only_labels())
    known = set(getattr(registry, "get_all_label_values", lambda: [])()) or (
        trained | set(remap) | eval_only
    )

    # pull ALL annotated tasks — including test-marked ones, we want full label coverage
    tasks = list_tasks_by_project(project_id, ls_client=ls_client, only_annotated=True)

    agg = defaultdict(lambda: {"count": 0, "groups": set()})
    for task in tasks:
        group_id, _ = get_group_id_and_provenance(task)
        anns = []
        if hasattr(task, "annotations") and task.annotations:
            for a in task.annotations:
                if isinstance(a, dict) and "result" in a:
                    anns.extend(a.get("result", []))
        for item in anns:
            for lab in item.get("value", {}).get("labels", []) or []:
                agg[lab]["count"] += 1
                agg[lab]["groups"].add(group_id)

    trained_rows, excluded, orphaned, eval_rows = [], [], [], []
    for lab, a in sorted(agg.items()):
        row = {"label": lab, "count": a["count"], "groups": len(a["groups"])}
        tgt = remap.get(lab)
        if lab in eval_only:
            eval_rows.append(row)
        elif lab in trained:
            trained_rows.append(row)
        elif tgt:
            row["merges_into"] = tgt
            trained_rows.append(row)  # trained via merge
        elif lab in known:
            row["reason"] = (
                getattr(registry, "get_label_description", lambda x: None)(lab)
                or "excluded (no description)"
            )
            excluded.append(row)
        else:
            orphaned.append(row)  # in LS data, not in config -> noise/artifact

    return {
        "project_id": project_id,
        "trained": trained_rows,
        "excluded_in_config": excluded,
        "orphaned_in_data": orphaned,
        "eval_only": eval_rows,
    }


def _label_studio_training(
    tasks,
    multi_stall_manager: MultiStallDetectorManager,
    split_test_set=0.15,
    mlflow_tracker=None,
    model_architecture="panns",
):
    """Background training function for Label Studio data format.

    Args:
        tasks: List of Label Studio tasks with annotations
        split_test_set: Fraction to hold out for evaluation (0-1)
        mlflow_tracker: Optional SoundscapeTrainingTracker for logging

    Returns:
        Evaluation results if split_test_set > 0, else None
    """
    global training_state

    if multi_stall_manager is None:
        return [], 500

    test_set_by_label = {}
    tracker_created_here = False

    try:
        training_state["is_training"] = True
        training_state["last_training_time"] = datetime.datetime.now()

        logger.info(f"Starting Label Studio training with {len(tasks)} tasks")

        # Dictionary to collect segments by label
        training_segments_by_label = {}

        accepted_labels = (
            get_label_registry().get_trainable_label_values_for_classifier()
        )
        merged_labels = get_label_registry().get_merged_labels()
        labels_to_collect = accepted_labels + list(merged_labels)
        excluded_labels = (
            get_label_registry().get_labels_to_exclude_from_training_for_classifier()
        )
        logger.info(f"Accepted labels for training: {accepted_labels}")
        logger.info(f"Merged labels: {merged_labels}")
        logger.info(f"Excluded labels for training: {excluded_labels}")

        # Counters for summary logging
        tasks_processed = 0
        tasks_skipped_no_audio = 0
        tasks_skipped_no_annotations = 0
        total_segments_extracted = 0

        # Process each task
        for task in tasks:
            # Get audio file path
            audio_url = task.data.get("audio", "")
            if not audio_url:
                tasks_skipped_no_audio += 1
                continue

            audio_path = ls_url_to_local_path(audio_url)
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file not found: {audio_path}")
                tasks_skipped_no_audio += 1
                continue

            # Get annotations for this task - they're embedded in the task now
            task_annotations = []

            # SDK data has annotations directly in task
            if hasattr(task, "annotations") and task.annotations:
                for ann in task.annotations:
                    if isinstance(ann, dict) and "result" in ann:
                        # Each annotation has a "result" field with the actual annotations
                        task_annotations.extend(ann.get("result", []))

            if not task_annotations:
                tasks_skipped_no_annotations += 1
                continue

            # Extract segments from annotations - now returns segments by label
            # Using include_metadata=True to get (audio, SampleID) tuples for tracking
            segments_by_label = multi_stall_manager.audio_processor.extract_segments_from_ls_annotations_multiclass(
                audio_path,
                task_annotations,
                accepted_labels=labels_to_collect,
                include_metadata=True,
            )

            # Accumulate segments by label
            for label, segments in segments_by_label.items():
                if label not in training_segments_by_label:
                    training_segments_by_label[label] = []
                training_segments_by_label[label].extend(segments)
                total_segments_extracted += len(segments)

            tasks_processed += 1

        # Log processing summary
        logger.info(
            f"Task processing complete: {tasks_processed} processed, "
            f"{tasks_skipped_no_audio} skipped (no audio), "
            f"{tasks_skipped_no_annotations} skipped (no annotations), "
            f"{total_segments_extracted} total segments extracted"
        )

        # Log total segments by label (before remapping)
        for label, segments in training_segments_by_label.items():
            logger.info(f"Total {label} segments: {len(segments)}")

        # Load training config for label remapping and exclusion
        CONFIG_BASE_DIR = os.getenv(
            "SMARTSTABLE_CONFIG_DIR",
            os.path.join(os.path.dirname(__file__), "..", "smartstablemodel", "config"),
        )
        CONFIG_BASE_DIR = os.path.abspath(CONFIG_BASE_DIR)
        training_settings_path = os.path.join(CONFIG_BASE_DIR, "training_settings.toml")
        if not os.path.isfile(training_settings_path):
            logger.error(
                f"training_settings.toml not found at {training_settings_path}. "
                "Create it or set SMARTSTABLE_CONFIG_DIR. Aborting training step."
            )
            raise FileNotFoundError(training_settings_path)

        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with open(training_settings_path, "rb") as f:
            logger.info(f"Loading training settings from {training_settings_path}")
            training_config = tomllib.load(f)

        # Apply label remapping from LabelRegistry (labels.json merge_into field)

        label_remapping = get_label_remapping()
        if label_remapping:
            logger.info(f"Applying label remapping from labels.json: {label_remapping}")
            for old_label, new_label in label_remapping.items():
                if old_label in training_segments_by_label:
                    if new_label not in training_segments_by_label:
                        training_segments_by_label[new_label] = []
                    training_segments_by_label[new_label].extend(
                        training_segments_by_label.pop(old_label)
                    )
                    logger.info(
                        f"Remapped label '{old_label}' -> '{new_label}' ({len(training_segments_by_label[new_label])} total samples)"
                    )
                else:
                    logger.info(
                        f"No samples found for label '{old_label}' to remap in training_segments_by_label: {training_segments_by_label.keys()}"
                    )

        # Log final label distribution after remapping/exclusion
        total_samples = sum(
            len(segments) for segments in training_segments_by_label.values()
        )
        logger.info(
            f"Final label distribution after remapping/exclusion for {total_samples} samples:"
        )
        for label, segments in training_segments_by_label.items():
            logger.info(f"  {label}: {len(segments)} samples")

        training_set_by_label = {}
        if split_test_set > 0:
            # Split off a test set if requested
            for label in training_segments_by_label:
                segments = training_segments_by_label[label]
                np.random.shuffle(segments)
                split_idx = int(len(segments) * (1 - split_test_set))
                training_segments_by_label[label] = segments[:split_idx]
                test_segments = segments[split_idx:]
                training_set_by_label[label] = segments[:split_idx]
                test_set_by_label[label] = test_segments

            logger.info(
                f"Split training and test sets with ratio {1 - split_test_set:.2f}/{split_test_set:.2f}"
            )
        else:
            training_set_by_label = training_segments_by_label

        # Perform multi-class training for the classifier
        if training_set_by_label:
            logger.info("Starting multi-class training...")

            # Extract training hyperparameters (config already loaded above)
            sd_config = training_config.get("soundscape_detector", {})
            learning_rate = float(sd_config.get("learning_rate", 1e-4))
            epochs = sd_config.get("num_epochs", 10)
            batch_size = sd_config.get("batch_size", 32)
            dropout_rates = sd_config.get("dropout_rates", None)
            min_samples = sd_config.get("min_samples_per_class", 5)
            max_samples = sd_config.get("max_samples_per_class", None)  # None = no cap
            max_samples_overrides = get_max_samples_per_class_overrides()
            augmentation_factor = sd_config.get("augmentation_factor", 2)
            training_metric = sd_config.get(
                "training_metric", "accuracy"
            )  # accuracy, macro_f1, balanced_accuracy, weighted_f1
            short_event_padding_enabled = sd_config.get(
                "short_event_padding_enabled", True
            )
            short_event_threshold_seconds = float(
                sd_config.get("short_event_threshold_seconds", 2.0)
            )
            short_event_crossfade_ms = int(
                sd_config.get("short_event_crossfade_ms", 40)
            )
            short_event_position_mode = str(
                sd_config.get("short_event_position_mode", "random")
            ).lower()
            short_event_seed = sd_config.get("short_event_seed", None)

            canonical_background_label = None
            try:
                metamodel_config = load_metamodel_config()
                canonical_background_label = (
                    str(
                        metamodel_config.model_parameters.get("background_label", "")
                    ).strip()
                    or None
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load canonical background label from metamodel config: {e}"
                )

            configured_short_event_background_label = str(
                sd_config.get(
                    "short_event_background_label",
                    canonical_background_label or "background",
                )
            )

            candidate_background_labels = [
                configured_short_event_background_label,
                canonical_background_label,
                "background",
                "normal",
            ]
            short_event_background_label = next(
                (
                    lbl
                    for lbl in candidate_background_labels
                    if lbl and lbl in training_set_by_label
                ),
                configured_short_event_background_label,
            )
            if short_event_background_label != configured_short_event_background_label:
                logger.info(
                    "Short-event padding background label fallback: "
                    f"requested='{configured_short_event_background_label}', "
                    f"using='{short_event_background_label}'"
                )

            if short_event_position_mode not in {"random", "center"}:
                logger.warning(
                    f"Invalid short_event_position_mode='{short_event_position_mode}', defaulting to 'random'"
                )
                short_event_position_mode = "random"

            padding_stats = _apply_short_event_padding(
                training_set_by_label,
                audio_processor=multi_stall_manager.audio_processor,
                background_label=short_event_background_label,
                enabled=bool(short_event_padding_enabled),
                threshold_seconds=short_event_threshold_seconds,
                crossfade_ms=short_event_crossfade_ms,
                position_mode=short_event_position_mode,
                seed=short_event_seed,
            )
            logger.info(
                "Short-event padding summary: "
                f"eligible={padding_stats['eligible_short']}, "
                f"composited={padding_stats['composited']}, "
                f"skipped_no_background={padding_stats['skipped_no_background']}, "
                f"background_label='{short_event_background_label}', "
                f"configured_background_label='{configured_short_event_background_label}', "
                f"canonical_background_label='{canonical_background_label}', "
                f"threshold_s={short_event_threshold_seconds}, "
                f"crossfade_ms={short_event_crossfade_ms}, "
                f"mode={short_event_position_mode}"
            )

            # Get label weights for weighted metrics (from labels.json training_importance)
            label_weights = get_training_importance_weights()
            if max_samples_overrides:
                logger.info(
                    f"Using per-label max_samples_per_class overrides: {max_samples_overrides}"
                )

            # Initialize MLflow tracking for soundscape classifier training (if not passed in)
            if mlflow_tracker is None:
                mlflow_tracker = SoundscapeTrainingTracker(
                    experiment_name="soundscape_classifier"
                )
                tracker_created_here = True
                # Default tags for webhook-triggered training (no holdout test set)
                default_tags = {
                    "training_source": "label_studio",
                    "tasks_count": str(len(tasks)),
                    "endpoint": "webhook",
                    "evaluation_type": "none",  # No proper evaluation - metrics are overfitted
                }
            else:
                # Tags will be set by the calling endpoint (e.g., train_use_config)
                default_tags = {
                    "training_source": "label_studio",
                    "tasks_count": str(len(tasks)),
                    "model_architecture": model_architecture,
                }

            mlflow_tracker.start_training_run(
                training_segments_by_label=training_segments_by_label,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                tags=default_tags,
                dropout_rates=dropout_rates,
                min_samples_per_class=min_samples,
                max_samples_per_class=max_samples,
                augmentation_factor=augmentation_factor,
                label_remapping=label_remapping,
                exclude_labels=excluded_labels,
                training_metric=training_metric,
                label_weights=label_weights,
                run_name=f"train_{model_architecture}_{len(tasks)}_tasks",
                audio_backend=getattr(
                    multi_stall_manager.audio_processor, "backend", "unknown"
                ),
            )

            try:
                history = multi_stall_manager.train_soundscape_classifier_multiclass(
                    training_segments_by_label,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    batch_size=batch_size,
                    dropout_rates=dropout_rates,
                    min_samples_per_class=min_samples,
                    max_samples_per_class=max_samples,
                    max_samples_per_class_overrides=max_samples_overrides,
                    augmentation_factor=augmentation_factor,
                    training_metric=training_metric,
                    label_weights=label_weights,
                    model_architecture=model_architecture,
                )

                # Log training metrics to MLflow
                mlflow_tracker.log_history(history)

                # Log model artifact and actual class info
                classifier = multi_stall_manager.get_soundscape_classifier()

                # CRITICAL: Validate model output size matches label mapping size before logging to MLflow
                try:
                    if hasattr(classifier, "model") and classifier.model is not None:
                        model_output_size = classifier.model.output_shape[-1]
                        label_mapping_size = (
                            len(classifier.valid_labels)
                            if classifier.valid_labels
                            else 0
                        )

                        if model_output_size != label_mapping_size:
                            error_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           🚨 CRITICAL: MODEL/LABEL MAPPING SIZE MISMATCH 🚨                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Model output size: {model_output_size} classes
║ Label mapping size: {label_mapping_size} classes
║ 
║ REFUSING TO LOG INVALID MODEL TO MLFLOW!
║ 
║ The model was trained with {model_output_size} labels but the
║ label_mapping has {label_mapping_size} labels.
╚══════════════════════════════════════════════════════════════════════════════╝
"""
                            logger.error(error_msg)
                            raise RuntimeError(error_msg)
                        else:
                            logger.info(
                                f"✓ Model output ({model_output_size}) matches label mapping ({label_mapping_size}) - Safe to log to MLflow"
                            )
                except Exception as e:
                    if "CRITICAL" in str(e):
                        raise
                    logger.warning(
                        f"Could not validate model/label mapping sizes before MLflow logging: {e}"
                    )

                # Get actual training sample counts (after capping/augmentation)
                actual_training_samples = None
                if (
                    hasattr(classifier, "_last_training_info")
                    and classifier._last_training_info
                ):
                    actual_training_samples = classifier._last_training_info.get(
                        "samples_after_augmentation"
                    )
                    logger.info(
                        f"Actual training samples after augmentation: {actual_training_samples}"
                    )

                if hasattr(classifier, "model_path") and classifier.model_path:
                    mlflow_tracker.log_model(str(classifier.model_path))

                    # Register model in MLflow Model Registry if configured
                    sd_config = training_config.get("soundscape_detector", {})
                    should_register = sd_config.get("register_model", False)
                    auto_promote = sd_config.get("auto_promote_to_production", False)

                    if should_register:
                        # Get label mapping path
                        model_path_str = str(classifier.model_path)
                        label_mapping_path = model_path_str.replace(
                            ".keras", "_label_mapping.json"
                        )

                        # Build description from training info
                        num_classes = (
                            len(classifier.valid_labels)
                            if classifier.valid_labels
                            else 0
                        )
                        total_samples = sum(
                            len(segs) for segs in training_segments_by_label.values()
                        )
                        description = (
                            f"Classes: {num_classes}, Samples: {total_samples}"
                        )

                        version = register_model(
                            model_path=model_path_str,
                            label_mapping_path=label_mapping_path,
                            description=description,
                            model_architecture=model_architecture,
                        )

                        if version:
                            logger.info(f"Registered model as version {version}")
                            if auto_promote:
                                if promote_model_to_production(
                                    version=version,
                                    model_architecture=model_architecture,
                                ):
                                    logger.info(
                                        f"Promoted version {version} to Production"
                                    )
                                else:
                                    logger.warning(
                                        f"Failed to promote version {version} to Production"
                                    )
                        else:
                            logger.warning(
                                "Failed to register model in MLflow registry"
                            )

                if hasattr(classifier, "label_to_idx") and classifier.label_to_idx:
                    mlflow_tracker.log_labels(
                        {
                            "label_to_idx": classifier.label_to_idx,
                            "idx_to_label": classifier.idx_to_label,
                            "valid_labels": classifier.valid_labels,
                        }
                    )
                    # Log actual class count (may differ from initial if human_speech added/failed)
                    actual_classes = classifier.valid_labels or []
                    total_samples = sum(
                        len(segs) for segs in training_segments_by_label.values()
                    )
                    mlflow_tracker.log_final_class_info(actual_classes, total_samples)

                # Log the Test Pool Manifest (SampleIDs NOT used for training)
                if (
                    hasattr(classifier, "_last_training_info")
                    and classifier._last_training_info
                ):
                    test_pool = classifier._last_training_info.get("test_pool")
                    if test_pool:
                        from ml_backend.smartstable_core.mlflow_support import (
                            log_test_pool_manifest,
                        )

                        log_test_pool_manifest(test_pool)

                # Run evaluation on test set BEFORE ending the MLflow run
                if test_set_by_label:
                    logger.info(
                        "Starting evaluation on test set using freshly trained model..."
                    )
                    trained_classifier = multi_stall_manager.get_soundscape_classifier()
                    eval_results = evaluate_segments(
                        test_set_by_label, batch_size=32, classifier=trained_classifier
                    )
                    logger.info(f"Evaluation results: {eval_results}")

                    # Use actual training samples after capping/augmentation if available
                    samples_before_augmentation = None
                    if (
                        hasattr(trained_classifier, "_last_training_info")
                        and trained_classifier._last_training_info
                    ):
                        training_samples = trained_classifier._last_training_info.get(
                            "samples_after_augmentation", {}
                        )
                        samples_before_augmentation = (
                            trained_classifier._last_training_info.get(
                                "samples_after_cap", {}
                            )
                        )
                        logger.info(
                            f"Using actual training samples for reporting: {training_samples}"
                        )
                    else:
                        training_samples = {
                            label: len(segs)
                            for label, segs in training_set_by_label.items()
                        }
                    test_samples = {
                        label: len(segs) for label, segs in test_set_by_label.items()
                    }
                    mlflow_tracker.log_evaluation(
                        eval_results.get("metrics", {}),
                        training_samples=training_samples,
                        test_samples=test_samples,
                        samples_before_augmentation=samples_before_augmentation,
                    )

                if tracker_created_here:
                    mlflow_tracker.end_run(status="FINISHED")
            except Exception as train_error:
                if tracker_created_here:
                    mlflow_tracker.end_run(status="FAILED")
                raise train_error

            training_state["training_history"].append(
                {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "type": "multiclass_training",
                    "segments_by_label": {
                        label: len(segments)
                        for label, segments in training_segments_by_label.items()
                    },
                    "tasks_processed": len(tasks),
                    "success": True,
                    "labels_trained": list(training_segments_by_label.keys()),
                }
            )

            logger.info("Multi-class training completed successfully")
        else:
            logger.warning("No training segments extracted from annotations")

    except Exception as e:
        logger.error(f"Multi-class training failed: {str(e)}", exc_info=True)
        training_state["training_history"].append(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "multiclass_training",
                "error": str(e),
                "success": False,
            }
        )
    finally:
        training_state["is_training"] = False
