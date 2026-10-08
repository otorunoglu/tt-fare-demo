"""
External human-speech provider (Common Voice). GDPR-clean: stable human voices
are never stored as blobs; human_talking is sourced entirely from this public
dataset. Returns speaker-disjoint train/test so it flows through the same
group-aware split + invariant as blob classes. group_id = Common Voice speaker.
"""

import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import audio_preprocessing  # same Rust backend used everywhere -> parity

logger = logging.getLogger("HumanSpeech")

_SR = 16000


def _resolve_cv_dir(data_dir: Path, language: str) -> Path | None:
    hits = list(data_dir.glob(f"cv-corpus-*/{language}"))
    return hits[0] if hits else None


def _load_split_rows(cv_dir: Path, which: str, max_rows: int, seed: int):
    import csv

    fname = {"train": "train.tsv", "test": "test.tsv"}[which]
    tsv = cv_dir / fname
    used_official = tsv.exists()
    if not used_official:
        tsv = cv_dir / "validated.tsv"
    if not tsv.exists():
        return pd.DataFrame(), used_official

    df = pd.read_csv(
        tsv,
        sep="\t",
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        on_bad_lines="skip",
    )
    if "up_votes" in df and "down_votes" in df:
        # votes are strings now (keep_default_na), coerce for the comparison
        df["up_votes"] = pd.to_numeric(df["up_votes"], errors="coerce").fillna(0)
        df["down_votes"] = pd.to_numeric(df["down_votes"], errors="coerce").fillna(0)
        df = df[df["up_votes"] >= df["down_votes"]]
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
    return df, used_official


def _segment_clip(audio_path: Path, win_seconds: float) -> np.ndarray | None:
    """Decode via Rust backend, mono, fixed window. Returns None on failure/too-short-after-pad."""
    try:
        audio, sr = audio_preprocessing.load_audio(str(audio_path), _SR)
    except Exception as e:
        logger.debug(f"decode failed {audio_path.name}: {e}")
        return None
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    target = int(round(win_seconds * _SR))
    if len(audio) >= target:
        seg = audio[:target]
    else:
        seg = np.pad(audio, (0, target - len(audio)))
    return seg.astype(np.float32)


def load_human_speech(
    data_dir,
    win_seconds: float,
    languages=("en", "fi"),
    max_per_language=400,
    seed=42,
):
    """
    Returns (train_segs, test_segs), each a list of (audio, group_id) tuples where
    group_id is 'cv:<client_id>' — the speaker. Speaker-disjoint across train/test.

    win_seconds MUST be the model's window (audio_processor.win_seconds) so these
    segments are the same length as blob segments.
    """
    data_dir = Path(data_dir)

    def _collect(which: str):
        out = []
        for lang in languages:
            cv_dir = _resolve_cv_dir(data_dir, lang)
            if cv_dir is None:
                logger.warning(f"No Common Voice dir for '{lang}' under {data_dir}")
                continue
            df, official = _load_split_rows(cv_dir, which, max_per_language * 3, seed)
            if df.empty:
                continue
            clips = cv_dir / "clips"
            if not clips.exists():
                clips = cv_dir
            n = 0
            for _, row in df.iterrows():
                if n >= max_per_language:
                    break
                seg = _segment_clip(clips / row["path"], win_seconds)
                if seg is None:
                    continue
                speaker = row.get("client_id", row["path"])  # speaker = group
                out.append((seg, f"cv:{speaker}"))
                n += 1
            logger.info(
                f"{lang} {which}: {n} segments "
                f"({'official split' if official else 'validated.tsv fallback'})"
            )
        return out

    train = _collect("train")
    test = _collect("test")

    # Fallback safety: if official splits were absent and both pulled from validated.tsv,
    # they could share speakers. Enforce disjointness by moving test speakers out of train.
    train_speakers = {g for _, g in train}
    test_speakers = {g for _, g in test}
    leaked = train_speakers & test_speakers
    if leaked:
        logger.warning(
            f"{len(leaked)} speakers in both splits (no official split?) "
            f"-> removing them from train to enforce disjointness"
        )
        train = [(a, g) for (a, g) in train if g not in test_speakers]

    logger.info(
        f"human_speech: {len(train)} train / {len(test)} test, "
        f"{len(train_speakers - test_speakers)} train speakers / "
        f"{len(test_speakers)} test speakers"
    )
    return train, test

