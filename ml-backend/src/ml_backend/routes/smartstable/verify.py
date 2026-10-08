"""
Verification endpoint for comparing Python (Keras) vs ONNX inference pipelines.

Runs the full Python pipeline (including silence gate, anomaly detection)
and then ONNX inference on the same audio, reporting per-segment alignment.
"""

import json
import logging
import os


import numpy as np
import onnxruntime as ort
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

verify_router = APIRouter()


# ── ONNX Session Cache ──────────────────────────────────────────────────────
_onnx_sessions = {}


def _get_onnx_sessions(onnx_dir: str):
    """Load (or return cached) ONNX embedder + classifier sessions."""
    if onnx_dir in _onnx_sessions:
        return _onnx_sessions[onnx_dir]

    embedder_path = os.path.join(onnx_dir, "full_pipeline_embedder.onnx")
    classifier_path = os.path.join(onnx_dir, "full_pipeline_classifier.onnx")

    if not os.path.exists(embedder_path):
        raise FileNotFoundError(f"Embedder ONNX not found: {embedder_path}")
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Classifier ONNX not found: {classifier_path}")

    embedder = ort.InferenceSession(embedder_path, providers=["CPUExecutionProvider"])
    classifier = ort.InferenceSession(
        classifier_path, providers=["CPUExecutionProvider"]
    )

    _onnx_sessions[onnx_dir] = (embedder, classifier)
    return embedder, classifier


def _load_onnx_label_mapping(onnx_dir: str) -> dict:
    """Load idx→label mapping from the ONNX model directory."""
    label_path = os.path.join(onnx_dir, "label_mapping.json")
    if not os.path.exists(label_path):
        return {}
    with open(label_path) as f:
        raw = json.load(f)
    if "idx_to_label" in raw:
        raw = raw["idx_to_label"]
    return {int(k): v for k, v in raw.items()}


def _run_onnx_inference(audio_window: np.ndarray, embedder, classifier, clamp=True):
    """Run a single audio window through the ONNX pipeline (mimicking Rust)."""
    audio = audio_window.astype(np.float32)
    if clamp:
        audio = np.clip(audio, -1.0, 1.0)

    input_name = embedder.get_inputs()[0].name
    embed_result = embedder.run(None, {input_name: audio.reshape(1, -1)})

    output_names = [o.name for o in embedder.get_outputs()]
    if "embeddings" in output_names:
        embedding = embed_result[output_names.index("embeddings")]
    else:
        embedding = embed_result[-1]

    cls_input_name = classifier.get_inputs()[0].name
    cls_result = classifier.run(None, {cls_input_name: embedding})
    probs = cls_result[0]
    return embedding.flatten(), probs.flatten()


# ── Loudness (same as ml_audio_core) ─────────────────────────────────────────
def _compute_loudness_dbfs(audio: np.ndarray) -> float:
    """Compute loudness in dBFS (same as ml_audio_core.extract_loudness_features)."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return -100.0
    rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
    dbfs = 20.0 * np.log10(rms + 1e-12)
    return max(dbfs, -100.0)


# ── Request / Response ───────────────────────────────────────────────────────
class VerifyRequest(BaseModel):
    task_id: int = Field(..., description="Label Studio task ID to verify")
    onnx_dir: str = Field(
        default="/src/stable-edge-monitor/data/models",
        description="Path to ONNX model directory",
    )
    max_segments: int = Field(
        default=0, description="Max segments to compare (0 = all)"
    )
    silence_threshold: float = Field(
        default=-40.0,
        description="Loudness threshold (dBFS) below which segments are 'silence'",
    )


class SegmentComparison(BaseModel):
    index: int
    time_range: str
    loudness_dbfs: float
    python_label: str
    python_confidence: float
    onnx_label: str
    onnx_confidence: float
    label_agrees: bool
    confidence_diff: float
    is_silence: bool


class VerifyResponse(BaseModel):
    task_id: int
    audio_file: str
    total_segments: int
    segments_compared: int
    label_agreement_count: int
    label_agreement_pct: float
    avg_confidence_diff: float
    max_confidence_diff: float
    silence_segments: int
    python_model_version: str
    onnx_label_map: dict
    python_label_map: dict
    segments: list[SegmentComparison]


@verify_router.post("/verify-pipeline", response_model=VerifyResponse)
async def verify_pipeline(request: VerifyRequest):
    """
    Compare the full Python inference pipeline (with silence gate)
    against ONNX inference for a given Label Studio task.

    Returns per-segment label + confidence comparison.
    """
    try:
        # 1. Load ONNX models
        embedder, classifier = _get_onnx_sessions(request.onnx_dir)
        onnx_label_map = _load_onnx_label_mapping(request.onnx_dir)

        # 2. Load Python pipeline via the full manager
        from ml_backend.smartstable_core.manager_loader import get_multistall_manager

        manager = get_multistall_manager()

        # 3. Fetch task + resolve audio path
        from ml_backend.ls_client import get_ls_client
        from ml_backend.util.label_studio_helper import ls_url_to_local_path

        ls = get_ls_client()
        task_obj = ls.tasks.get(id=request.task_id)
        if hasattr(task_obj, "data"):
            task_data = task_obj.data
        elif isinstance(task_obj, dict):
            task_data = task_obj.get("data", task_obj)
        else:
            task_data = {}

        audio_url = task_data.get("audio", "")
        local_fp = ls_url_to_local_path(audio_url)
        logger.info(f"Verify pipeline: task={request.task_id}, audio={local_fp}")

        if not os.path.exists(local_fp):
            raise HTTPException(
                status_code=404, detail=f"Audio file not found: {local_fp}"
            )

        # 4. Run FULL Python pipeline (this includes silence gate, anomaly etc.)
        py_results = manager.process_audio_file(
            local_fp,
            stable_id="stable01",
            stall_id="stall01",
        )

        # 5. Also load raw audio for ONNX comparison
        from ml_audio_core.audio_processing import AudioProcessor

        ap = AudioProcessor(sr=16000, win_seconds=2.0, win_overlap=0.5)
        audio, sr = ap.load_and_resample_audio(local_fp)
        windows, times, _ = ap.create_sliding_windows(
            audio, segment_duration=2.0, overlap=0.5
        )

        total_segments = len(windows)
        n = request.max_segments if request.max_segments > 0 else total_segments
        n = min(n, total_segments, len(py_results))

        # 6. Get Python model info
        clf = manager.get_soundscape_classifier()
        py_version = (
            clf.get_model_version() if hasattr(clf, "get_model_version") else "unknown"
        )
        py_label_map = clf.idx_to_label or {}

        # 7. Compare segment by segment
        comparisons = []
        agree_count = 0
        total_conf_diff = 0.0
        max_conf_diff = 0.0
        silence_count = 0

        for i in range(n):
            py = py_results[i]
            window = windows[i]
            t_start, t_end = times[i]

            # Python side
            py_probs = py.label_probabilities or {}
            if py_probs:
                # Find raw top label
                raw_py_label = max(py_probs, key=lambda k: float(py_probs[k]))
                raw_py_conf = float(py_probs[raw_py_label])
            else:
                raw_py_label = py.predicted_label or "unknown"
                raw_py_conf = py.probability

            py_final_label = py.predicted_label
            loudness = py.loudness
            is_silence = py_final_label == "silence"
            if is_silence:
                silence_count += 1

            # ONNX side (with Rust-style clamp)
            _, onnx_probs_raw = _run_onnx_inference(
                window, embedder, classifier, clamp=True
            )
            onnx_probs_dict = {
                onnx_label_map.get(j, str(j)): float(p)
                for j, p in enumerate(onnx_probs_raw)
            }

            # Apply same silence gate as Rust
            # Note: Python's process_audio_file already applied silence gate logic
            # via extract_loudness_features -> < silence_threshold -> "silence" label.
            # We replicate that here for ONNX to match system behavior.
            onnx_loudness = _compute_loudness_dbfs(window)
            if onnx_loudness < request.silence_threshold:
                onnx_label = "silence"
                onnx_conf = 1.0
            else:
                onnx_label = max(onnx_probs_dict, key=onnx_probs_dict.get)
                onnx_conf = onnx_probs_dict[onnx_label]

            # Compare raw model output (unless silenced)
            # If silence gate triggered on Py side, we expect it on ONNX side too.
            if is_silence:
                compare_py_label = "silence"
                compare_py_conf = 1.0
            else:
                compare_py_label = raw_py_label
                compare_py_conf = raw_py_conf

            agrees = compare_py_label == onnx_label
            if agrees:
                agree_count += 1
            conf_diff = abs(compare_py_conf - onnx_conf)
            total_conf_diff += conf_diff
            max_conf_diff = max(max_conf_diff, conf_diff)

            comparisons.append(
                SegmentComparison(
                    index=i,
                    time_range=f"{t_start:.2f}-{t_end:.2f}s",
                    loudness_dbfs=round(loudness, 2),
                    python_label=compare_py_label,  # Showing RAW label for verification
                    python_confidence=round(compare_py_conf, 6),
                    onnx_label=onnx_label,
                    onnx_confidence=round(onnx_conf, 6),
                    label_agrees=agrees,
                    confidence_diff=round(conf_diff, 6),
                    is_silence=is_silence,
                )
            )

        return VerifyResponse(
            task_id=request.task_id,
            audio_file=str(local_fp),
            total_segments=total_segments,
            segments_compared=n,
            label_agreement_count=agree_count,
            label_agreement_pct=round(100 * agree_count / n, 1) if n > 0 else 0,
            avg_confidence_diff=round(total_conf_diff / n, 6) if n > 0 else 0,
            max_confidence_diff=round(max_conf_diff, 6),
            silence_segments=silence_count,
            python_model_version=py_version,
            onnx_label_map={str(k): v for k, v in onnx_label_map.items()},
            python_label_map={str(k): v for k, v in py_label_map.items()},
            segments=comparisons,
        )

    except Exception as e:
        logger.exception(f"Verification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Artifact Export ──────────────────────────────────────────────────────────


class VerificationArtifactRequest(BaseModel):
    task_id: int = Field(..., description="Label Studio task ID to export")
    stable_id: str = "stable01"
    stall_id: str = "stall01"


@verify_router.post("/verify-pipeline-export")
async def verify_pipeline_export(request: VerificationArtifactRequest):
    """
    Generate a 'Golden Artifact' JSON containing:
    1. Raw audio (Base64)
    2. Expected Python pipeline results
    3. Config snapshot

    This artifact can be loaded by the Rust `verify-artifact` command for offline verification.
    """
    import base64

    try:
        # 1. Load Manager
        from ml_backend.smartstable_core.manager_loader import get_multistall_manager

        manager = get_multistall_manager()

        # 2. Fetch task & audio path
        from ml_backend.ls_client import get_ls_client
        from ml_backend.util.label_studio_helper import ls_url_to_local_path

        ls = get_ls_client()
        task_obj = ls.tasks.get(id=request.task_id)
        if hasattr(task_obj, "data"):
            task_data = task_obj.data
        elif isinstance(task_obj, dict):
            task_data = task_obj.get("data", task_obj)
        else:
            task_data = {}

        audio_url = task_data.get("audio", "")
        local_fp = ls_url_to_local_path(audio_url)

        if not os.path.exists(local_fp):
            raise HTTPException(status_code=404, detail=f"Audio not found: {local_fp}")

        # 3. Read Raw Audio (for the artifact)
        # We read as bytes to base64 encode directly
        with open(local_fp, "rb") as f:
            audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        # 4. Run Inference (Get Expected Results)
        # We rely on the manager's processing which handles silence gate, normalization etc.
        py_results = manager.process_audio_file(
            local_fp,
            stable_id=request.stable_id,
            stall_id=request.stall_id,
        )

        expected_segments = []
        for i, res in enumerate(py_results):
            # Normalize label probabilities for export
            probs = res.label_probabilities or {}
            # Ensure "normal" logic is reflected in the top label
            # (The Python manager already applies the fallback logic in .predicted_label)

            expected_segments.append(
                {
                    "index": i,
                    "start_time": res.start_time,
                    "end_time": res.end_time,
                    "label": res.predicted_label,
                    "confidence": res.probability,
                    "loudness": res.loudness,
                    "is_anomaly": res.is_anomaly,
                    "label_probs": probs,
                }
            )

        # 5. Config Snapshot
        # Get threshold used
        # Retrieve what we effectively used (hard to get exact runtime value if dynamic, but we know the lookup)
        # We can construct a snapshot manually for reference
        from ml_backend.smartstable_core.manager_loader import get_model_info

        model_info = get_model_info()

        # Load config to check thresholds
        try:
            from smartstablemodel.config.metamodel_config import MetaModelConfig

            full_config = MetaModelConfig.from_toml()
            clf_thresholds = full_config.model_parameters.get(
                "classifier_thresholds", {}
            )
            silence_threshold = full_config.model_parameters.get(
                "silence_threshold", -40.0
            )
        except Exception:
            clf_thresholds = {}
            silence_threshold = -40.0

        artifact = {
            "metadata": {
                "task_id": request.task_id,
                "filename": os.path.basename(local_fp),
                "generated_at": str(np.datetime64("now")),
                "model_version": model_info.get("model_name", "unknown"),
            },
            "audio_data": audio_b64,  # The raw file
            "config_snapshot": {
                "classifier_thresholds": clf_thresholds,
                "silence_threshold": silence_threshold,
            },
            "expected_results": expected_segments,
        }

        return artifact

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Artifact export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
