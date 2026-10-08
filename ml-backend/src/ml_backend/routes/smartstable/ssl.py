from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import logging
import os
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# from sklearn.manifold import TSNE  # Optional for better viz but slower
import tensorflow as tf

from ml_backend.smartstable_core.task_filtering import list_tasks_by_recording_period
from ml_backend.util.label_studio_helper import ls_url_to_local_path
from ml_audio_core.ssl.trainer import SSLTrainer
from ml_audio_core.audio_processing import AudioProcessor

logger = logging.getLogger("SSLRouter")

ssl_router = APIRouter(prefix="/ssl", tags=["ssl"])

# Status tracking (simple in-memory for MVP)
_ssl_job = {"status": "idle", "job_id": None, "progress": 0, "message": ""}


class TrainSSLRequest(BaseModel):
    project_id: int = 1
    stable_id: str
    stall_id: str = "all"
    date_from: str
    date_until: str
    max_tasks: int = 1000
    epochs: int = 20
    batch_size: int = 32


class VisualizeRequest(BaseModel):
    project_id: int = 1
    stable_id: str
    stall_id: str = "all"
    date_from: str
    date_until: str
    max_samples: int = 200


def run_ssl_training(file_paths, output_dir, epochs, batch_size):
    global _ssl_job
    try:
        _ssl_job["status"] = "running"
        _ssl_job["message"] = f"Training on {len(file_paths)} files..."
        logger.info(_ssl_job["message"])

        trainer = SSLTrainer(output_dir=output_dir)
        history = trainer.train(
            file_paths=file_paths, epochs=epochs, batch_size=batch_size
        )

        _ssl_job["status"] = "completed"
        _ssl_job["message"] = (
            f"Training finished. Loss: {history.history['loss'][-1]:.4f}"
        )
        logger.info(_ssl_job["message"])

    except Exception as e:
        logger.error(f"SSL Training failed: {e}", exc_info=True)
        _ssl_job["status"] = "failed"
        _ssl_job["message"] = str(e)


@ssl_router.post("/train")
def train_ssl(request: TrainSSLRequest, background_tasks: BackgroundTasks):
    global _ssl_job

    if _ssl_job["status"] == "running":
        return {"status": "error", "message": "SSL training already in progress"}

    # 1. Fetch tasks
    tasks = list_tasks_by_recording_period(
        project_id=request.project_id,
        stable_id=request.stable_id,
        stall_id=None if request.stall_id == "all" else request.stall_id,
        date_from=request.date_from,
        date_until=request.date_until,
        max_tasks=request.max_tasks,
        return_summary=True,
    )

    if not tasks:
        raise HTTPException(400, "No audio files found for the given criteria.")

    # 2. Extract file paths
    file_paths = []
    for t in tasks:
        url = t.get("audio")
        if url:
            # Convert LS url to local path, e.g. /label-studio/data/audio/...
            # ls_url_to_local_path handles /data/local-files/?d=... conversion
            path = ls_url_to_local_path(url)
            if os.path.exists(path):
                file_paths.append(path)

    if not file_paths:
        raise HTTPException(400, "Resolved 0 valid local file paths.")

    # 3. Start background job
    # output dir inside models folder
    output_dir = "/home/johannesgeisler/dev/smartstablemodel/models/ssl_experiment"

    background_tasks.add_task(
        run_ssl_training, file_paths, output_dir, request.epochs, request.batch_size
    )

    return {
        "status": "started",
        "file_count": len(file_paths),
        "output_dir": output_dir,
    }


@ssl_router.get("/status")
def get_status():
    return _ssl_job


@ssl_router.post("/visualize")
def visualize_embeddings(request: VisualizeRequest):
    """
    Generate a 2D PCA plot of embeddings for the specified data using the trained SSL encoder.
    """

    # 1. Load Encoder
    model_path = "/home/johannesgeisler/dev/smartstablemodel/models/ssl_experiment/ssl_encoder.keras"
    if not os.path.exists(model_path):
        raise HTTPException(404, "No trained SSL model found. Run /train first.")

    try:
        encoder = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {e}")

    # 2. Fetch Sample Data
    tasks = list_tasks_by_recording_period(
        project_id=request.project_id,
        stable_id=request.stable_id,
        stall_id=None if request.stall_id == "all" else request.stall_id,
        date_from=request.date_from,
        date_until=request.date_until,
        max_tasks=request.max_samples,
        return_summary=True,
    )

    files = []
    for t in tasks:
        p = ls_url_to_local_path(t.get("audio"))
        if p and os.path.exists(p):
            files.append(p)

    if not files:
        raise HTTPException(400, "No files found to visualize.")

    # 3. Inference
    ap = AudioProcessor()
    embeddings = []
    valid_files = 0

    # We need to know input shape expected by encoder (usually (None, 64, 96, 1))
    # AudioProcessor default is compatible.

    for f in files:
        try:
            # Load ONE window
            aud, _ = ap.load_and_resample_audio(f, duration=2.0)
            if len(aud) < 32000:
                continue  # Skip short files

            mel = ap.audio_to_mel_spectrogram(aud[:32000])  # (n_mels, tbins)
            mel_batch = mel[np.newaxis, ..., np.newaxis]  # (1, 64, 96, 1)

            emb = encoder(mel_batch)  # (1, 128)
            embeddings.append(emb.numpy()[0])
            valid_files += 1
        except Exception as e:
            logger.warning(f"Failed to infer on {f}: {e}")

    if not embeddings:
        raise HTTPException(500, "Failed to generate any embeddings.")

    embeddings = np.array(embeddings)

    # 4. dimensionality reduction (PCA)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)  # (N, 2)

    # 5. Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.7, c="blue")
    plt.title(f"SSL Embeddings PCA (n={len(embeddings)})")
    plt.grid(True, alpha=0.3)

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    return {"image_base64": img_str, "sample_count": len(embeddings)}
