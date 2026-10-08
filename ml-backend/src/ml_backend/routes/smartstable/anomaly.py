from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging
import os
import numpy as np

from ml_backend.smartstable_core.task_filtering import list_tasks_by_recording_period
from ml_backend.util.label_studio_helper import (
    ls_url_to_local_path,
    parse_ls_task_annotations,
)
from ml_backend.smartstable_core.predict import group_consecutive_segments
from ml_backend.smartstable_core.mlflow_support import (
    CentroidTrainingTracker,
    promote_centroid_to_production,
)
from ml_backend.ls_client import get_ls_client, LABEL_STUDIO_URL
from smartstablemodel.soundscape_classifier import SoundscapeClassifier
from smartstablemodel.labels import get_label_registry
from smartstablemodel.smartstable_types import MultiStallBatchResultReturn
from smartstablemodel.config import load_anomaly_config
from ml_audio_core.audio_processing import AudioProcessor

logger = logging.getLogger("AnomalyRouter")

anomaly_router = APIRouter(prefix="/anomaly", tags=["anomaly"])

# Status tracking
_anomaly_job = {"status": "idle", "job_id": None, "progress": 0, "message": ""}
_scan_job = {
    "status": "idle",
    "job_id": None,
    "progress": 0,
    "message": "",
    "results": [],
}


class TrainCentroidRequest(BaseModel):
    project_id: int = 1
    stable_id: str
    stall_id: str = "all"
    date_from: str
    date_until: str
    max_tasks: int = 200
    batch_size: int = 32
    include_human_speech: bool = True
    human_speech_samples: int = 400
    human_speech_language: str = "mixed"

    max_segments: int = 50000
    max_normal_segments: int = (
        20000  # Limit unannotated background to avoid drowning out specific sounds
    )


class ScanRequest(BaseModel):
    project_id: int = 1
    stable_id: str
    stall_id: str = "all"
    date_from: str
    date_until: str
    max_tasks: int = 200
    label_tasks: bool = False


def get_label_for_segment(
    start: float, end: float, annotations: List[Dict]
) -> Optional[str]:
    """
    Determine label for a segment based on overlap with annotations.
    Returns None if no significant annotation overlap is found.
    """
    best_label = None
    max_overlap = 0.0

    for ann in annotations:
        # Calculate intersection
        inter_start = max(start, ann["start"])
        inter_end = min(end, ann["end"])
        overlap = max(0, inter_end - inter_start)

        # If significant overlap found
        if overlap > 0:
            if overlap > max_overlap:
                max_overlap = overlap
                best_label = ann["label"]

    # Threshold: if we only caught < 0.1s overlap, likely edge noise, ignore.
    if max_overlap < 0.1:
        return None

    return best_label


def run_centroid_training(tasks, request: TrainCentroidRequest):
    global _anomaly_job
    try:
        _anomaly_job["status"] = "running"
        _anomaly_job["message"] = (
            f"Starting centroid training for {len(tasks)} tasks..."
        )
        logger.info(_anomaly_job["message"])

        # Initialize classifier and processor
        classifier = SoundscapeClassifier()
        audio_processor = AudioProcessor()

        # Load valid labels for anomaly
        valid_labels = set(get_label_registry().get_anomaly_detector_labels())
        logger.info(f"Using labels for anomaly baseline: {valid_labels}")

        from collections import defaultdict

        # Group embeddings by label
        embeddings_by_label = defaultdict(list)
        total_segments = 0
        normal_segments_count = 0
        max_segments_limit = request.max_segments
        max_normal_limit = request.max_normal_segments

        # Processing loop
        for i, task in enumerate(tasks):
            if total_segments >= max_segments_limit:
                logger.info(
                    f"Reached segment limit ({max_segments_limit}). Stopping collection."
                )
                break

            if i % 50 == 0:
                logger.info(
                    f"Progress: {i}/{len(tasks)} tasks processed. Collected {total_segments} segments..."
                )
                _anomaly_job["message"] = (
                    f"Processed {i}/{len(tasks)} tasks... ({total_segments} segments)"
                )

            url = task.data.get("audio")
            path = ls_url_to_local_path(url)
            if not os.path.exists(path):
                continue

            annotations = parse_ls_task_annotations(task)

            # Optimization: If no annotations, skip expensive audio loading
            if not annotations:
                continue

            try:
                # Load and split audio
                audio, _ = audio_processor.load_and_resample_audio(path)
                windows, times, _ = audio_processor.create_sliding_windows(
                    audio, segment_duration=2.0, overlap=0.0
                )

                # Filter windows by label
                task_segments_by_label = defaultdict(list)

                for window, (start, end) in zip(windows, times):
                    label = get_label_for_segment(start, end, annotations)

                    if label is None:
                        continue

                    # Special handling for 'normal' (unannotated) to prevent imbalance
                    if label == "normal":
                        if normal_segments_count >= max_normal_limit:
                            continue
                        normal_segments_count += 1

                    if label in valid_labels:
                        task_segments_by_label[label].append(window)

                # Extract embeddings for each label group in this task
                for label, segments in task_segments_by_label.items():
                    if segments:
                        # Batch extract
                        # Note: get_embeddings returns a list of vectors, but we can stack them later
                        # or keep them as list of arrays. classifier.get_embeddings returns np.ndarray of shape (N, 2048) usually.
                        embeddings = classifier.get_embeddings(segments, batch_size=32)
                        embeddings_by_label[label].append(embeddings)
                        total_segments += len(segments)

            except Exception as e:
                logger.warning(f"Error processing {path}: {e}")
                continue

        if not embeddings_by_label:
            raise ValueError("No valid segments found across processed tasks.")

        # Consolidate lists of arrays into single array per label
        final_embeddings_map = {}
        for label, embedding_list in embeddings_by_label.items():
            final_embeddings_map[label] = np.vstack(embedding_list)

        logger.info(
            f"Final data collection complete. Counts per label: { {k: v.shape[0] for k, v in final_embeddings_map.items()} }"
        )

        # Add human speech data if requested
        if request.include_human_speech:
            from smartstablemodel.human_speech_database import HumanSpeechDataset

            try:
                logger.info(
                    f"Loading {request.human_speech_samples} human speech samples..."
                )
                dataset = HumanSpeechDataset()
                speech_segments = dataset.prepare_speech_segments(
                    max_samples=request.human_speech_samples,
                    language=request.human_speech_language,
                )
                if speech_segments:
                    # prepare_speech_segments returns list of (audio, id)
                    speech_audio = [s[0] for s in speech_segments]
                    speech_embeddings = classifier.get_embeddings(
                        speech_audio, batch_size=32
                    )

                    # Add to "human_talking" label bucket
                    label_key = "human_talking"
                    if label_key in final_embeddings_map:
                        final_embeddings_map[label_key] = np.vstack(
                            [final_embeddings_map[label_key], speech_embeddings]
                        )
                    else:
                        final_embeddings_map[label_key] = speech_embeddings

                    logger.info(
                        f"Added {len(speech_embeddings)} human speech segments to '{label_key}' bucket."
                    )
            except Exception as e:
                logger.error(f"Failed to load human speech data: {e}")

        # Compute Centroid using the new balanced method
        logger.info("Computing final balanced centroid (Mean of Means)...")
        final_centroids = classifier.train_centroid_from_embeddings(
            final_embeddings_map
        )

        # Log to MLflow
        tracker = CentroidTrainingTracker()
        tracker.start_training_run(
            task_count=len(tasks),
            max_segments=request.max_segments,
            max_normal_segments=request.max_normal_segments,
            include_human_speech=request.include_human_speech,
            human_speech_samples=request.human_speech_samples,
            tags={"stable_id": request.stable_id, "stall_id": request.stall_id},
        )

        # Generate a timestamp for the filename
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save locally with timestamp
        final_path = classifier.save_normal_centroid(
            final_centroids, timestamp=timestamp
        )

        # Log to MLflow
        version = tracker.log_centroid_model(
            str(final_path), model_architecture=classifier.model_architecture
        )

        if version:
            logger.info(
                f"Centroid saved and registered in MLflow as version {version} at: {final_path}"
            )
            # Promote to champion for inference
            promote_centroid_to_production(
                model_architecture=classifier.model_architecture, version=version
            )
        else:
            logger.info(
                f"Centroid saved without MLflow version tracking to: {final_path}"
            )

        tracker.end_run("FINISHED")

        _anomaly_job["status"] = "completed"
        _anomaly_job["message"] = (
            f"Centroid training successful. Used {total_segments} task segments + speech. "
            f"Saved to disk."
        )
        logger.info(_anomaly_job["message"])

        # Reload in running backends to make it effective immediately
        from ml_backend.registry import PROJECT_BACKENDS
        from ml_backend.models.smartstable_backend import SmartStableBackend

        reloaded_count = 0
        for backend in PROJECT_BACKENDS.values():
            if isinstance(backend, SmartStableBackend):
                backend.manager.soundscape_classifier.load_normal_centroid()
                reloaded_count += 1

        if reloaded_count > 0:
            logger.info(
                f"Reloaded normal centroid in {reloaded_count} running backend instances"
            )

    except Exception as e:
        logger.error(f"Anomaly Training failed: {e}", exc_info=True)
        _anomaly_job["status"] = "failed"
        _anomaly_job["message"] = str(e)


@anomaly_router.post("/train_centroid")
def train_centroid(request: TrainCentroidRequest, background_tasks: BackgroundTasks):
    global _anomaly_job

    if _anomaly_job["status"] == "running":
        return {"status": "error", "message": "Training already in progress"}

    # 1. Fetch tasks (need full objects for annotations)
    # Support 'all' stables
    fetch_stable_id = None if request.stable_id == "all" else request.stable_id

    tasks = list_tasks_by_recording_period(
        project_id=request.project_id,
        stable_id=fetch_stable_id,
        stall_id=None if request.stall_id == "all" else request.stall_id,
        date_from=request.date_from,
        only_annotated=True,
        date_until=request.date_until,
        max_tasks=request.max_tasks,
        return_summary=False,  # We need annotations
    )

    if not tasks:
        raise HTTPException(400, "No tasks found for criteria.")

    background_tasks.add_task(run_centroid_training, tasks, request)

    return {
        "status": "started",
        "task_count": len(tasks),
        "message": "Background training started",
    }


@anomaly_router.get("/status")
def get_status():
    return _anomaly_job


def run_anomaly_scan(tasks, request: ScanRequest):
    global _scan_job
    try:
        _scan_job["status"] = "running"
        _scan_job["results"] = []
        _scan_job["message"] = f"Scanning {len(tasks)} tasks for anomalies..."
        logger.info(_scan_job["message"])

        audio_processor = AudioProcessor()
        classifier = SoundscapeClassifier()

        # Load anomaly config for threshold
        anomaly_config = load_anomaly_config()
        anomaly_threshold = anomaly_config.threshold

        anom_label = "anomaly_abnormal"
        anom_group = (
            get_label_registry().get_label_group_by_value(anom_label)
            or "anomaly_labels"
        )

        for i, task in enumerate(tasks):
            _scan_job["progress"] = int((i / len(tasks)) * 100)
            url = task.data.get("audio")
            path = ls_url_to_local_path(url)
            if not os.path.exists(path):
                continue

            try:
                audio, _ = audio_processor.load_and_resample_audio(path)
                windows, times, _ = audio_processor.create_sliding_windows(
                    audio, segment_duration=2.0, overlap=0.0
                )

                # Batch score anomalies
                scores = classifier.score_anomaly(windows)

                # Filter segments above threshold
                anomalous_segments = []
                for score, (start, end) in zip(scores, times):
                    if score > anomaly_threshold:
                        anomalous_segments.append(
                            MultiStallBatchResultReturn(
                                start_time=start,
                                end_time=end,
                                anomaly_score=float(score),
                                is_anomaly=True,
                                predicted_label=anom_label,
                            )
                        )

                if anomalous_segments:
                    # Group consecutive segments
                    events = group_consecutive_segments(
                        anomalous_segments, max_gap=2.1, probability_key="anomaly_score"
                    )

                    base_url = (LABEL_STUDIO_URL or "http://localhost:8080").rstrip("/")
                    task_results = {
                        "task_id": task.id,
                        "task_url": f"{base_url}/projects/{request.project_id}/data?task={task.id}",
                        "audio_path": path,
                        "anomalies": [
                            {
                                "start": e["start_time"],
                                "end": e["end_time"],
                                "score": e["avg_probability"],
                            }
                            for e in events
                        ],
                    }
                    _scan_job["results"].append(task_results)

                    if request.label_tasks:
                        # Add prediction to Label Studio
                        try:
                            ls = get_ls_client()
                            result = []
                            for e in events:
                                result.append(
                                    {
                                        "from_name": anom_group,
                                        "to_name": "audio",
                                        "type": "labels",
                                        "value": {
                                            "start": e["start_time"],
                                            "end": e["end_time"],
                                            "labels": [anom_label],
                                        },
                                        "score": e["avg_probability"],
                                    }
                                )

                            ls.predictions.create(
                                task=task.id,
                                result=result,
                                model_version=f"anomaly_scan_{classifier.get_model_version()}",
                            )
                            logger.info(f"Added anomaly labels to task {task.id}")
                        except Exception as lex:
                            logger.warning(
                                f"Failed to add LS labels for task {task.id}: {lex}"
                            )

            except Exception as e:
                logger.warning(f"Error scanning task {task.id}: {e}")
                continue
        if _scan_job["results"]:
            scores = [e["anomalies"][0]["score"] for e in _scan_job["results"]]
            max_score = max(scores)
            median_score = sorted(scores)[len(scores) // 2]
        else:
            max_score = 0
            median_score = 0
        _scan_job["status"] = "completed"
        _scan_job["progress"] = 100
        _scan_job["message"] = (
            f"Finished scanning {len(tasks)} tasks. Found anomalies in {len(_scan_job['results'])} tasks. "
            f"Maximum anomaly score: {max_score:.2f}, Median anomaly score: {median_score:.2f}"
        )
        logger.info(_scan_job["message"])

    except Exception as e:
        logger.error(f"Anomaly Scan failed: {e}", exc_info=True)
        _scan_job["status"] = "failed"
        _scan_job["message"] = str(e)


@anomaly_router.post("/scan")
def anomaly_scan(request: ScanRequest, background_tasks: BackgroundTasks):
    global _scan_job

    if _scan_job["status"] == "running":
        return {"status": "error", "message": "Scan already in progress"}

    tasks = list_tasks_by_recording_period(
        project_id=request.project_id,
        stable_id=request.stable_id,
        stall_id=None if request.stall_id == "all" else request.stall_id,
        date_from=request.date_from,
        date_until=request.date_until,
        max_tasks=request.max_tasks,
        return_summary=False,
    )

    if not tasks:
        raise HTTPException(400, "No tasks found for criteria.")

    background_tasks.add_task(run_anomaly_scan, tasks, request)

    return {
        "status": "started",
        "task_count": len(tasks),
        "message": "Background scanning started",
    }


@anomaly_router.get("/scan/status")
def get_scan_status():
    return _scan_job


@anomaly_router.get("/ui", response_class=HTMLResponse)
async def get_anomaly_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Anomaly Scan Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #2563eb;
                --primary-hover: #1d4ed8;
                --bg: #f8fafc;
                --card: #ffffff;
                --text: #1e293b;
                --text-light: #64748b;
                --border: #e2e8f0;
                --success: #22c55e;
                --danger: #ef4444;
            }
            body {
                font-family: 'Inter', sans-serif;
                background-color: var(--bg);
                color: var(--text);
                margin: 0;
                padding: 2rem;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .container {
                max-width: 800px;
                width: 100%;
                background: var(--card);
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            }
            h1 { margin-top: 0; font-weight: 600; color: #0f172a; }
            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
                margin-bottom: 1.5rem;
            }
            .form-group { display: flex; flex-direction: column; gap: 0.4rem; }
            label { font-size: 0.875rem; font-weight: 600; color: var(--text-light); }
            input {
                padding: 0.6rem;
                border: 1px solid var(--border);
                border-radius: 6px;
                font-size: 1rem;
            }
            button {
                padding: 0.75rem 1.5rem;
                background-color: var(--primary);
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.2s;
            }
            button:hover { background-color: var(--primary-hover); }
            button:disabled { background-color: var(--text-light); cursor: not-allowed; }
            #status-card {
                margin-top: 2rem;
                padding: 1.5rem;
                border-radius: 8px;
                border: 1px solid var(--border);
                display: none;
            }
            .status-running { border-left: 4px solid var(--primary); }
            .status-completed { border-left: 4px solid var(--success); }
            .status-failed { border-left: 4px solid var(--danger); }
            
            .progress-bar {
                height: 8px;
                background: var(--border);
                border-radius: 4px;
                overflow: hidden;
                margin: 1rem 0;
            }
            #progress-inner {
                height: 100%;
                background: var(--primary);
                width: 0%;
                transition: width 0.3s;
            }
            .result-item {
                padding: 0.75rem;
                border-bottom: 1px solid var(--border);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .result-item:last-child { border-bottom: none; }
            .tag {
                font-size: 0.75rem;
                padding: 0.2rem 0.5rem;
                border-radius: 9999px;
                background: #dcfce7;
                color: #166534;
            }
            a { color: var(--primary); text-decoration: none; font-weight: 500; }
            a:hover { text-decoration: underline; }
            .checkbox-group {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                grid-column: span 2;
                margin-top: 0.5rem;
            }
            .checkbox-group input { width: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Anomaly Scan</h1>
            <div class="grid">
                <div class="form-group">
                    <label>Project ID</label>
                    <input type="number" id="project_id" value="1">
                </div>
                <div class="form-group">
                    <label>Stable ID</label>
                    <input type="text" id="stable_id" value="stable01" placeholder="stable01">
                </div>
                <div class="form-group">
                    <label>Stall ID</label>
                    <input type="text" id="stall_id" value="all" placeholder="all or stall06">
                </div>
                <div class="form-group">
                    <label>Max Tasks</label>
                    <input type="number" id="max_tasks" value="100">
                </div>
                <div class="form-group">
                    <label>Date From</label>
                    <input type="date" id="date_from">
                </div>
                <div class="form-group">
                    <label>Date Until</label>
                    <input type="date" id="date_until">
                </div>
                <div class="checkbox-group">
                    <input type="checkbox" id="label_tasks">
                    <label for="label_tasks">Automatically label anomalous tasks in Label Studio</label>
                </div>
            </div>
            <button id="start-btn" onclick="startScan()">Start Global Scan</button>

            <div id="status-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 id="status-text" style="margin:0">Idle</h3>
                    <span id="progress-text">0%</span>
                </div>
                <div class="progress-bar">
                    <div id="progress-inner"></div>
                </div>
                <p id="message" style="color: var(--text-light); font-size: 0.9rem;"></p>
                <div id="results-list" style="margin-top: 1rem;"></div>
            </div>
        </div>

        <script>
            // Set default dates
            const today = new Date().toISOString().split('T')[0];
            const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];
            document.getElementById('date_from').value = yesterday;
            document.getElementById('date_until').value = today;

            let pollInterval;

            async function startScan() {
                const btn = document.getElementById('start-btn');
                const data = {
                    project_id: parseInt(document.getElementById('project_id').value),
                    stable_id: document.getElementById('stable_id').value,
                    stall_id: document.getElementById('stall_id').value,
                    date_from: document.getElementById('date_from').value,
                    date_until: document.getElementById('date_until').value,
                    max_tasks: parseInt(document.getElementById('max_tasks').value),
                    label_tasks: document.getElementById('label_tasks').checked
                };

                try {
                    const resp = await fetch('../anomaly/scan', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    const res = await resp.json();
                    if (res.status === 'started') {
                        showStatus();
                        startPolling();
                    } else {
                        alert(res.message || 'Failed to start scan');
                    }
                } catch (e) {
                    alert('Error: ' + e);
                }
            }

            function showStatus() {
                document.getElementById('status-card').style.display = 'block';
                document.getElementById('start-btn').disabled = true;
            }

            async function updateStatus() {
                try {
                    const resp = await fetch('../anomaly/scan/status');
                    const job = await resp.json();
                    
                    const card = document.getElementById('status-card');
                    card.className = 'status-' + job.status;
                    
                    document.getElementById('status-text').innerText = job.status.toUpperCase();
                    document.getElementById('progress-text').innerText = job.progress + '%';
                    document.getElementById('progress-inner').style.width = job.progress + '%';
                    document.getElementById('message').innerText = job.message;

                    if (job.results && job.results.length > 0) {
                        const list = document.getElementById('results-list');
                        list.innerHTML = '<h4>Findings (' + job.results.length + ')</h4>';
                        job.results.forEach(res => {
                            const div = document.createElement('div');
                            div.className = 'result-item';
                            const maxScore = Math.max(...res.anomalies.map(a => a.score));
                            div.innerHTML = `
                                <div>
                                    <a href="${res.task_url}" target="_blank">Task #${res.task_id}</a>
                                    <span style="font-size: 0.8rem; color: var(--text-light); margin-left: 0.5rem;">
                                        (${res.anomalies.length} segments)
                                    </span>
                                </div>
                                <span class="tag">Max Score: ${maxScore.toFixed(2)}</span>
                            `;
                            list.appendChild(div);
                        });
                    }

                    if (job.status !== 'running') {
                        stopPolling();
                        document.getElementById('start-btn').disabled = false;
                    }
                } catch (e) {
                    console.error('Polling error', e);
                }
            }

            function startPolling() {
                if (pollInterval) clearInterval(pollInterval);
                updateStatus();
                pollInterval = setInterval(updateStatus, 2000);
            }

            function stopPolling() {
                if (pollInterval) clearInterval(pollInterval);
            }

            // Check initial status
            updateStatus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
