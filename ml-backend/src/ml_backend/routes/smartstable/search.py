from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import logging
import os
import numpy as np

from ml_backend.smartstable_core.task_filtering import list_tasks_by_recording_period
from ml_backend.util.label_studio_helper import (
    ls_url_to_local_path,
    parse_ls_task_annotations,
)
from ml_backend.smartstable_core.predict import group_consecutive_segments
from ml_backend.ls_client import get_ls_client, LABEL_STUDIO_URL
from smartstablemodel.soundscape_classifier import SoundscapeClassifier
from smartstablemodel.labels import get_label_registry
from smartstablemodel.smartstable_types import MultiStallBatchResultReturn
from ml_audio_core.audio_processing import AudioProcessor

logger = logging.getLogger("SearchRouter")

search_router = APIRouter(prefix="/search", tags=["search"])

# Status tracking
_search_job = {
    "status": "idle",
    "job_id": None,
    "progress": 0,
    "message": "",
    "results": [],
}
_cancel_requested = False


class SearchRequest(BaseModel):
    project_id: int = 1
    stable_id: str = "all"
    stall_id: str = "all"
    date_from: str
    date_until: str
    target_labels: list[str]
    threshold_min: float = 0.5
    max_tasks: int = 200
    label_tasks: bool = True


def run_label_scan(tasks, request: SearchRequest):
    global _search_job, _cancel_requested
    try:
        _cancel_requested = False
        _search_job["status"] = "running"
        _search_job["results"] = []
        labels_str = ", ".join(request.target_labels)
        _search_job["message"] = (
            f"Scanning {len(tasks)} tasks for labels: {labels_str}..."
        )
        logger.info(_search_job["message"])

        audio_processor = AudioProcessor()
        classifier = SoundscapeClassifier()

        label_registry = get_label_registry()
        # Get label groups for all target labels
        label_groups = {}
        for label in request.target_labels:
            label_groups[label] = label_registry.get_label_group_by_value(label) or "label"

        for i, task in enumerate(tasks):
            if _cancel_requested:
                _search_job["status"] = "cancelled"
                _search_job["message"] = (
                    f"Scan cancelled by user at {i}/{len(tasks)} tasks."
                )
                logger.info(_search_job["message"])
                return

            _search_job["progress"] = int((i / len(tasks)) * 100)
            url = task.data.get("audio")
            path = ls_url_to_local_path(url)
            if not os.path.exists(path):
                continue

            try:
                audio, _ = audio_processor.load_and_resample_audio(path)
                windows, times, _ = audio_processor.create_sliding_windows(
                    audio, segment_duration=2.0, overlap=0.0
                )

                # Batch predict
                # results is a list of probability dicts
                results = classifier.predict_batch(
                    np.array(windows), return_probabilities=True, batch_size=32
                )

                # Filter segments for any of the target labels and threshold
                # Group by label to preserve label information through grouping
                segments_by_label = {label: [] for label in request.target_labels}
                for res, (start, end) in zip(results, times):
                    # res is a dict mapping label name -> probability
                    for target_label in request.target_labels:
                        prob = res.get(target_label, 0.0)
                        if prob >= request.threshold_min:
                            segments_by_label[target_label].append(
                                MultiStallBatchResultReturn(
                                    start_time=start,
                                    end_time=end,
                                    probability=float(prob),
                                    predicted_label=target_label,
                                )
                            )

                # Group consecutive segments for each label and combine results
                all_events = []
                for label, segments in segments_by_label.items():
                    if segments:
                        grouped_events = group_consecutive_segments(
                            segments, max_gap=2.1, probability_key="probability"
                        )
                        for e in grouped_events:
                            e["predicted_label"] = label
                            all_events.append(e)

                if all_events:
                    # Sort by start time
                    all_events.sort(key=lambda x: x["start_time"])

                    base_url = (LABEL_STUDIO_URL or "http://localhost:8080").rstrip("/")
                    task_results = {
                        "task_id": task.id,
                        "task_url": f"{base_url}/projects/{request.project_id}/data?task={task.id}",
                        "audio_path": path,
                        "events": [
                            {
                                "start": e["start_time"],
                                "end": e["end_time"],
                                "score": e["avg_probability"],
                                "label": e["predicted_label"],
                            }
                            for e in all_events
                        ],
                    }
                    _search_job["results"].append(task_results)

                    if request.label_tasks:
                        # Add prediction to Label Studio
                        try:
                            ls = get_ls_client()
                            result = []
                            for e in all_events:
                                label = e["predicted_label"]
                                target_group = label_groups[label]
                                result.append(
                                    {
                                        "from_name": target_group,
                                        "to_name": "audio",
                                        "type": "labels",
                                        "value": {
                                            "start": e["start_time"],
                                            "end": e["end_time"],
                                            "labels": [label],
                                        },
                                        "score": e["avg_probability"],
                                    }
                                )

                            ls.predictions.create(
                                task=task.id,
                                result=result,
                                model_version=f"label_scan_{'_'.join(request.target_labels)}_{classifier.get_model_version()}",
                            )
                            logger.info(
                                f"Added pre-labels to task {task.id}"
                            )
                        except Exception as lex:
                            logger.warning(
                                f"Failed to add LS labels for task {task.id}: {lex}"
                            )

            except Exception as e:
                logger.warning(f"Error scanning task {task.id}: {e}")
                continue

        _search_job["status"] = "completed"
        _search_job["progress"] = 100
        labels_str = ", ".join(request.target_labels)
        _search_job["message"] = (
            f"Finished scanning {len(tasks)} tasks for {labels_str}. Found events in {len(_search_job['results'])} tasks."
        )
        logger.info(_search_job["message"])

    except Exception as e:
        logger.error(f"Label Scan failed: {e}", exc_info=True)
        _search_job["status"] = "failed"
        _search_job["message"] = str(e)


@search_router.post("/scan")
def search_scan(request: SearchRequest, background_tasks: BackgroundTasks):
    global _search_job

    if _search_job["status"] == "running":
        return {"status": "error", "message": "Scan already in progress"}

    tasks = list_tasks_by_recording_period(
        project_id=request.project_id,
        stable_id=None if request.stable_id == "all" else request.stable_id,
        stall_id=None if request.stall_id == "all" else request.stall_id,
        date_from=request.date_from,
        date_until=request.date_until,
        max_tasks=request.max_tasks,
        return_summary=False,
    )

    if not tasks:
        raise HTTPException(400, "No tasks found for criteria.")

    background_tasks.add_task(run_label_scan, tasks, request)

    return {
        "status": "started",
        "task_count": len(tasks),
        "message": "Background label scan started",
    }


@search_router.get("/scan/status")
def get_scan_status():
    return _search_job


@search_router.post("/scan/stop")
def stop_scan():
    global _cancel_requested
    if _search_job["status"] == "running":
        _cancel_requested = True
        return {"status": "stopping", "message": "Cancellation request sent"}
    return {"status": "ignored", "message": "No scan running"}


@search_router.get("/labels")
def get_labels():
    """Get list of available labels for scanning."""
    registry = get_label_registry()
    return {
        "labels": registry.get_label_values(),
        "display_labels": registry.get_display_labels(),
    }


@search_router.get("/stats")
def get_label_stats(project_id: int = 1):
    """
    Get distribution of currently labeled samples in the project.
    Iterates over all annotated tasks and counts labels.
    """
    try:
        from ml_backend.util.label_studio_helper import list_tasks_by_project

        ls = get_ls_client()

        # Get annotated tasks
        tasks = list_tasks_by_project(project_id, ls_client=ls, only_annotated=True)

        counts = {}
        for task in tasks:
            annotations = parse_ls_task_annotations(task)
            for ann in annotations:
                lbl = ann["label"]
                counts[lbl] = counts.get(lbl, 0) + 1

        # Sort by count descending
        sorted_counts = dict(
            sorted(counts.items(), key=lambda item: item[1], reverse=True)
        )

        return {
            "project_id": project_id,
            "annotated_tasks": len(tasks),
            "label_distribution": sorted_counts,
        }
    except Exception as e:
        logger.error(f"Failed to fetch label stats: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@search_router.get("/ui", response_class=HTMLResponse)
async def get_search_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Label Search Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #6366f1;
                --primary-hover: #4f46e5;
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
                max-width: 900px;
                width: 100%;
                background: var(--card);
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            }
            h1 { margin-top: 0; font-weight: 600; color: #0f172a; }
            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 1rem;
                margin-bottom: 1.5rem;
            }
            .form-group { display: flex; flex-direction: column; gap: 0.4rem; }
            label { font-size: 0.875rem; font-weight: 600; color: var(--text-light); }
            input, select {
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
            button.danger { background-color: var(--danger); }
            button.danger:hover { background-color: #dc2626; }
            
            .stats-container {
                margin-top: 2rem;
                padding: 1rem;
                background: #f1f5f9;
                border-radius: 8px;
            }
            .stats-grid {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-top: 0.5rem;
            }
            .stat-chip {
                background: white;
                padding: 0.3rem 0.6rem;
                border-radius: 6px;
                border: 1px solid var(--border);
                font-size: 0.8rem;
            }

            #status-card {
                margin-top: 2rem;
                padding: 1.5rem;
                border-radius: 8px;
                border: 1px solid var(--border);
                display: none;
            }
            .status-running { border-left: 4px solid var(--primary); }
            .status-completed { border-left: 4px solid var(--success); }
            .status-cancelled { border-left: 4px solid #f59e0b; }
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
                background: #e0e7ff;
                color: #3730a3;
            }
            a { color: var(--primary); text-decoration: none; font-weight: 500; }
            a:hover { text-decoration: underline; }
            .checkbox-group {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                grid-column: span 3;
                margin-top: 0.5rem;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔎 Rare Label Finder</h1>
            
            <div class="stats-container">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <label>Current Labeled Dataset Distribution</label>
                    <button onclick="loadStats()" style="padding: 0.3rem 0.7rem; font-size: 0.75rem; background: var(--text-light);">Refresh Stats</button>
                </div>
                <div id="stats-list" class="stats-grid">Loading stats...</div>
            </div>

            <hr style="margin: 2rem 0; border: none; border-top: 1px solid var(--border);">

            <div class="grid">
                <div class="form-group" style="grid-column: span 1;">
                    <label>Target Labels (Select one or more)</label>
                    <select id="target_labels" multiple style="min-height: 100px;">
                        <option value="">Loading labels...</option>
                    </select>
                    <small style="color: var(--text-light); margin-top: 0.2rem;">Hold Ctrl (Cmd on Mac) to select multiple</small>
                </div>
                <div class="form-group">
                    <label>Min Probability Threshold</label>
                    <input type="number" id="threshold_min" value="0.5" step="0.05" min="0" max="1">
                </div>
                <div class="form-group">
                    <label>Max Tasks to Scan</label>
                    <input type="number" id="max_tasks" value="200">
                </div>
                <div class="form-group">
                    <label>Stable ID</label>
                    <input type="text" id="stable_id" value="all">
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
                    <label for="label_tasks">Automatically add "Pre-labels" to Label Studio for findings</label>
                </div>
            </div>
            <div style="display: flex; gap: 1rem;">
                <button id="start-btn" onclick="startScan()" style="flex: 2;">Start Label Search</button>
                <button id="stop-btn" onclick="stopScan()" class="danger" style="flex: 1; display: none;">Stop Search</button>
            </div>

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
            const lastWeek = new Date(Date.now() - 7 * 86400000).toISOString().split('T')[0];
            document.getElementById('date_from').value = lastWeek;
            document.getElementById('date_until').value = today;

            let pollInterval;

            async function loadLabels() {
                try {
                    const resp = await fetch('labels');
                    const data = await resp.json();
                    const sel = document.getElementById('target_labels');
                    sel.innerHTML = '';
                    data.labels.forEach(lbl => {
                        const opt = document.createElement('option');
                        opt.value = lbl;
                        opt.innerText = data.display_labels[lbl] || lbl;
                        sel.appendChild(opt);
                    });
                } catch (e) {
                    console.error('Failed to load labels', e);
                }
            }

            async function loadStats() {
                const statsList = document.getElementById('stats-list');
                statsList.innerHTML = 'Refreshing...';
                try {
                    const resp = await fetch('stats?project_id=1');
                    const data = await resp.json();
                    statsList.innerHTML = '';
                    Object.entries(data.label_distribution).forEach(([lbl, count]) => {
                        const div = document.createElement('div');
                        div.className = 'stat-chip';
                        div.innerHTML = `<strong>${lbl}:</strong> ${count}`;
                        statsList.appendChild(div);
                    });
                } catch (e) {
                    statsList.innerHTML = 'Failed to load stats';
                }
            }

            async function startScan() {
                const btn = document.getElementById('start-btn');
                const selectElement = document.getElementById('target_labels');
                const selectedLabels = Array.from(selectElement.selectedOptions).map(opt => opt.value);
                
                if (selectedLabels.length === 0) {
                    alert('Please select at least one label to search for');
                    return;
                }
                
                const data = {
                    project_id: 1,
                    stable_id: document.getElementById('stable_id').value,
                    target_labels: selectedLabels,
                    threshold_min: parseFloat(document.getElementById('threshold_min').value),
                    date_from: document.getElementById('date_from').value,
                    date_until: document.getElementById('date_until').value,
                    max_tasks: parseInt(document.getElementById('max_tasks').value),
                    label_tasks: document.getElementById('label_tasks').checked
                };

                try {
                    const resp = await fetch('scan', {
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
                document.getElementById('results-list').innerHTML = '';
                document.getElementById('start-btn').disabled = true;
                document.getElementById('stop-btn').style.display = 'block';
            }

            async function stopScan() {
                try {
                    await fetch('scan/stop', { method: 'POST' });
                } catch (e) {
                    alert('Error stopping scan: ' + e);
                }
            }

            async function updateStatus() {
                try {
                    const resp = await fetch('scan/status');
                    const job = await resp.json();
                    
                    const card = document.getElementById('status-card');
                    card.className = 'status-' + job.status;
                    
                    document.getElementById('status-text').innerText = job.status.toUpperCase();
                    document.getElementById('progress-text').innerText = job.progress + '%';
                    document.getElementById('progress-inner').style.width = job.progress + '%';
                    document.getElementById('message').innerText = job.message;

                    if (job.status !== 'running') {
                        stopPolling();
                        document.getElementById('start-btn').disabled = false;
                        document.getElementById('stop-btn').style.display = 'none';
                    } else {
                        document.getElementById('start-btn').disabled = true;
                        document.getElementById('stop-btn').style.display = 'block';
                    }

                    const list = document.getElementById('results-list');
                    if (job.results && job.results.length > 0) {
                        list.innerHTML = '<h4>Findings (' + job.results.length + ')</h4>';
                        job.results.forEach(res => {
                            const div = document.createElement('div');
                            div.className = 'result-item';
                            const maxScore = Math.max(...res.events.map(a => a.score));
                            const labels = [...new Set(res.events.map(e => e.label))].join(', ');
                            div.innerHTML = `
                                <div>
                                    <a href="${res.task_url}" target="_blank">Task #${res.task_id}</a>
                                    <span style="font-size: 0.8rem; color: var(--text-light); margin-left: 0.5rem;">
                                        (${res.events.length} segments, Labels: ${labels})
                                    </span>
                                </div>
                                <span class="tag">Max Score: ${maxScore.toFixed(2)}</span>
                            `;
                            list.appendChild(div);
                        });
                    } else if (job.status === 'completed' || job.status === 'cancelled') {
                        list.innerHTML = '<p style="color: var(--text-light); text-align: center; margin-top: 1rem;">No findings for this scan.</p>';
                    } else if (job.status === 'running') {
                        list.innerHTML = '';
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

            // Init
            loadLabels();
            loadStats();
            updateStatus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
