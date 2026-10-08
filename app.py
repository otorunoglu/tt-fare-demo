import os
import json
import time
import traceback
import random
import re
import threading
from datetime import datetime
import subprocess
from pathlib import Path
import queue
from concurrent.futures import ThreadPoolExecutor


from flask import Flask, render_template, jsonify, send_file, request

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa

from mic_benchmark import mic_benchmark as mb


# ============================================================
# AUDIO SETTINGS
# ============================================================

SAMPLE_RATE = 16000

# RØDE microphone native recording rate
DEVICE_SAMPLE_RATE = 48000

# Recording duration
DURATION = 5

# How often the live waveform/spectrogram is recomputed
VISUALIZATION_INTERVAL = 0.1


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

os.makedirs("sounds", exist_ok=True)
os.makedirs("recordings", exist_ok=True)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

repo_root = Path(__file__).parent

config_path = (
    repo_root
    / "mic_benchmark"
    / "benchmark_config.toml"
)

cfg = mb.load_benchmark_config(
    config_path,
    repo_root
)

embedder_path = cfg["embedder"]
classifier_path = cfg["classifier"]
label_mapping_path = cfg["label_mapping"]
model_config_path = cfg["model_config"]

for required_file in (
    embedder_path,
    classifier_path,
    label_mapping_path
):
    if not Path(required_file).exists():
        raise SystemExit(
            f"Model file missing: {required_file}\n"
            "Copy the champion model files there "
            "(paths are set in mic_benchmark/benchmark_config.toml)."
        )

if not Path(model_config_path).exists():
    print(
        f"WARNING: {model_config_path} not found, "
        "falling back to default thresholds"
    )

segment_duration = float(
    cfg["segment_duration"]
)

segment_overlap = float(
    cfg["segment_overlap"]
)

silence_threshold, classifier_threshold = (
    mb.parse_model_config(
        model_config_path
    )
)

label_map = mb.load_label_mapping(
    label_mapping_path
)

mb.require_runtime_dependencies()


# ============================================================
# LOAD ONNX MODEL
# ============================================================

print(
    "Loading ONNX sessions for real champion model..."
)

session_options = mb.ort.SessionOptions()

# Leave one core free for audio capture, the live display
# and Flask; on a 4-core Pi the model would otherwise take all.
session_options.intra_op_num_threads = max(
    1,
    (os.cpu_count() or 4) - 1
)

# Don't busy-wait between runs (steals CPU from the audio).
session_options.add_session_config_entry(
    "session.intra_op.allow_spinning",
    "0"
)

embedder_sess = mb.ort.InferenceSession(
    str(embedder_path),
    session_options,
    providers=["CPUExecutionProvider"]
)

classifier_sess = mb.ort.InferenceSession(
    str(classifier_path),
    providers=["CPUExecutionProvider"]
)

emb_in_name = (
    embedder_sess
    .get_inputs()[0]
    .name
)

cls_in_name = (
    classifier_sess
    .get_inputs()[0]
    .name
)

print(
    f"Model loaded: {embedder_path.name}, "
    f"classifier threshold {classifier_threshold}, "
    f"silence threshold {silence_threshold} dBFS, "
    f"{session_options.intra_op_num_threads} threads"
)


# ============================================================
# HORSE SOUND CLASSES
# ============================================================

ALLOWED_CLASSES = [
    "drinking",
    "eating",
    "horse_kick",
    "neigh",
    "nicker",
    "rattle",
    "snort"
]


# ============================================================
# GLOBAL STATUS
# ============================================================

latest_status = {
    "action": "System Ready",
    "timestamp": "-",
    "filename": "-",
    "prediction": "Waiting for sound...",
    "confidence": "0%",
    "probabilities": {
        cls: 0
        for cls in ALLOWED_CLASSES
    }
}


# ============================================================
# LIVE VISUALIZATION
# ============================================================

# The browser shows the recording on a fixed 0..DURATION
# timeline: waveform and spectrogram fill in from the left and
# the playhead sits at the amount of audio actually recorded,
# so the animation stays in step with what comes out of the
# speakers.

audio_lock = threading.Lock()

RECORDING_SAMPLES = int(
    DURATION
    * DEVICE_SAMPLE_RATE
)

WAVEFORM_POINTS = 180

SPECTROGRAM_COLUMNS = 160
SPECTROGRAM_ROWS = 64
SPECTROGRAM_N_FFT = 2048

SPECTROGRAM_HOP = (
    RECORDING_SAMPLES
    // SPECTROGRAM_COLUMNS
)

SPECTROGRAM_WINDOW = np.hanning(
    SPECTROGRAM_N_FFT
).astype(np.float32)

# 0 - 8 kHz, the range the model sees after resampling to 16 kHz
SPECTROGRAM_BINS = int(
    (SAMPLE_RATE / 2)
    / (DEVICE_SAMPLE_RATE / SPECTROGRAM_N_FFT)
)

SPECTROGRAM_ROW_EDGES = np.linspace(
    0,
    SPECTROGRAM_BINS,
    SPECTROGRAM_ROWS + 1
).astype(int)

WAVEFORM_EDGES = np.linspace(
    0,
    RECORDING_SAMPLES,
    WAVEFORM_POINTS + 1
).astype(int)


# ------------------------------------------------------------
# Microphone audio of the current run (48 kHz)
# ------------------------------------------------------------

live_audio_buffer = np.zeros(
    0,
    dtype=np.float32
)


# ------------------------------------------------------------
# Data sent to browser
# ------------------------------------------------------------

def empty_visualization(active):
    return {
        "active": active,
        "waveform": [],
        "spectrogram": [],
        "sample_rate": SAMPLE_RATE,
        "min_frequency": 0,
        "max_frequency": SAMPLE_RATE // 2,
        "elapsed": 0,
        "duration": DURATION
    }


visualization_data = empty_visualization(
    False
)


# ============================================================
# RESET VISUALIZATION
# ============================================================

def reset_visualization():
    global live_audio_buffer
    global visualization_data

    with audio_lock:
        live_audio_buffer = np.zeros(
            0,
            dtype=np.float32
        )

        visualization_data = empty_visualization(
            True
        )


# ============================================================
# UPDATE LIVE VISUALIZATION
# ============================================================

def waveform_envelope(audio):
    # Peak level per point across the whole recording time;
    # points that haven't been recorded yet stay 0.
    envelope = np.zeros(
        WAVEFORM_POINTS,
        dtype=np.float32
    )

    for i in range(WAVEFORM_POINTS):
        segment = audio[
            WAVEFORM_EDGES[i]:
            WAVEFORM_EDGES[i + 1]
        ]

        if segment.size == 0:
            break

        envelope[i] = np.max(
            np.abs(segment)
        )

    # Normalise, without blowing quiet room noise up to full height
    return envelope / max(
        float(envelope.max()),
        0.05
    )


def spectrogram_columns(audio):
    # One column per time step (left to right), each column
    # SPECTROGRAM_ROWS values from low to high frequency in 0..1.
    columns = np.zeros(
        (SPECTROGRAM_COLUMNS, SPECTROGRAM_ROWS),
        dtype=np.float32
    )

    frames = min(
        (audio.size - SPECTROGRAM_N_FFT)
        // SPECTROGRAM_HOP
        + 1,
        SPECTROGRAM_COLUMNS
    )

    if frames <= 0:
        return columns

    starts = (
        np.arange(frames)
        * SPECTROGRAM_HOP
    )

    windows = (
        audio[
            starts[:, None]
            + np.arange(SPECTROGRAM_N_FFT)
        ]
        * SPECTROGRAM_WINDOW
    )

    power = np.abs(
        np.fft.rfft(windows, axis=1)[
            :,
            :SPECTROGRAM_BINS
        ]
    ) ** 2

    rows = np.add.reduceat(
        power,
        SPECTROGRAM_ROW_EDGES[:-1],
        axis=1
    )

    db = 10 * np.log10(rows + 1e-12)
    db -= db.max()

    columns[:frames] = np.clip(
        (db + 80) / 80.0,
        0.0,
        1.0
    )

    return columns


def update_visualization(
    audio_chunk
):
    global live_audio_buffer
    global visualization_data

    chunk = np.asarray(
        audio_chunk,
        dtype=np.float32
    )

    if chunk.ndim > 1:
        chunk = chunk[:, 0]

    with audio_lock:
        live_audio_buffer = np.concatenate(
            (
                live_audio_buffer,
                chunk.ravel()
            )
        )[:RECORDING_SAMPLES]

        audio = live_audio_buffer

    try:
        waveform = waveform_envelope(audio)
        spectrogram = spectrogram_columns(audio)
    except Exception as e:
        print(
            "Visualization error: "
            f"{e}"
        )
        return

    elapsed = (
        audio.size
        / DEVICE_SAMPLE_RATE
    )

    with audio_lock:
        visualization_data = {
            "active":
                elapsed < DURATION,
            "waveform":
                waveform.round(3).tolist(),
            "spectrogram":
                spectrogram.round(3).tolist(),
            "sample_rate":
                SAMPLE_RATE,
            "min_frequency":
                0,
            "max_frequency":
                SAMPLE_RATE // 2,
            "elapsed":
                round(elapsed, 2),
            "duration":
                DURATION
        }


# ============================================================
# FIND RØDE MICROPHONE
# ============================================================

def get_device_indices():
    devices = sd.query_devices()
    mic_idx = None

    for i, dev in enumerate(devices):
        name = dev["name"]

        if (
            "RØDE VideoMic GO II" in name
            and
            dev["max_input_channels"] > 0
        ):
            mic_idx = i
            break

    return mic_idx


# ============================================================
# FIND ALSA CARD
# ============================================================

def get_alsa_card_by_keyword(
    keyword
):
    try:
        result = subprocess.run(
            ["aplay", "-l"],
            capture_output=True,
            text=True,
            check=True
        )

        for line in (
            result.stdout.splitlines()
        ):
            if (
                keyword.lower()
                in line.lower()
            ):
                match = re.search(
                    r"card\s+(\d+):",
                    line
                )

                if match:
                    return match.group(1)

    except Exception as e:
        print(
            "Error finding ALSA card "
            f"for {keyword}: {e}"
        )

    return None


# ============================================================
# REAL AI MODEL
# ============================================================

def run_real_ai_model(
    audio_file_path
):
    audio_file = Path(
        audio_file_path
    )

    if not audio_file.exists():
        return (
            "Error: Audio file not found",
            {}
        )

    try:
        audio, sr = (
            mb.audio_preprocessing.load_audio(
                str(audio_file),
                SAMPLE_RATE
            )
        )

        # Only classify what the audience hears: the speakers
        # play the first DURATION seconds of the file. (The demo
        # sounds are up to 2 minutes long; classifying all of it
        # took up to ~30 s on the Pi and judged audio that was
        # never played.)
        audio = np.asarray(
            audio,
            dtype=np.float32
        )[:int(SAMPLE_RATE * DURATION)]

        segment_samples = int(
            SAMPLE_RATE
            * segment_duration
        )

        hop_samples = int(
            segment_samples
            * (1.0 - segment_overlap)
        )

        content_samples = audio.size

        # The model reads 2 s windows. Anything shorter than
        # that is right-padded with silence as a fallback (training mixes short events
        # onto stable background instead).
        if audio.size < segment_samples:
            audio = np.pad(
                audio,
                (0, segment_samples - audio.size)
            )

        starts = list(
            range(
                0,
                audio.size
                - segment_samples
                + 1,
                hop_samples
            )
        )

        # Like _window_segment in training: add a final window
        # aligned to the end if at least half a window is left
        # (and the last window doesn't already end there).
        if (
            starts[-1] + segment_samples
            < audio.size
            and
            audio.size
            - (starts[-1] + hop_samples)
            >= segment_samples // 2
        ):
            starts.append(
                audio.size
                - segment_samples
            )

        best_label = "unknown"
        highest_conf = 0.0
        best_probs = {}

        for start in starts:
            segment = audio[
                start:
                start + segment_samples
            ]

            # Gate on the real audio only, not the zero padding.
            dbfs = mb.loudness_dbfs(
                segment[
                    :max(
                        1,
                        content_samples - start
                    )
                ]
            )

            if dbfs < silence_threshold:
                continue

            audio_vec = np.clip(
                segment.astype(
                    np.float32
                ),
                -1.0,
                1.0
            )

            emb = embedder_sess.run(
                None,
                {
                    emb_in_name:
                    audio_vec.reshape(
                        1,
                        -1
                    )
                }
            )

            embedding = np.asarray(
                emb[0],
                dtype=np.float32
            )

            cls = classifier_sess.run(
                None,
                {
                    cls_in_name:
                    embedding
                }
            )

            probs = np.asarray(
                cls[0],
                dtype=np.float32
            ).reshape(-1)

            (
                label_probs,
                top_label,
                top_prob
            ) = mb.choose_top_label(
                probs,
                label_map
            )

            if top_prob > highest_conf:
                highest_conf = top_prob
                best_label = top_label
                best_probs = {
                    k.lower():
                    float(v * 100)
                    for k, v
                    in label_probs.items()
                }

        if (
            highest_conf
            < classifier_threshold
        ):
            return (
                "Unsure / Silence "
                f"(Conf: "
                f"{highest_conf:.2f})",
                {}
            )

        result_str = (
            f"{best_label.upper()} "
            f"("
            f"{highest_conf * 100:.1f}% "
            f"confidence"
            f")"
        )

        return (
            result_str,
            best_probs
        )

    except Exception as e:
        print(
            f"Inference error: {e}"
        )

        return (
            f"Error: {str(e)}",
            {}
        )


# ============================================================
# PREDICTION CACHE
# ============================================================

# The demo classifies the first DURATION seconds of the
# original sound file (what is played), so the result for a
# file never changes. It is computed once, kept in
# memory and on disk, and a Play only waits for the model when a
# file is new or has changed.

PREDICTION_CACHE_FILE = (
    repo_root
    / "prediction_cache.json"
)

# Held for a whole Play (or Replay); only one runs at a time.
run_lock = threading.Lock()

# One inference at a time; parallel runs only slow each other
# down on the Pi's CPU.
inference_lock = threading.Lock()

prediction_pool = ThreadPoolExecutor(
    max_workers=1
)


def load_prediction_cache():
    try:
        with open(
            PREDICTION_CACHE_FILE,
            encoding="utf-8"
        ) as f:
            return {
                key: tuple(value)
                for key, value
                in json.load(f).items()
            }
    except (OSError, ValueError):
        return {}


prediction_cache = load_prediction_cache()


def save_prediction_cache():
    tmp_file = PREDICTION_CACHE_FILE.with_suffix(
        ".tmp"
    )

    with open(
        tmp_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            prediction_cache,
            f,
            indent=1
        )

    os.replace(
        tmp_file,
        PREDICTION_CACHE_FILE
    )


def prediction_cache_key(
    sound_path
):
    stat = os.stat(sound_path)

    # A changed file, model or threshold gives a new key.
    return "|".join([
        Path(sound_path).name,
        str(stat.st_size),
        str(stat.st_mtime_ns),
        embedder_path.name,
        classifier_path.name,
        str(classifier_threshold),
        str(DURATION)
    ])


def predict_sound_file(
    sound_path
):
    key = prediction_cache_key(
        sound_path
    )

    if key in prediction_cache:
        return prediction_cache[key]

    with inference_lock:
        if key in prediction_cache:
            return prediction_cache[key]

        started = time.monotonic()

        result = run_real_ai_model(
            sound_path
        )

        print(
            f"Inference on {sound_path} took "
            f"{time.monotonic() - started:.1f} s: "
            f"{result[0]}"
        )

        if not result[0].startswith(
            "Error"
        ):
            prediction_cache[key] = result

            try:
                save_prediction_cache()
            except OSError as e:
                print(
                    "Could not save prediction "
                    f"cache: {e}"
                )

        return result


def warm_up_predictions():
    # Fill the cache in the background. It pauses while a Play
    # or Replay is running, so it never delays one by more than
    # the file it is currently working on.
    def warm_up():
        started = time.monotonic()

        for sound_file in sorted(
            Path("sounds").glob("*.wav")
        ):
            if sound_file.stem not in ALLOWED_CLASSES:
                continue

            while run_lock.locked():
                time.sleep(0.2)

            predict_sound_file(
                str(sound_file)
            )

        print(
            "Prediction warm-up finished in "
            f"{time.monotonic() - started:.1f} s"
        )

    threading.Thread(
        target=warm_up,
        daemon=True
    ).start()


# ============================================================
# STATUS
# ============================================================

def set_status(
    action,
    prediction,
    timestamp="-",
    filename="-",
    confidence="0%",
    probabilities=None
):
    global latest_status

    latest_status = {
        "action":
            action,
        "timestamp":
            timestamp,
        "filename":
            filename,
        "prediction":
            prediction,
        "confidence":
            confidence,
        "probabilities":
            probabilities
            or {
                cls: 0
                for cls
                in ALLOWED_CLASSES
            }
    }


# ============================================================
# SPEAKERS
# ============================================================

# Sound played by the last Play, for the Replay button
last_sound_path = None


def start_players(
    sound_path
):
    print("Playing sound via Jabra + VMK25")

    players = []

    # Jabra SPEAK 510 via ALSA
    jabra_card = get_alsa_card_by_keyword("Jabra")

    if jabra_card is not None:
        players.append(
            subprocess.Popen(
                [
                    "aplay",
                    "-D",
                    f"plughw:{jabra_card}",
                    sound_path
                ]
            )
        )
    else:
        print("Jabra device not found")

    # VMK25 via PipeWire
    players.append(
        subprocess.Popen(
            [
                "pw-play",
                sound_path
            ]
        )
    )

    return players


def stop_players(
    players,
    grace
):
    # Let the sound finish naturally, but never longer than grace
    deadline = time.monotonic() + grace

    for player in players:
        try:
            player.wait(
                timeout=max(
                    0,
                    deadline - time.monotonic()
                )
            )
        except subprocess.TimeoutExpired:
            player.terminate()
        except Exception:
            pass


# ============================================================
# PLAY AND RECORD
# ============================================================

def play_and_record():
    global last_sound_path

    reset_visualization()

    sound_files = [
        f.stem
        for f in Path("sounds").glob("*.wav")
        if f.stem in ALLOWED_CLASSES
    ]

    if not sound_files:
        raise RuntimeError(
            "No sound files found. Expected "
            "sounds/<class>.wav for one of: "
            + ", ".join(ALLOWED_CLASSES)
        )

    chosen_sound = random.choice(
        sound_files
    )

    sound_path = (
        f"sounds/{chosen_sound}.wav"
    )

    last_sound_path = sound_path

    print(
        "\n--- UI Triggered: "
        f"Random Sound "
        f"({chosen_sound.upper()}) ---"
    )

    # Classify the original file in the background while it is
    # played (instant when it is already in the cache).
    prediction = prediction_pool.submit(
        predict_sound_file,
        sound_path
    )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
    )

    filename = (
        f"{timestamp}_"
        f"{chosen_sound}.wav"
    )

    rec_filename = (
        f"recordings/{filename}"
    )

    set_status(
        "PLAYING & RECORDING",
        "RECORDING AUDIO...",
        timestamp,
        filename
    )

    mic_idx = (
        get_device_indices()
    )

    if mic_idx is not None:
        print(
            "Using RØDE microphone "
            f"device index {mic_idx}"
        )
        input_device = mic_idx
    else:
        print(
            "Warning: RØDE microphone "
            "not found. Using "
            "system default."
        )
        input_device = None

    recorded_chunks = []
    visualization_queue = queue.Queue()

    def recording_callback(
        indata,
        frames,
        callback_time,
        status
    ):
        if status:
            print(
                f"Recording status: "
                f"{status}"
            )

        chunk = indata.copy()

        recorded_chunks.append(
            chunk
        )

        visualization_queue.put(
            chunk
        )

    def visualization_worker():
        # The browser polls /audio-data every 100 ms, so redraw at
        # most that often; chunks arriving in between are merged
        # into one update.
        finished = False

        while not finished:
            chunk = visualization_queue.get()

            if chunk is None:
                break

            chunks = [chunk]

            while True:
                try:
                    next_chunk = (
                        visualization_queue.get_nowait()
                    )
                except queue.Empty:
                    break

                if next_chunk is None:
                    finished = True
                    break

                chunks.append(next_chunk)

            update_visualization(
                np.concatenate(
                    chunks,
                    axis=0
                )
            )

            time.sleep(
                VISUALIZATION_INTERVAL
            )

    visualization_thread = threading.Thread(
        target=visualization_worker,
        daemon=True
    )

    visualization_thread.start()

    print(
        "Starting REAL RØDE "
        f"microphone recording at "
        f"{DEVICE_SAMPLE_RATE} Hz "
        f"for {DURATION} seconds..."
    )

    players = []

    try:
        with sd.InputStream(
            samplerate=
                DEVICE_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=input_device,
            callback=
                recording_callback
        ):
            # Start the speakers only once the mic is live, so
            # the recording and the live display line up with
            # the sound.
            players = start_players(
                sound_path
            )

            time.sleep(
                DURATION
            )

    except Exception as e:
        print(
            "Microphone recording "
            f"error: {e}"
        )

    finally:
        visualization_queue.put(
            None
        )

    if not players:
        # No microphone: still play the sound for the audience
        players = start_players(
            sound_path
        )

    # Players can lag the mic slightly; let the end of the sound
    # play out in the background instead of cutting it off.
    threading.Thread(
        target=stop_players,
        args=(
            players,
            DURATION if not recorded_chunks else 1.0
        ),
        daemon=True
    ).start()

    visualization_thread.join(
        timeout=1
    )

    with audio_lock:
        visualization_data[
            "active"
        ] = False

    if recorded_chunks:
        recording = np.concatenate(
            recorded_chunks,
            axis=0
        ).flatten()
    else:
        print(
            "ERROR: No microphone "
            "data received."
        )

        recording = np.zeros(
            RECORDING_SAMPLES,
            dtype=np.float32
        )

    recording = recording[
        :RECORDING_SAMPLES
    ]

    resampled_recording = (
        librosa.resample(
            recording,
            orig_sr=
                DEVICE_SAMPLE_RATE,
            target_sr=
                SAMPLE_RATE
        )
    )

    sf.write(
        rec_filename,
        resampled_recording.reshape(
            -1,
            1
        ),
        SAMPLE_RATE
    )

    print(
        "Recording saved at "
        f"{SAMPLE_RATE} Hz:"
        f"\n{rec_filename}"
    )

    # ========================================================
    # REAL AI INFERENCE (DIRECTLY ON ORIGINAL SOUND FILE)
    # Started in the background when playback began. The
    # frontend keeps the result hidden until it is revealed.
    # ========================================================
    if not prediction.done():
        set_status(
            "ANALYSING",
            "AI IS ANALYSING...",
            timestamp,
            filename
        )

    ai_result, ai_probs = (
        prediction.result()
    )

    if "(" in ai_result:
        clean_pred = (
            ai_result.split(
                " ("
            )[0]
        )
    else:
        clean_pred = ai_result

    if "(" in ai_result:
        conf_val = (
            ai_result
            .split("(")[1]
            .replace(")", "")
        )
    else:
        conf_val = "0%"

    final_probs = {
        cls: 0.0
        for cls
        in ALLOWED_CLASSES
    }

    for k, v in ai_probs.items():
        matched_key = next(
            (
                cls
                for cls
                in ALLOWED_CLASSES
                if cls in k.lower()
            ),
            None
        )

        if matched_key:
            final_probs[
                matched_key
            ] = round(
                v,
                1
            )

    set_status(
        "COMPLETED",
        clean_pred,
        timestamp,
        filename,
        conf_val,
        final_probs
    )

    print(
        "Real AI Prediction: "
        f"{ai_result}"
    )


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template(
        "index.html"
    )


@app.route("/status")
def status():
    return jsonify(
        latest_status
    )


@app.route("/recording")
def recording():
    if (
        not latest_status.get(
            "filename"
        )
        or
        latest_status[
            "filename"
        ] == "-"
    ):
        return jsonify({
            "error":
                "No recording available"
        }), 404

    filepath = (
        Path("recordings")
        /
        latest_status[
            "filename"
        ]
    )

    if not filepath.exists():
        return jsonify({
            "error":
                "Recording not found"
        }), 404

    return send_file(
        filepath,
        mimetype="audio/wav"
    )


@app.route("/audio-data")
def audio_data():
    with audio_lock:
        data = {
            "active":
                visualization_data[
                    "active"
                ],
            "waveform":
                visualization_data[
                    "waveform"
                ],
            "spectrogram":
                visualization_data[
                    "spectrogram"
                ],
            "sample_rate":
                visualization_data[
                    "sample_rate"
                ],
            "min_frequency":
                visualization_data[
                    "min_frequency"
                ],
            "max_frequency":
                visualization_data[
                    "max_frequency"
                ],
            "elapsed":
                visualization_data[
                    "elapsed"
                ],
            "duration":
                visualization_data[
                    "duration"
                ]
        }

    return jsonify(
        data
    )


@app.route("/trigger/play")
def trigger_play():
    # Lagt til feilsøking for å se hvem som kaller ruten
    print(f">>> TRIGGERED! IP: {request.remote_addr} | Agent: {request.user_agent}")

    # Only one run at a time; a second trigger would start a
    # second recording and inference competing for the CPU.
    if not run_lock.acquire(
        blocking=False
    ):
        return jsonify({
            "success":
                False,
            "error":
                "A test or replay is already running"
        }), 409

    # Fresh status right away, so the browser never mistakes
    # the previous run's result or error for this one.
    set_status(
        "STARTING",
        "PREPARING..."
    )

    def run():
        try:
            play_and_record()
        except Exception as e:
            traceback.print_exc()
            set_status(
                "ERROR",
                f"ERROR: {e}"
            )
        finally:
            run_lock.release()

    worker = threading.Thread(
        target=run,
        daemon=True
    )
    worker.start()

    return jsonify({
        "success":
            True
    })


@app.route("/trigger/replay")
def trigger_replay():
    # Play the last sound again through the speakers (no
    # recording, no new prediction). Returns once it has finished.
    if last_sound_path is None:
        return jsonify({
            "success":
                False,
            "error":
                "No sound has been played yet"
        }), 404

    if not run_lock.acquire(
        blocking=False
    ):
        return jsonify({
            "success":
                False,
            "error":
                "A test or replay is already running"
        }), 409

    try:
        print(
            f"Replaying {last_sound_path}"
        )

        stop_players(
            start_players(
                last_sound_path
            ),
            grace=DURATION + 5
        )
    finally:
        run_lock.release()

    return jsonify({
        "success":
            True
    })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    warm_up_predictions()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )
