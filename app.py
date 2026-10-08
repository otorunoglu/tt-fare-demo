import os
import time
import random
import re
import threading
from datetime import datetime
import subprocess
from pathlib import Path
import queue


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

embedder_sess = mb.ort.InferenceSession(
    str(embedder_path),
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

audio_lock = threading.Lock()


# ------------------------------------------------------------
# Rolling microphone buffer
# ------------------------------------------------------------

live_audio_buffer = np.array(
    [],
    dtype=np.float32
)

LIVE_BUFFER_SECONDS = 5

LIVE_BUFFER_SAMPLES = (
    DEVICE_SAMPLE_RATE
    * LIVE_BUFFER_SECONDS
)


# ------------------------------------------------------------
# Data sent to browser
# ------------------------------------------------------------

visualization_data = {
    "active": False,
    "waveform": [],
    "spectrogram": [],
    "sample_rate": SAMPLE_RATE,
    "min_frequency": 0,
    "max_frequency": SAMPLE_RATE // 2,
    "elapsed": 0,
    "duration": DURATION
}


# ============================================================
# RESET VISUALIZATION
# ============================================================

def reset_visualization():
    global live_audio_buffer
    global visualization_data

    with audio_lock:
        live_audio_buffer = np.array(
            [],
            dtype=np.float32
        )

        visualization_data = {
            "active": True,
            "waveform": [],
            "spectrogram": [],
            "sample_rate": SAMPLE_RATE,
            "min_frequency": 0,
            "max_frequency": SAMPLE_RATE // 2,
            "elapsed": 0,
            "duration": DURATION
        }


# ============================================================
# UPDATE LIVE VISUALIZATION
# ============================================================

def update_visualization(
    audio_chunk,
    elapsed
):
    global live_audio_buffer
    global visualization_data

    if audio_chunk is None:
        return

    chunk = np.asarray(
        audio_chunk,
        dtype=np.float32
    )

    if chunk.ndim > 1:
        chunk = chunk[:, 0]

    chunk = chunk.flatten()

    if len(chunk) == 0:
        return

    with audio_lock:
        live_audio_buffer = np.concatenate(
            (
                live_audio_buffer,
                chunk
            )
        )

        if (
            len(live_audio_buffer)
            > LIVE_BUFFER_SAMPLES
        ):
            live_audio_buffer = (
                live_audio_buffer[
                    -LIVE_BUFFER_SAMPLES:
                ]
            )

        current_audio = (
            live_audio_buffer.copy()
        )

    if len(current_audio) < 1024:
        return

    try:
        audio_16k = librosa.resample(
            current_audio,
            orig_sr=DEVICE_SAMPLE_RATE,
            target_sr=SAMPLE_RATE
        )
    except Exception as e:
        print(
            "Visualization resampling "
            f"error: {e}"
        )
        return

    waveform_points = 180

    if len(audio_16k) > waveform_points:
        waveform_indices = np.linspace(
            0,
            len(audio_16k) - 1,
            waveform_points
        ).astype(int)

        waveform = (
            audio_16k[
                waveform_indices
            ]
        )
    else:
        waveform = audio_16k

    waveform_max = np.max(
        np.abs(waveform)
    )

    if waveform_max > 0:
        waveform = (
            waveform / waveform_max
        )

    waveform = np.clip(
        waveform,
        -1.0,
        1.0
    )

    n_fft = 512
    hop_length = 128

    if len(audio_16k) >= n_fft:
        try:
            stft = librosa.stft(
                audio_16k,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                center=False
            )

            magnitude = np.abs(
                stft
            )

            db = librosa.amplitude_to_db(
                magnitude,
                ref=np.max
            )

            db = np.clip(
                db,
                -80,
                0
            )

            normalized = (
                db + 80
            ) / 80.0

            normalized = np.flipud(
                normalized
            )

            target_rows = 64
            target_cols = 160

            rows = normalized.shape[0]
            cols = normalized.shape[1]

            row_indices = np.linspace(
                0,
                rows - 1,
                target_rows
            ).astype(int)

            number_of_columns = min(
                target_cols,
                cols
            )

            col_indices = np.linspace(
                0,
                cols - 1,
                number_of_columns
            ).astype(int)

            spectrogram = normalized[
                np.ix_(
                    row_indices,
                    col_indices
                )
            ]

            spectrogram = (
                spectrogram.tolist()
            )

        except Exception as e:
            print(
                "Spectrogram calculation "
                f"error: {e}"
            )
            spectrogram = []
    else:
        spectrogram = []

    with audio_lock:
        visualization_data = {
            "active":
                elapsed < DURATION,
            "waveform":
                waveform.tolist(),
            "spectrogram":
                spectrogram,
            "sample_rate":
                SAMPLE_RATE,
            "min_frequency":
                0,
            "max_frequency":
                SAMPLE_RATE // 2,
            "elapsed":
                round(
                    elapsed,
                    2
                ),
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

        audio = np.asarray(
            audio,
            dtype=np.float32
        )

        segment_samples = int(
            SAMPLE_RATE
            * segment_duration
        )

        hop_samples = int(
            segment_samples
            * (1.0 - segment_overlap)
        )

        if audio.size < segment_samples:
            return (
                "Audio too short for inference",
                {}
            )

        best_label = "unknown"
        highest_conf = 0.0
        best_probs = {}

        for start in range(
            0,
            audio.size
            - segment_samples
            + 1,
            hop_samples
        ):
            segment = audio[
                start:
                start + segment_samples
            ]

            dbfs = mb.loudness_dbfs(
                segment
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
# PLAY AND RECORD
# ============================================================

def play_and_record():
    global latest_status

    reset_visualization()

    sounds_dir = Path("sounds")
    sound_files = [
        f.stem
        for f in sounds_dir.glob("*.wav")
        if f.stem in ALLOWED_CLASSES
    ]

    if not sound_files:
        sound_files = ALLOWED_CLASSES

    chosen_sound = random.choice(
        sound_files
    )

    sound_path = (
        f"sounds/{chosen_sound}.wav"
    )

    if not os.path.exists(
        sound_path
    ):
        print(
            f"Error: {sound_path} "
            "not found!"
        )
        return

    print(
        "\n--- UI Triggered: "
        f"Random Sound "
        f"({chosen_sound.upper()}) ---"
    )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
    )

    rec_filename = (
        f"recordings/"
        f"{timestamp}_"
        f"{chosen_sound}.wav"
    )

    latest_status = {
        "action":
            "PLAYING & RECORDING",
        "timestamp":
            timestamp,
        "filename":
            f"{timestamp}_"
            f"{chosen_sound}.wav",
        "prediction":
            "RECORDING AUDIO...",
        "confidence":
            "0%",
        "probabilities":
            {
                cls: 0
                for cls
                in ALLOWED_CLASSES
            }
    }

    players = []

    print("Playing sound via Jabra + VMK25")

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

        elapsed = (
            time.monotonic()
            -
            recording_callback.start_time
        )

        if elapsed <= DURATION:
            visualization_queue.put(
    (
        chunk,
        elapsed
    )
)

    recording_callback.start_time = (
        time.monotonic()
    )
    def visualization_worker():
        while True:
            try:
                chunk, elapsed = visualization_queue.get()

                if chunk is None:
                    break

                update_visualization(
                    chunk,
                    elapsed
                )

            except Exception as e:
                print(
                    "Visualization worker "
                    f"error: {e}"
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

    try:
        # Standard blocksize (eller fjernet for automatisk standard)
        with sd.InputStream(
            samplerate=
                DEVICE_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=input_device,
            callback=
                recording_callback
        ):
            time.sleep(
                DURATION
            )

    except Exception as e:
        print(
            "Microphone recording "
            f"error: {e}"
        )

    finally:
        with audio_lock:
            visualization_data[
                "active"
            ] = False
            visualization_data[
                "elapsed"
            ] = DURATION

    visualization_queue.put(
        (None, None)
    )

    visualization_thread.join(
        timeout=1
    )


    for player in players:
        try:
            player.terminate()
        except Exception:
            pass

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
            int(
                DURATION
                * DEVICE_SAMPLE_RATE
            ),
            dtype=np.float32
        )

    expected_samples = int(
        DURATION
        * DEVICE_SAMPLE_RATE
    )

    recording = recording[
        :expected_samples
    ]

    print(
        "Resampling RØDE recording "
        "from 48 kHz to 16 kHz..."
    )

    resampled_recording = (
        librosa.resample(
            recording,
            orig_sr=
                DEVICE_SAMPLE_RATE,
            target_sr=
                SAMPLE_RATE
        )
    )

    resampled_recording = (
        resampled_recording.reshape(
            -1,
            1
        )
    )

    sf.write(
        rec_filename,
        resampled_recording,
        SAMPLE_RATE
    )

    print(
        "Recording saved at "
        f"{SAMPLE_RATE} Hz:"
        f"\n{rec_filename}"
    )

    latest_status = {
        "action":
            "GUESSING TIME",
        "timestamp":
            timestamp,
        "filename":
            f"{timestamp}_"
            f"{chosen_sound}.wav",
        "prediction":
            "GUESS THE SOUND...",
        "confidence":
            "0%",
        "probabilities":
            {
                cls: 0
                for cls
                in ALLOWED_CLASSES
            }
    }

    time.sleep(5)

    # ========================================================
    # REAL AI INFERENCE (DIRECTLY ON ORIGINAL SOUND FILE)
    # ========================================================
    ai_result, ai_probs = (
        run_real_ai_model(
            sound_path
        )
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

    latest_status = {
        "action":
            "COMPLETED",
        "timestamp":
            timestamp,
        "filename":
            f"{timestamp}_"
            f"{chosen_sound}.wav",
        "prediction":
            clean_pred,
        "confidence":
            conf_val,
        "probabilities":
            final_probs
    }

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
    
    worker = threading.Thread(
        target=play_and_record,
        daemon=True
    )
    worker.start()

    return jsonify({
        "success":
            True
    })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )
