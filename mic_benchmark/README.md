# Microphone Benchmark

Quick benchmark tool for trying different microphones and distances with the current ONNX model pipeline.

It reads live microphone audio, runs the same Rust-backed resampling as the main system (`audio_preprocessing`), and prints JSON predictions to stdout.

## 1) Fresh Clone Setup (using uv)

Run these commands from repo root:

```bash
# 1. Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate

# 2. Install Python runtime dependencies for this script
uv pip install numpy sounddevice onnxruntime tomli maturin
```

### One-command setup helper (recommended for students)

From repo root:

```bash
# Microphone + file inference setup
uv run apps/microphone-benchmark/setup_benchmark_env.py --use-uv --mode mic --release

# File-only setup (no sounddevice/PortAudio dependency)
uv run apps/microphone-benchmark/setup_benchmark_env.py --use-uv --mode file --release
```

This installs Python dependencies and runs `maturin develop --features python`
inside `libs/audio_preprocessing` for the active environment.

### macOS system dependency

`sounddevice` needs PortAudio:

```bash
brew install portaudio
```

## 2) Build/install the Rust Python bridge (`audio_preprocessing`)

Still from repo root:

```bash
cd libs/audio_preprocessing
maturin develop --release --features python
cd ../..
```

This compiles and installs the local Rust extension into your active uv virtual environment.

## 3) Run the benchmark

List microphones (optional helper):

```bash
uv run apps/microphone-benchmark/list_microphones.py
```

Then open and edit config values in:

- `apps/microphone-benchmark/benchmark_config.toml`

Run benchmark:

```bash
uv run apps/microphone-benchmark/mic_benchmark.py
```

In VS Code, students can just press F5 while `mic_benchmark.py` is open.

To select a specific microphone, set `runtime.device` in the TOML file:

```bash
# by index
device = "2"

# or by partial name
device = "usb"
```

Run for a fixed time by editing TOML:

```bash
run_seconds = 30.0
```

## 4) Common options

- `runtime.log_threshold = 0.5`: minimum confidence to print
- `runtime.segment_duration = 2.0`: window size in seconds
- `runtime.segment_overlap = 0.5`: overlap between windows
- `runtime.sample_rate = 16000`: target model sample rate

Path values live in `paths.*` inside `benchmark_config.toml`.

If students want to start from defaults, keep this file as-is and only change:

- `runtime.device`
- `runtime.run_seconds`
- `runtime.log_threshold`

## 5) If imports fail after fresh pull

```bash
source .venv/bin/activate
uv pip install numpy sounddevice onnxruntime tomli maturin
cd libs/audio_preprocessing
maturin develop --release --features python
cd ../..
```

## Notes

- Default model paths point to `apps/edge-monitor/data/models`.
- If you updated Rust code in `libs/audio_preprocessing`, run `maturin develop --release --features python` again.
- The script prints one JSON line per emitted prediction, which is easy to copy into analysis notebooks later.
