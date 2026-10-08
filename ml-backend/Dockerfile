FROM ghcr.io/astral-sh/uv:latest AS uv_bin
FROM panns-base:latest

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

# Copy uv binary
COPY --from=uv_bin /uv /uvx /bin/

# Install system dependencies including Rust build tools
RUN apt-get update && apt-get install -y ffmpeg curl rclone build-essential pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 1. Install External Dependencies (The slow part - Cached)
COPY pyproject.toml uv.lock ./
COPY apps/ml-backend/pyproject.toml ./apps/ml-backend/
COPY libs/smartstablemodel/pyproject.toml ./libs/smartstablemodel/
COPY libs/ml_audio_core/pyproject.toml ./libs/ml_audio_core/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages -r apps/ml-backend/pyproject.toml

# 2. Build and Install Rust Audio Preprocessing Crate
# Copy the libs directory (assuming context is set to repo root)
COPY libs/audio_preprocessing /src/libs/audio_preprocessing

# Build the wheel
WORKDIR /src/libs/audio_preprocessing
RUN uv pip install --system --break-system-packages maturin
RUN maturin build -m Cargo.toml --release --features python

# Install the wheel
RUN uv pip install --system --break-system-packages target/wheels/audio_preprocessing*.whl

# Return to app dir
WORKDIR /app

# 3. Copy Source Code
COPY apps/ml-backend/src/ ./src/

# 4. Install Local Project (The fast part)
RUN uv pip install --system --break-system-packages -e apps/ml-backend/
RUN uv pip install --system --break-system-packages debugpy
RUN uv pip install --system --break-system-packages pytest

CMD ["python", "-m", "ml_backend.ml_backend"]