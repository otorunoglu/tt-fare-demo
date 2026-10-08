#!/usr/bin/env python3
"""One-command setup helper for microphone benchmark scripts.

Installs Python dependencies into the active environment and builds the local
Rust Python extension `audio_preprocessing` with maturin.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_command(name: str, install_hint: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Missing required command '{name}'. {install_hint}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install benchmark deps and build Rust audio_preprocessing module"
    )
    parser.add_argument(
        "--mode",
        choices=["file", "mic"],
        default="mic",
        help="file = file_inference only, mic = include microphone runtime deps",
    )
    parser.add_argument(
        "--use-uv",
        action="store_true",
        help="Use uv pip instead of python -m pip",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Build audio_preprocessing in release mode (default: debug for speed)",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    audio_preproc_dir = repo_root / "libs" / "audio_preprocessing"

    if not audio_preproc_dir.exists():
        raise FileNotFoundError(f"audio_preprocessing crate not found: {audio_preproc_dir}")

    ensure_command("cargo", "Install Rust toolchain (https://rustup.rs).")

    pip_cmd = [sys.executable, "-m", "pip"]
    if args.use_uv:
        ensure_command("uv", "Install uv: https://docs.astral.sh/uv/getting-started/installation/")
        pip_cmd = ["uv", "pip"]

    base_packages = ["numpy", "onnxruntime", "tomli", "maturin"]
    mic_packages = ["sounddevice"]

    packages = list(base_packages)
    if args.mode == "mic":
        packages.extend(mic_packages)

    print("Installing Python packages into current environment...")
    run(pip_cmd + ["install"] + packages)

    maturin_cmd = ["maturin", "develop", "--features", "python"]
    if args.release:
        maturin_cmd.insert(2, "--release")

    print("Building/installing Rust Python extension audio_preprocessing...")
    run(maturin_cmd, cwd=audio_preproc_dir)

    print("\nSetup complete.")
    print("Next steps:")
    print("  - File inference:")
    print(
        "    "
        + f"{sys.executable} apps/microphone-benchmark/file_inference.py "
        + "apps/microphone-benchmark/data/audio/mic1_src1_3_blue_Eating_2026-06-09_12-37-25.flac"
    )
    if args.mode == "mic":
        print("  - Microphone benchmark:")
        print("    " + f"{sys.executable} apps/microphone-benchmark/mic_benchmark.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
