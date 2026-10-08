#!/usr/bin/env python3
"""List available input microphones for benchmark_config.toml device selection."""

from __future__ import annotations

from typing import Any


def main() -> int:
    try:
        import sounddevice as sd  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "sounddevice (and PortAudio) is required. Install PortAudio and `uv pip install sounddevice`."
        ) from exc

    devices: Any = sd.query_devices()
    print("Available audio input devices:")
    for idx, dev in enumerate(devices):
        if int(dev.get("max_input_channels", 0)) <= 0:
            continue
        print(
            f"  [{idx}] {dev['name']} "
            f"(default_sr={int(dev['default_samplerate'])}, "
            f"in_channels={int(dev['max_input_channels'])})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
