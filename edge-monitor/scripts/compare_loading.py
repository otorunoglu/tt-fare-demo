import numpy as np
import librosa
import soundfile as sf
import audio_preprocessing
import os


def generate_sine_wave(filename, duration=1.0, sr=48000, freq=440.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * freq * t)
    sf.write(filename, x, sr)
    print(f"Generated {filename} (sr={sr}, duration={duration}s)")


def main():
    filename = "test_sine.wav"
    target_sr = 16000

    # Generate test file
    generate_sine_wave(filename)

    # Load with librosa
    print(f"Loading with librosa (target_sr={target_sr})...")
    y_librosa, sr_librosa = librosa.load(filename, sr=target_sr, mono=True)

    # Load with Rust extension
    print(f"Loading with audio_preprocessing (target_sr={target_sr})...")
    try:
        y_rust, sr_rust = audio_preprocessing.load_audio(filename, target_sr)
    except Exception as e:
        print(f"Rust loading failed: {e}")
        return

    print(f"Librosa shape: {y_librosa.shape}, sr: {sr_librosa}")
    print(f"Rust shape:    {y_rust.shape}, sr: {sr_rust}")

    # Compare lengths
    min_len = min(len(y_librosa), len(y_rust))
    y_librosa = y_librosa[:min_len]
    y_rust = y_rust[:min_len]

    # Compare
    diff = np.abs(y_librosa - y_rust)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")

    if max_diff < 0.01:
        print("SUCCESS: Waveforms are similar.")
    else:
        print("WARNING: Waveforms differ significantly.")

    # Cleanup
    if os.path.exists(filename):
        os.remove(filename)


if __name__ == "__main__":
    main()
