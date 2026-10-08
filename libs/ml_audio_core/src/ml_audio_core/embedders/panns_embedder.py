# ml_audio_core/embedders/panns_embedder.py

from pathlib import Path
import shutil
import urllib.request
import numpy as np
import torch

from tqdm import tqdm
from logging import getLogger

logger = getLogger(__name__)

# Monkey-patch torch.load for PANNs compatibility (PyTorch 2.6+)
_original_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)


torch.load = _patched_load

from ml_audio_core.embedders.base_embedder import BaseEmbedder


class PannsEmbedder(BaseEmbedder):
    """
    Wrapper around PANNs Cnn14 for embedding extraction.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        sample_rate: int = 16000,
        onnx_export_mode: bool = False,
    ):
        from panns_inference import AudioTagging
        from panns_inference.models import Cnn14

        # Allow numpy globals for PyTorch 2.6+ compatibility
        torch.serialization.add_safe_globals(
            [np.core.multiarray._reconstruct, np.ndarray, np.dtype]
        )

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            logger.warning(
                f"PANNs checkpoint not found at {self.checkpoint_path} downloading model ..."
            )
            if sample_rate == 16000:
                model_url = (
                    "https://zenodo.org/records/3987831/files/Cnn14_16k_mAP=0.438.pth"
                )
            elif sample_rate == 32000:
                model_url = (
                    "https://zenodo.org/records/3987831/files/Cnn14_mAP=0.431.pth"
                )
            else:
                raise ValueError(
                    "Unsupported sample rate for PANNs model. Use 16000 or 32000."
                )

            labels_url = "https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv"

            try:
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                _download_file(model_url, self.checkpoint_path)

                labels_path = self.checkpoint_path.parent / "class_labels_indices.csv"
                if not labels_path.exists():
                    _download_file(labels_url, labels_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download PANNs checkpoint to {self.checkpoint_path}"
                ) from exc

        self.device = device
        self.sample_rate = sample_rate

        # Create model with correct sample rate
        if sample_rate == 16000:
            # 16kHz model uses smaller FFT
            model = Cnn14(
                sample_rate=16000,
                window_size=512,
                hop_size=160,
                mel_bins=64,
                fmin=50,
                fmax=8000,
                classes_num=527,
            )
        else:
            logger.warning("Using 32kHz PANNs model for embedding extraction")
            # 32kHz model
            model = Cnn14(
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=527,
            )

        # Pass the custom model to AudioTagging
        self.model = AudioTagging(
            model=model, checkpoint_path=str(self.checkpoint_path), device=self.device
        )

        import torch.nn as nn
        import torch.nn.functional as F

        class Conv1dSTFT(nn.Module):
            def __init__(self, spec_layer, logmel_layer):
                super().__init__()
                self.n_fft = spec_layer.stft.n_fft
                self.hop_length = spec_layer.stft.hop_length
                self.center = spec_layer.stft.center
                self.pad_mode = spec_layer.stft.pad_mode or "reflect"

                # Initialize Weights
                # 1. DFT Matrix
                def dft_matrix(n):
                    (x, y) = np.meshgrid(np.arange(n), np.arange(n))
                    omega = np.exp(-2 * np.pi * 1j / n)
                    W = np.power(omega, x * y)
                    return W

                W = dft_matrix(self.n_fft)

                # 2. Window
                window = None
                if hasattr(spec_layer.stft, "window"):
                    w = spec_layer.stft.window
                    if isinstance(w, torch.Tensor):
                        window = w.cpu().numpy()

                if window is None:
                    window = torch.hann_window(self.n_fft).numpy()

                # Broadcast Multiply: (Freq, Time) * (Time,)
                W = W * window

                # 3. Cutoff & Split
                cutoff = self.n_fft // 2 + 1
                W = W[:cutoff]

                W_real = (
                    torch.from_numpy(np.real(W)).float().unsqueeze(1)
                )  # (Out, 1, Kernel)
                W_imag = torch.from_numpy(np.imag(W)).float().unsqueeze(1)

                self.register_buffer("W_real", W_real)
                self.register_buffer("W_imag", W_imag)

                # 4. Mel Matrix
                mel_W = logmel_layer.melW
                if isinstance(mel_W, torch.Tensor):
                    mel_W = mel_W.detach().cpu().float()
                else:
                    mel_W = torch.from_numpy(mel_W).float()

                self.register_buffer("mel_W", mel_W)

                # 5. Config
                self.amin = logmel_layer.amin
                self.ref_value = getattr(logmel_layer, "ref_value", 1.0)
                self.top_db = getattr(logmel_layer, "top_db", None)

            def forward(self, x):
                if x.dim() == 3:
                    x = x.squeeze(1)
                x = x.unsqueeze(1)

                # Padding (Center=True behavior)
                if self.center:
                    pad = self.n_fft // 2
                    x = F.pad(x, (pad, pad), mode=self.pad_mode)

                real = F.conv1d(x, self.W_real, stride=self.hop_length)
                imag = F.conv1d(x, self.W_imag, stride=self.hop_length)

                power_spec = real.pow(2) + imag.pow(2)
                power_spec = power_spec.transpose(1, 2)

                melspec = torch.matmul(power_spec, self.mel_W)
                melspec = torch.clamp(melspec, min=self.amin)

                log_melspec = 10.0 * torch.log10(melspec)
                log_melspec = log_melspec - 10.0 * np.log10(self.ref_value)

                if self.top_db is not None:
                    log_melspec = torch.maximum(
                        log_melspec, torch.max(log_melspec) - self.top_db
                    )

                return log_melspec.unsqueeze(1)

        # Replace internal layers with native Conv1dSTFT
        # Handle DataParallel wrapper if present
        if hasattr(self.model.model, "module"):
            cnn14 = self.model.model.module
        else:
            cnn14 = self.model.model

        cnn14.spectrogram_extractor = Conv1dSTFT(
            cnn14.spectrogram_extractor, cnn14.logmel_extractor
        )
        # Move the new layer to the same device as the rest of the model
        cnn14.spectrogram_extractor.to(self.device)

        cnn14.logmel_extractor = nn.Identity()

    @property
    def embedding_dim(self) -> int:
        return 2048  # Cnn14 hardcoded dim

    def embed(self, audio_batch: np.ndarray) -> np.ndarray:
        """
        audio_batch: (N, samples)
        returns embedding: (N, 2048)
        """
        _, emb = self.model.inference(audio_batch)
        return emb


def _download_file(url: str, dest: Path) -> None:
    """Download a file to destination path."""

    with urllib.request.urlopen(url) as response, open(dest, "wb") as f:
        total_size = int(response.headers.get("content-length", 0))

        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc=url.split("/")[-1]
        ) as pbar:
            chunk_size = 8192
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
