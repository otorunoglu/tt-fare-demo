#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import torch
import librosa
import logging

# Setup paths
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_root / 'ml_audio_core' / 'src'))

from ml_audio_core.embedders.panns_embedder import PannsEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EMBED_DEBUG")

def main():
    # Load 1 second of audio
    # Load actual file
    audio_path = "../stable-ml-monitor-setup/data/stable03_stall01_horse01_20251214_000008_mic.flac"
    try:
        audio, _ = librosa.load(audio_path, sr=16000, duration=2.0)
    except:
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # 1. Standard Embedding
    logger.info("Running Standard Embedder...")
    emb_std_obj = PannsEmbedder(checkpoint_path="panns_data/Cnn14_16k_mAP=0.438.pth", device='cpu', onnx_export_mode=False)
    # We need to access internal model to process one segment without batch management overhead if possible,
    # but .embed() handles it.
    
    # embed expects (N, samples)
    batch = np.expand_dims(audio, axis=0)
    
    with torch.no_grad():
        # Get intermediate feature: LogMel Spectrogram?
        # The standard model doesn't easily expose it unless we hook it.
        # Let's just compare final embeddings first.
        emb_std = emb_std_obj.embed(batch)
        
    # 2. ONNX Mode Embedder (PyTorch implementation of the ONNX path)
    logger.info("Running ONNX-Mode Embedder...")
    emb_onnx_obj = PannsEmbedder(checkpoint_path="panns_data/Cnn14_16k_mAP=0.438.pth", device='cpu', onnx_export_mode=True)
    
    with torch.no_grad():
        emb_onnx = emb_onnx_obj.embed(batch)
        
    # Compare
    diff = np.abs(emb_std - emb_onnx)
    mse = np.mean((emb_std - emb_onnx)**2)
    mae = np.mean(diff)
    max_diff = np.max(diff)
    
    logger.info("=== Embedding Comparison ===")
    logger.info(f"Shape: {emb_std.shape}")
    logger.info(f"Standard Mean: {np.mean(emb_std):.6f} | Std: {np.std(emb_std):.6f}")
    logger.info(f"ONNX Mean:     {np.mean(emb_onnx):.6f} | Std: {np.std(emb_onnx):.6f}")
    logger.info("-" * 30)
    logger.info(f"MSE: {mse:.6f}")
    logger.info(f"MAE: {mae:.6f}")
    logger.info(f"Max Diff: {max_diff:.6f}")
    
    if mae > 0.1:
        logger.error("❌ FAILURE: Embeddings are significantly different!")
    else:
        logger.info("✅ SUCCESS: Embeddings match reasonably well.")

if __name__ == "__main__":
    main()
