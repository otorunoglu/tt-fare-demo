from pathlib import Path
import numpy as np
import os

class BlobStore:
    """Content-addressed audio store. Filenames = hash. Write-once, immutable."""
    def __init__(self, root):
        self.root = Path(root)

    def path_for(self, blob_hash: str) -> Path:
        # shard by 2-char prefix so one dir doesn't hold 15k+ files
        return self.root / blob_hash[:2] / f"{blob_hash}.npy"

    def exists(self, blob_hash: str) -> bool:
        return self.path_for(blob_hash).exists()
    
    def load(self, blob_hash: str) -> np.ndarray:
        return np.load(self.path_for(blob_hash))

    def write(self, blob_hash: str, audio: np.ndarray) -> Path:
        p = self.path_for(blob_hash)
        if p.exists():
            return p  # identical bytes already stored — content-addressed, leave it
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            np.save(f, np.ascontiguousarray(audio, dtype=np.float32))
        os.replace(tmp, p)  # atomic: never a half-written canonical blob
        return p