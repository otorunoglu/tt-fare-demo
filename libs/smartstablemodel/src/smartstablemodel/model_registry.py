import os
import json
from pathlib import Path

from .smartstable_types import SoundscapeClassifierConfig, StallConfig

# Default model paths
CAE_MODEL_PATH_DEFAULT = "cae_2s_latent_dim_32.keras"
CAE_PARAMS_PATH_DEFAULT = "cae_2s_latent_dim_32.keras.json"
CONFIG_PATH_DEFAULT = "default_config.json"

# Resolve base models directory from env or fallback to local models/
MODELS_BASE_DIR = os.getenv(
    "SMARTSTABLE_MODELS_DIR", str(Path(__file__).parent / "models")
)
MODELS_BASE_DIR = os.path.abspath(MODELS_BASE_DIR)


class ModelRegistry:
    def __init__(self, base_path: str | None = None):
        self.base_path = os.path.abspath(base_path or MODELS_BASE_DIR)
        self.loaded_stalls = {}
        self.default_config_path = os.path.join(
            self.base_path, "default", CONFIG_PATH_DEFAULT
        )

    def get_stall_config(self, stable_id, stall_id) -> StallConfig:
        """Get (or create) stall-specific configuration JSON."""
        stall_dir = self.get_model_path(stable_id, stall_id)
        config_path = os.path.join(stall_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config_data = json.load(f)
                return StallConfig(**config_data)
        print(
            f"Config not found for {stable_id}/{stall_id} at {config_path}; creating from defaults"
        )
        default_config: StallConfig = self.get_default_config()
        if default_config:
            default_config.stall_id = stall_id
            default_config.stable_id = stable_id
            config_json = default_config.model_dump()
            os.makedirs(stall_dir, exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(config_json, f, indent=2)
            return default_config
        else:
            print("No default config found, returning empty StallConfig")
            return StallConfig(stall_id=stall_id, stable_id=stable_id)

    def get_default_config(self) -> StallConfig:
        if os.path.exists(self.default_config_path):
            with open(self.default_config_path) as f:
                return StallConfig(**json.load(f))
        print(f"Default config not found at {self.default_config_path}")
        return StallConfig()

    def get_soundscape_classifier_config(
        self, stable_id, stall_id
    ) -> SoundscapeClassifierConfig:
        config = self.get_stall_config(stable_id, stall_id)
        return config.soundscape_classifier_config

    def get_model_path(self, stable_id, stall_id) -> str:
        """Return absolute path to stall-specific model directory."""
        return os.path.join(self.base_path, stable_id, stall_id)
