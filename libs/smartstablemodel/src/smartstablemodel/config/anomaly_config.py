from dataclasses import dataclass, fields
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib
import logging

logger = logging.getLogger(__name__)
CONFIG_DIR = Path(__file__).parent


@dataclass
class AnomalyConfig:
    enabled: bool = True
    threshold: float = 0.8
    normal_vocalization_threshold: float = 0.7

    @classmethod
    def from_toml(cls, path: str | Path) -> "AnomalyConfig":
        """Load from a specific TOML file."""
        p = Path(path)
        if not p.is_absolute():
            p = CONFIG_DIR / p

        if not p.exists():
            logger.warning(f"Anomaly config file not found: {p}")
            return cls()

        try:
            with open(p, "rb") as f:
                data = tomllib.load(f)

            if not data:
                return cls()

            # Filter to valid fields
            valid_keys = {f.name for f in fields(cls)}
            filtered_data = {k: v for k, v in data.items() if k in valid_keys}
            return cls(**filtered_data)
        except Exception as e:
            logger.error(f"Failed to load anomaly config from {p}: {e}")
            return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnomalyConfig":
        """Deprecated: Use from_toml."""
        return cls.from_toml(str(path).replace(".yaml", ".toml"))


def load_anomaly_config() -> AnomalyConfig:
    """Load anomaly config from standard locations."""
    possible_paths = [
        "/app/src/smartstablemodel/config/anomaly_config.toml",  # Docker mapped
        CONFIG_DIR / "anomaly_config.toml",  # Package default (legacy)
        CONFIG_DIR.parents[3]
        / "config"
        / "anomaly_config.toml",  # Monorepo root config
        "src/smartstablemodel/config/anomaly_config.toml",  # Local dev
    ]

    for path in possible_paths:
        p = Path(path)
        if p.exists():
            return AnomalyConfig.from_toml(p)

    return AnomalyConfig()
