from .metamodel_config import MetaModelConfig
from .anomaly_config import AnomalyConfig, load_anomaly_config

__all__ = [
    "MetaModelConfig",
    "load_metamodel_config",
    "AnomalyConfig",
    "load_anomaly_config",
]


def load_metamodel_config(
    path: str = "smart_stable_model_config.toml",
) -> MetaModelConfig:
    return MetaModelConfig.from_toml(path)
