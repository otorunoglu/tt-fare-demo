from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib

CONFIG_DIR = Path(__file__).parent


@dataclass
class MetaModelConfig:
    alert_labels: Dict[str, float]
    alert_confidence_thresholds: Dict[str, float]
    alert_labels_confidence_floor: float
    cluster_seconds_per_severity_level: Dict[str, float]
    default_cluster_confidence_floor: float
    cluster_confidence_floor_overrides: Dict[str, float]
    cluster_warning_min_severity_overrides: Dict[str, float]
    label_groups: Dict[str, list[str]]
    group_seconds_per_severity_level: Dict[str, float]
    model_parameters: Dict[str, Any]
    loudness_parameters: Dict[str, Any]
    # Top-level metadata from smart_stable_model_config.toml
    version: int | None = None
    description: str | None = None
    created_by: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_toml(
        cls, path: str | Path = "smart_stable_model_config.toml"
    ) -> "MetaModelConfig":
        from smartstablemodel.labels import get_label_registry

        if isinstance(path, str):
            p = CONFIG_DIR / path
            if not p.exists():
                monorepo_p = CONFIG_DIR.parents[3] / "config" / path
                if monorepo_p.exists():
                    p = monorepo_p
            path = p

        with open(path, "rb") as f:
            data: Dict[str, Any] = tomllib.load(f)

        # Override label-based configs from LabelRegistry (Single Source of Truth)
        registry = get_label_registry()
        all_labels = registry.get_all_labels()

        # Reconstruct alert_labels
        alert_labels = {
            label_obj.value: (
                label_obj.alert_severity_score
                if label_obj.alert_severity_score is not None
                else 1.0
            )
            for label_obj in all_labels
            if label_obj.is_alert
        }

        # Reconstruct per-label alert confidence thresholds
        alert_conf_thresholds = {}
        for label_obj in all_labels:
            if not label_obj.is_alert:
                continue
            if label_obj.alert_confidence_threshold is not None:
                alert_conf_thresholds[label_obj.value] = label_obj.alert_confidence_threshold

        # Reconstruct per-label cluster seconds per severity level
        cluster_seconds_per_severity_level = {
            label_obj.value: label_obj.cluster_seconds_per_severity_level
            for label_obj in all_labels
            if label_obj.is_cluster
        }

        # Reconstruct confidence overrides
        cluster_overrides = {
            label_obj.value: label_obj.cluster_confidence_floor
            for label_obj in all_labels
            if label_obj.cluster_confidence_floor is not None
        }

        cluster_warning_min_severity_overrides = {
            label_obj.value: label_obj.cluster_warning_min_severity
            for label_obj in all_labels
            if label_obj.is_cluster
        }

        # Reconstruct label_groups
        label_groups = {}
        for label_obj in all_labels:
            group = (
                label_obj.label_group
            )  # This is the "logic" group (normal, distress, etc)
            if group not in label_groups:
                label_groups[group] = []
            label_groups[group].append(label_obj.value)

        # Update data dict with reconstructed values
        data["alert_labels"] = alert_labels
        data["alert_confidence_thresholds"] = alert_conf_thresholds
        data["cluster_seconds_per_severity_level"] = cluster_seconds_per_severity_level
        data["cluster_confidence_floor_overrides"] = cluster_overrides
        data["cluster_warning_min_severity_overrides"] = cluster_warning_min_severity_overrides
        data["label_groups"] = label_groups

        # Ensure optional fields exist if missing in TOML (legacy compat)
        if "group_seconds_per_severity_level" not in data:
            data["group_seconds_per_severity_level"] = {}
        if "alert_labels_confidence_floor" not in data:
            data["alert_labels_confidence_floor"] = 0.6
        if "default_cluster_confidence_floor" not in data:
            data["default_cluster_confidence_floor"] = 0.95

        # Keep top-level config metadata accessible on the config object.
        metadata_keys = [
            "version",
            "description",
            "createdBy",
            "createdAt",
            "updatedAt",
        ]
        data["version"] = data.get("version")
        data["description"] = data.get("description")
        data["created_by"] = data.get("createdBy")
        data["created_at"] = data.get("createdAt")
        data["updated_at"] = data.get("updatedAt")
        data["metadata"] = {k: data[k] for k in metadata_keys if k in data}

        # Filter out keys not in the dataclass
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}

        return cls(**filtered_data)

    @classmethod
    def from_yaml(
        cls, path: str | Path = "smart_stable_model_config.yaml"
    ) -> "MetaModelConfig":
        """Deprecated: Use from_toml instead."""
        # For backward compatibility if needed, or redirect to toml if filename matches
        if str(path).endswith(".toml"):
            return cls.from_toml(path)
        # If we really need to support YAML still:
        import yaml

        path = CONFIG_DIR / path if isinstance(path, str) else path
        with open(path, "r", encoding="utf-8") as f:
            yaml.safe_load(f)
        # ... logic repetition avoid ...
        # For now, let's assume valid migration means we switch consumers to use from_toml
        # or we update this to load toml by default.
        # But to be safe, I'll redirect to from_toml if the file is strictly the new default
        return cls.from_toml("smart_stable_model_config.toml")
