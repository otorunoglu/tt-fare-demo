"""
Label definitions for horse sound classification.
Provides type-safe access to label values and metadata.
"""

from typing import Dict, List, Optional, Set
import os
import logging
from pathlib import Path
from .smartstable_types import Label

logger = logging.getLogger(__name__)


class LabelRegistry:
    """Central registry for managing labels with metadata."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_dir = Path(__file__).parent / "config"
            p = config_dir / "smart_stable_model_config.toml"
            if not p.exists():
                monorepo_p = (
                    config_dir.parents[3] / "config" / "smart_stable_model_config.toml"
                )
                if monorepo_p.exists():
                    p = monorepo_p
            self.config_path = p
        else:
            self.config_path = Path(config_path)
        self._labels: Dict[str, Label] = {}
        self._load_from_config()

    def _load_from_config(self):
        """Load labels from TOML configuration file."""
        try:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib

            if not os.path.exists(self.config_path):
                logger.error(f"Label config file not found: {self.config_path}")
                return

            with open(self.config_path, "rb") as f:
                # Always load as TOML for the main config, keep simple logic
                # If we really need legacy support for explicit JSON paths passed in:
                if str(self.config_path).endswith(".json"):
                    import json

                    config_data = json.load(f)
                else:
                    # Default / .toml
                    config_data = tomllib.load(f)

            # Extract labels from the config
            labels_data = config_data.get("labels", [])

            for label_data in labels_data:
                if not isinstance(label_data, dict):
                    logger.warning(f"Invalid label data format: {label_data}")
                    continue

                try:
                    label = Label(
                        name=label_data.get("name", ""),
                        value=label_data.get("value", ""),
                        ls_group=label_data.get("ls_group", "label"),
                        label_group=label_data.get("label_group", "other"),
                        is_alert=label_data.get("is_alert", False),
                        alert_confidence_threshold=label_data.get(
                            "alert_confidence_threshold", None
                        ),
                        alert_severity_score=label_data.get(
                            "alert_severity_score", None
                        ),
                        min_train_carrier_fraction=label_data.get(
                            "min_train_carrier_fraction", None
                        ),
                        is_cluster=label_data.get("is_cluster", False),
                        cluster_seconds_per_severity_level=label_data.get(
                            "cluster_seconds_per_severity_level", 0.0
                        ),
                        cluster_warning_min_severity=label_data.get(
                            "cluster_warning_min_severity", 1.0
                        ),
                        use_for_classifier_training=label_data.get(
                            "use_for_classifier_training", True
                        ),
                        use_for_anomaly_training=label_data.get(
                            "use_for_anomaly_training", False
                        ),
                        show_in_ls=label_data.get("show_in_ls", True),
                        description=label_data.get("description", ""),
                        merge_into=label_data.get("merge_into", None),
                        training_importance=label_data.get("training_importance", 1.0),
                        max_samples_per_class=label_data.get(
                            "max_samples_per_class", None
                        ),
                    )
                    self._labels[label.value] = label
                except ValueError as e:
                    logger.warning(f"Invalid label: {e}")
                    continue

            logger.info(f"Loaded {len(self._labels)} labels from {self.config_path}")

        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load label config from {self.config_path}: {e}")

    def get_label_group_by_value(self, value: str) -> Optional[str]:
        """Get the group of a label by its value."""
        label = self._labels.get(value)
        return label.ls_group if label else None

    def get_min_train_carrier_fraction(self, label_value: str) -> Optional[float]:
        """Get the minimum training carrier fraction for a label, if defined."""
        label = self._labels.get(label_value)
        if label and label.min_train_carrier_fraction is not None:
            # Example logic: higher importance means we want more samples of this class
            return label.min_train_carrier_fraction
        return None

    def create_mapping_from_labels(self, label_list: List[str]) -> Dict[str, Dict]:
        """Create label-to-index and index-to-label mappings from a list of labels."""
        # Validate all labels exist
        for label in label_list:
            if not self.validate_label(label):
                raise ValueError(f"Unknown label: {label}")

        label_to_idx = {label: idx for idx, label in enumerate(label_list)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        return {
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "valid_labels": label_list,  # type: ignore
        }

    def save_label_mapping(self, mapping: Dict, save_path: str):
        """Save label mapping to JSON file."""
        import json

        with open(save_path, "w") as f:
            json.dump(mapping, f, indent=2)

    def load_label_mapping(self, mapping_path: str) -> Optional[Dict]:
        """Load label mapping from JSON file."""
        import json

        if not os.path.exists(mapping_path):
            return None

        try:
            with open(mapping_path, "r") as f:
                mapping_data = json.load(f)

            # Convert string keys back to integers for idx_to_label
            return {
                "label_to_idx": mapping_data["label_to_idx"],
                "idx_to_label": {
                    int(k): v for k, v in mapping_data["idx_to_label"].items()
                },
                "valid_labels": mapping_data.get(
                    "valid_labels", list(mapping_data["label_to_idx"].keys())
                ),
            }
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load label mapping from {mapping_path}: {e}")
            return None

    def create_fallback_mapping(
        self, num_classes: int, model_type: str = "auto"
    ) -> Dict:
        """Create fallback mapping when no mapping file exists."""
        if num_classes <= 2:
            # Keep simple sane default if ever needed
            return {
                "label_to_idx": {"uncertain": 0, "horse_kick": 1},
                "idx_to_label": {0: "uncertain", 1: "horse_kick"},
                "valid_labels": ["uncertain", "horse_kick"],
            }
        else:
            # Multi-class: use trainable labels (excludes 'unsure')
            all_labels = self.get_trainable_label_values_for_classifier()[:num_classes]
            return self.create_mapping_from_labels(all_labels)

    def get_mapping_for_model(self, model_path: str, num_classes: int) -> Dict:
        """Get label mapping for a model. CRASHES if mapping is not found.

        IMPORTANT: We do NOT create fallback mappings anymore because they
        caused catastrophic label shifts where the model predicted one class
        but we interpreted it as a completely different class.

        If you're seeing this error, ensure your MLflow model has a label_mapping
        artifact, or provide an explicit label_mapping_path when loading the model.
        """
        # Try to load existing mapping
        mapping_path = model_path.replace(".keras", "_label_mapping.json")
        mapping = self.load_label_mapping(mapping_path)

        if mapping is not None:
            return mapping

        # NO FALLBACK - CRASH LOUDLY
        error_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        🚨 CRITICAL LABEL MAPPING ERROR 🚨                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ No label mapping found for model: {model_path[:50]}...
║ 
║ Expected mapping file at: {mapping_path[:50]}...
║ 
║ This is a FATAL error because using wrong labels causes the model to
║ predict one thing but we interpret it as something completely different!
║ 
║ Example of what goes wrong with fallback mappings:
║   - Model predicts index 3 = 'neigh' (how it was trained)
║   - Fallback mapping says index 3 = 'bird_tweet' (wrong order)
║   - Result: All neighs are classified as bird tweets and hidden!
║ 
║ TO FIX THIS:
║   1. Ensure your MLflow run has a 'label_mapping' or 'model_artifacts' 
║      artifact containing the *_label_mapping.json file
║   2. Or provide explicit label_mapping_path when loading the model
║   3. Or retrain the model - it will save the correct mapping
║ 
║ Model expects {num_classes} classes.
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def get_label_values(self) -> List[str]:
        """Get list of all label values."""
        return list(self._labels.keys())

    def get_anomaly_detector_labels(self) -> List[str]:
        """Get list of labels for the anomaly detector."""
        return [
            label.value
            for label in self._labels.values()
            if label.use_for_anomaly_training
        ]

    def get_labels_to_exclude_from_training_for_classifier(self) -> List[str]:
        """Get list of labels to exclude from training."""
        return [
            label.value
            for label in self._labels.values()
            if not label.use_for_classifier_training
        ]

    def get_label_remapping(self) -> Dict[str, str]:
        """
        Get label remapping for training.
        Returns a dict mapping source labels to their merge targets.
        E.g., {"other_impact": "horse_kick", "rolling": "change_stance"}
        """
        remapping = {}
        for label in self._labels.values():
            if label.merge_into:
                # Validate that the target label exists
                if label.merge_into in self._labels:
                    remapping[label.value] = label.merge_into
                else:
                    logger.warning(
                        f"Label '{label.value}' has merge_into='{label.merge_into}' but target doesn't exist"
                    )
        return remapping

    def get_merged_labels(self) -> Set[str]:
        """Get set of labels that are merged into other labels (should not appear as separate classes)."""
        return {label.value for label in self._labels.values() if label.merge_into}

    def get_training_importance_weights(self) -> Dict[str, float]:
        """
        Get training importance weights for each label.
        Returns a dict mapping label value to its training importance (default 1.0).
        Higher values mean the label is more important in weighted metrics.
        """
        return {
            label.value: label.training_importance for label in self._labels.values()
        }

    def get_max_samples_per_class_overrides(self) -> Dict[str, int]:
        """
        Get per-label training caps.
        Returns a dict mapping label value -> max samples for labels that define a
        positive max_samples_per_class override.
        """
        overrides: Dict[str, int] = {}
        for label in self._labels.values():
            cap = label.max_samples_per_class
            if cap is None:
                continue
            if cap <= 0:
                logger.warning(
                    f"Ignoring non-positive max_samples_per_class={cap} for label '{label.value}'"
                )
                continue
            overrides[label.value] = int(cap)
        return overrides

    def get_display_labels(self) -> Dict[str, str]:
        """
        Get display names for labels, showing merged labels in parentheses.

        For labels that have other labels merged into them, returns:
        "horse_kick" -> "horse_kick (other_impact, rattle)"

        For regular labels, returns the label as-is:
        "uncertain" -> "uncertain"

        Returns:
            Dict mapping label value to display name with merge info.
        """
        # Build reverse mapping: target -> list of sources merged into it
        merge_sources: Dict[str, List[str]] = {}
        for label in self._labels.values():
            if label.merge_into:
                if label.merge_into not in merge_sources:
                    merge_sources[label.merge_into] = []
                merge_sources[label.merge_into].append(label.value)

        # Build display names
        display_labels = {}
        for label in self._labels.values():
            if label.value in merge_sources:
                # This label has others merged into it
                merged_list = ", ".join(sorted(merge_sources[label.value]))
                display_labels[label.value] = f"{label.value} (+{merged_list})"
            else:
                display_labels[label.value] = label.value

        return display_labels

    def get_trainable_label_values_for_classifier(
        self, extra_exclude: Optional[List[str]] = None
    ) -> List[str]:
        """
        Labels eligible for training and fallback mappings.
        Excludes labels marked as not trainable AND labels that are merged into others.
        """
        excluded: Set[str] = set(
            self.get_labels_to_exclude_from_training_for_classifier()
        )
        # Also exclude labels that are merged into other labels
        excluded.update(self.get_merged_labels())
        if extra_exclude:
            excluded.update(extra_exclude)
        return [
            label.value
            for label in self._labels.values()
            if label.value not in excluded
        ]

    def get_labels_to_show_in_ls(self) -> List[str]:
        """Get list of labels to show in Label Studio."""
        return [label.value for label in self._labels.values() if label.show_in_ls]

    def get_label(self, value: str) -> Optional[Label]:
        """Get label object by value."""
        return self._labels.get(value)

    def validate_label(self, value: str) -> bool:
        """Validate if a label value exists."""
        return value in self._labels

    def get_label_description(self, value: str) -> Optional[str]:
        """Get description for a label."""
        label = self._labels.get(value)
        return label.description if label else None

    def get_all_labels(self) -> List[Label]:
        """Get all label objects."""
        return list(self._labels.values())
    
    def get_eval_only_labels(self) -> List[str]:
        """Get list of labels that are only used for evaluation, not training."""
        return [
            label.value
            for label in self._labels.values()
            if label.eval_only and not label.use_for_classifier_training
        ]


# Global registry instance
_registry = None


def get_label_registry() -> LabelRegistry:
    """Get the global label registry instance."""
    global _registry
    if _registry is None:
        _registry = LabelRegistry()
    return _registry


def get_all_labels() -> List[str]:
    """Get all label values."""
    return get_label_registry().get_label_values()


def get_trainable_labels_for_classifier() -> List[str]:
    """Get label values eligible for training (excludes 'unsure')."""
    return get_label_registry().get_trainable_label_values_for_classifier()


def validate_label(value: str) -> bool:
    """Validate if a label value exists."""
    return get_label_registry().validate_label(value)


def get_label_description(value: str) -> Optional[str]:
    """Get description for a label."""
    return get_label_registry().get_label_description(value)


def get_label_remapping() -> Dict[str, str]:
    """Get label remapping for training (from labels.json merge_into fields)."""
    return get_label_registry().get_label_remapping()


def get_training_importance_weights() -> Dict[str, float]:
    """Get training importance weights for weighted metrics."""
    return get_label_registry().get_training_importance_weights()


def get_max_samples_per_class_overrides() -> Dict[str, int]:
    """Get per-label max sample caps for training."""
    return get_label_registry().get_max_samples_per_class_overrides()


def get_display_labels() -> Dict[str, str]:
    """Get display names for labels, showing merged labels in parentheses."""
    return get_label_registry().get_display_labels()


def create_label_mapping(label_list: List[str]) -> Dict:
    """Create label mapping from list of labels."""
    return get_label_registry().create_mapping_from_labels(label_list)


def load_model_label_mapping(model_path: str, num_classes: int) -> Dict:
    """Load or create label mapping for a model."""
    return get_label_registry().get_mapping_for_model(model_path, num_classes)


def save_label_mapping(mapping: Dict, save_path: str):
    """Save label mapping to file."""
    get_label_registry().save_label_mapping(mapping, save_path)
