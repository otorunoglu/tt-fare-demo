from datetime import datetime
from pydantic import BaseModel, Field, root_validator, validator
from typing import Dict, List, Optional, Any
from .generated_types import LabelProbabilities


class SoundscapeClassifierConfig(BaseModel):
    """
    Configuration for the soundscape classifier.
    """

    # Deprecated: will be migrated into thresholds['horse_kick'] if provided
    kick_prop_threshold: Optional[float] = Field(
        default=None,
        description="DEPRECATED. Use thresholds['horse_kick'] instead.",
    )

    # Default threshold used when a label-specific threshold is not set
    default_threshold: float = Field(
        default=0.9, description="Default probability threshold for class acceptance."
    )

    # Per-label probability thresholds, e.g. {'horse_kick': 0.9, 'moving': 0.7}
    thresholds: Dict[str, float] = Field(
        default_factory=dict, description="Per-label probability thresholds."
    )

    @root_validator(pre=True)
    def _migrate_kick_threshold(cls, values):
        # Migrate old kick_prop_threshold into thresholds['horse_kick']
        thr = values.pop("kick_prop_threshold", None)
        if thr is not None:
            thresholds = dict(values.get("thresholds") or {})
            thresholds.setdefault("horse_kick", thr)
            values["thresholds"] = thresholds
        return values

    @validator("default_threshold")
    def _validate_default(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("default_threshold must be in [0, 1]")
        return float(v)

    @validator("thresholds")
    def _validate_thresholds(cls, v):
        cleaned = {}
        for k, val in v.items():
            fv = float(val)
            if not 0.0 <= fv <= 1.0:
                raise ValueError(f"threshold for '{k}' must be in [0, 1]")
            cleaned[k] = fv
        return cleaned

    def get_threshold(self, label: str, fallback: Optional[float] = None) -> float:
        """Return threshold for a label or default."""
        if label in self.thresholds:
            return float(self.thresholds[label])
        return float(self.default_threshold if fallback is None else fallback)

    class Config:
        extra = "ignore"


class AnomalyDetectorConfig(BaseModel):
    """
    Configuration for the anomaly detector.
    """

    anomaly_threshold: float = Field(
        default=0.00001, description="Threshold for detecting anomalies in soundscapes."
    )
    training_timestamp: str = Field(
        default="0", description="Timestamp of the model training."
    )
    latent_dim: int = Field(
        default=128,
        description="Dimensionality of the latent space in the anomaly detection model.",
    )
    n_mels: int = Field(
        default=64,
        description="Number of mel frequency bins used in the sound analysis.",
    )
    fixed_tbins: int = Field(
        default=96, description="Fixed number of time bins for sound analysis."
    )
    training_samples: int = Field(
        default=0,
        description="Number of samples used for training the anomaly detection model.",
    )
    mean_reconstruction_error: float = Field(
        default=0.0006, description="Mean reconstruction error from the training data."
    )
    std_reconstruction_error: float = Field(
        default=0.0002,
        description="Standard deviation of the reconstruction error from the training data.",
    )
    model_version: str = Field(
        default="1.0.0", description="Version of the anomaly detection model."
    )
    mel_mean: float = Field(
        default=0.0, description="Mean value for mel spectrogram normalization."
    )
    mel_std: float = Field(
        default=1.0, description="Standard deviation for mel spectrogram normalization."
    )


class StallConfig(BaseModel):
    """
    Configuration for a stall in the stable.
    """

    stall_id: str = Field(
        default="default_stall", description="Unique identifier for the stall."
    )
    stable_id: str = Field(
        default="default_stable",
        description="Identifier for the stable this stall belongs to.",
    )
    anomaly_model_config: AnomalyDetectorConfig = Field(
        default_factory=lambda: AnomalyDetectorConfig(),
        description="Configuration for the anomaly detection model.",
    )
    soundscape_classifier_config: SoundscapeClassifierConfig = Field(
        default_factory=lambda: SoundscapeClassifierConfig(),
        description="Configuration for the soundscape classifier.",
    )

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure anomaly model config is always initialized
        if not isinstance(self.anomaly_model_config, AnomalyDetectorConfig):
            self.anomaly_model_config = AnomalyDetectorConfig(
                **self.anomaly_model_config
            )
        # Ensure soundscape classifier config is always initialized
        if not isinstance(
            self.soundscape_classifier_config, SoundscapeClassifierConfig
        ):
            self.soundscape_classifier_config = SoundscapeClassifierConfig(
                **self.soundscape_classifier_config
            )


class Label(BaseModel):
    """Represents a single label with metadata."""

    name: str
    value: str
    ls_group: str = "label"
    label_group: str = "other"
    is_alert: bool = False
    alert_confidence_threshold: Optional[float] = None
    alert_severity_score: Optional[float] = None
    is_cluster: bool = False
    min_train_carrier_fraction: Optional[float] = None  # Minimum fraction of training samples that should carry this label
    is_external_source: Optional[bool] = None  # Flag indicating if this label represents an external source
    cluster_seconds_per_severity_level: float = 0.0
    cluster_warning_min_severity: float = 1.0
    cluster_confidence_floor: Optional[float] = None
    show_in_ls: bool
    use_for_classifier_training: bool
    use_for_anomaly_training: bool
    description: str
    merge_into: Optional[str] = (
        None  # If set, this label is merged into another during training
    )
    training_importance: float = (
        1.0  # Weight for this label in weighted metrics (default 1.0)
    )
    max_samples_per_class: Optional[int] = (
        None  # Optional per-label cap that overrides global training setting
    )
    eval_only: bool = False  # If true, this label is only used for evaluation, not training

    def __post_init__(self):
        """Validate label after initialization."""
        if not self.name or not self.value:
            raise ValueError("Label name and value cannot be empty")


class MultiStallBatchResultReturn(BaseModel):
    """
    Result of a multi-stall batch prediction.
    """

    start_time: float = Field(
        default=0.0, description="Start time of the prediction segment."
    )
    end_time: float = Field(
        default=0.0, description="End time of the prediction segment."
    )
    is_anomaly: bool = Field(
        default=False,
        description="Indicates if this segment is classified as an anomaly.",
    )
    anomaly_score: float = Field(
        default=0.0, description="Anomaly score for this segment."
    )
    predicted_label: Optional[str] = Field(
        default=None, description="Predicted label for this segment, if applicable."
    )
    probability: float = Field(
        default=0.0, description="Probability of the predicted label."
    )
    label_probabilities: LabelProbabilities = Field(
        default_factory=dict,
        description="Probabilities for all labels in this segment.",
    )
    loudness: float = Field(
        default=0.0,
        description="Loudness of the audio segment, used for metamodel analysis.",
    )
    spectral_centroid: float = Field(
        default=0.0,
        description="Spectral centroid of the audio segment in Hz.",
    )
    high_freq_ratio: float = Field(
        default=0.0,
        description="Ratio of energy above 4kHz to total segment energy.",
    )


class MetaModelResult(
    BaseModel
):  # TODO: THIS IS NOT REALLY USED! (COMPARE WITH MODELS IN metamodel_rules.py)
    timestamp: datetime
    primary_alert: Optional[str]  # Main alert type if any
    confidence: float
    contributing_factors: List[str]
    label_distribution: Dict[str, float]  # Labels in current window
    loudness_metrics: Dict[
        str, Dict[str, float]
    ]  # Multi-scale metrics: immediate/stable/trend
    time_context: Dict[str, Any]  # Time of day, weather etc.
