use clap::Parser;
use serde::Deserialize;
use shared::NotificationSettings;
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;

/// CLI Arguments structure (Parsing logic)
#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct CliArgs {
    #[arg(long, env = "STABLE_ID", default_value = "stable01")]
    pub stable_id: String,

    #[arg(long, env = "STALL_ID", default_value = "stall01")]
    pub stall_id: String,

    // Made Optional to allow overrides from options/config
    #[arg(long)]
    pub embedder_path: Option<PathBuf>,

    #[arg(long)]
    pub classifier_path: Option<PathBuf>,

    #[arg(long)]
    pub label_mapping_path: Option<PathBuf>,

    #[arg(long, env = "SMART_STABLE_MODEL_CONFIG_PATH", default_value = "../../config/smart_stable_model_config.toml")]
    pub config_path: PathBuf,

    #[arg(long, default_value_t = 16000)]
    pub sample_rate: u32,

    #[arg(long, default_value_t = 2.0)]
    pub segment_duration: f32,

    #[arg(long, default_value_t = 0.5)]
    pub segment_overlap: f32,

    #[arg(long, default_value_t = 64)]
    pub max_queue_size: usize,

    #[arg(long)]
    pub simulate_file: Option<PathBuf>,

    #[arg(long, env = "AUDIO_DEVICE")]
    pub audio_device: Option<String>,

    #[arg(long, default_value_t = 0.5)]
    pub log_threshold: f32,

    #[arg(long, default_value_t = false)]
    pub inference_logging: bool,

    // Made Optional to allow overrides
    #[arg(long)]
    pub database_path: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

/// Final Runtime Configuration (Application State)
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub stable_id: String,
    pub stall_id: String,
    pub embedder_path: PathBuf,
    pub classifier_path: PathBuf,
    pub label_mapping_path: PathBuf,
    pub config_path: PathBuf,
    pub sample_rate: u32,
    pub segment_duration: f32,
    pub segment_overlap: f32,
    pub max_queue_size: usize,
    pub simulate_file: Option<PathBuf>,
    pub audio_device: Option<String>,
    pub log_threshold: f32,
    pub database_path: PathBuf,
    pub inference_logging: bool,
    pub event_source: String,
    pub replay_tag: Option<String>,
    pub stop_on_stream_end: bool,

    pub command: Option<Commands>,
    // Store edge config for verification tests accessibility
    pub edge_config: EdgeMonitorConfig,
}

#[derive(clap::Subcommand, Debug, Clone)]
pub enum Commands {
    /// Run inference (default)
    Run,

    /// Verify logic against a Python-generated artifact
    VerifyArtifact {
        #[arg(long)]
        file: PathBuf,
    },
}

#[derive(Debug, Deserialize, Clone)]
pub struct SmartStableModelConfig {
    pub labels: Vec<LabelInfo>,
    pub model_parameters: HashMap<String, toml::Value>,
    pub loudness_parameters: HashMap<String, toml::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub group_seconds_per_severity_level: HashMap<String, f32>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LabelInfo {
    #[allow(dead_code)]
    pub name: String,
    pub value: String,
    #[allow(dead_code)]
    pub label_group: String,
    #[serde(default)]
    pub is_alert: bool,
    #[serde(default)]
    pub alert_confidence_threshold: Option<f32>,
    #[serde(default)]
    pub alert_severity_score: Option<f32>,
    #[serde(default)]
    pub is_cluster: bool,
    #[serde(default)]
    pub cluster_seconds_per_severity_level: f32,
    #[serde(default = "default_cluster_warning_min_severity")]
    pub cluster_warning_min_severity: f32,
    #[serde(default)]
    pub cluster_confidence_floor: Option<f32>,
    #[serde(default)]
    pub auto_thresholding: bool,
    #[serde(default)]
    pub threshold_mode: ThresholdMode,
    #[serde(default = "default_threshold_percentile")]
    pub threshold_percentile: f32,
    #[serde(default)]
    pub manual_threshold_db: Option<f32>,
    #[serde(default = "default_safety_margin_db")]
    pub safety_margin_db: f32,
    #[serde(default = "default_min_samples_for_calibration")]
    pub min_samples_for_calibration: usize,
    #[serde(default)]
    pub calibration_mode: CalibrationMode,
    #[serde(default = "default_rolling_window")]
    pub rolling_window: String,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationMode {
    #[default]
    Rolling,
    FixedOnce,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ThresholdMode {
    #[default]
    Auto,
    Manual,
}

fn default_cluster_warning_min_severity() -> f32 {
    1.0
}

fn default_threshold_percentile() -> f32 {
    0.80
}

fn default_safety_margin_db() -> f32 {
    3.0
}

fn default_min_samples_for_calibration() -> usize {
    200
}

fn default_rolling_window() -> String {
    "14d".to_string()
}

impl RuntimeConfig {
    pub fn initialize() -> Self {
        let args = CliArgs::parse();
        Self::from_args(args)
    }

    pub fn from_args(args: CliArgs) -> Self {
        // Load the optional TOML config
        let edge_config = RuntimeConfig::load_edge_monitor_config();

        let explicit_embedder_path = args
            .embedder_path
            .clone()
            .or_else(|| edge_config.embedder_path.as_ref().map(PathBuf::from));

        let explicit_classifier_path = args
            .classifier_path
            .clone()
            .or_else(|| edge_config.classifier_path.as_ref().map(PathBuf::from));

        let explicit_label_mapping_path = args
            .label_mapping_path
            .clone()
            .or_else(|| edge_config.label_mapping_path.as_ref().map(PathBuf::from));

        let env_embedder_path = std::env::var("EMBEDDER_PATH").ok().map(PathBuf::from);
        let env_classifier_path = std::env::var("CLASSIFIER_PATH").ok().map(PathBuf::from);
        let env_label_mapping_path = std::env::var("LABEL_MAPPING_PATH").ok().map(PathBuf::from);

        let auto_discover_models = std::env::var("AUTO_DISCOVER_MODEL_PATHS")
            .map(|value| {
                let normalized = value.trim().to_ascii_lowercase();
                !matches!(normalized.as_str(), "0" | "false" | "no" | "off")
            })
            .unwrap_or(true);

        let current_model_path = std::env::var("Current_Model_Path")
            .ok()
            .or_else(|| std::env::var("CURRENT_MODEL_PATH").ok())
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from);

        let discovered_models = if !auto_discover_models
            || (explicit_embedder_path.is_some()
                && explicit_classifier_path.is_some()
                && explicit_label_mapping_path.is_some())
        {
            None
        } else if let Some(current_path) = current_model_path.as_deref() {
            discover_model_triplet_from_current_dir(current_path)
        } else {
            discover_latest_model_triplet(Path::new("data/models"))
        };

        // MERGE LOGIC:
        // 1) CLI/TOML explicit model paths
        // 2) Auto-discovered latest complete triplet (if enabled)
        // 3) Legacy model env vars
        // 4) Hardcoded fallback defaults

        let embedder_path = explicit_embedder_path
            .or_else(|| discovered_models.as_ref().map(|models| models.embedder_path.clone()))
            .or(env_embedder_path)
            .unwrap_or_else(|| PathBuf::from("data/models/full_pipeline_embedder.onnx"));

        let classifier_path = explicit_classifier_path
            .or_else(|| {
                discovered_models
                    .as_ref()
                    .map(|models| models.classifier_path.clone())
            })
            .or(env_classifier_path)
            .unwrap_or_else(|| PathBuf::from("data/models/full_pipeline_classifier.onnx"));

        let label_mapping_path = explicit_label_mapping_path
            .or_else(|| {
                discovered_models
                    .as_ref()
                    .map(|models| models.label_mapping_path.clone())
            })
            .or(env_label_mapping_path)
            .unwrap_or_else(|| PathBuf::from("data/models/label_mapping.json"));

        let database_path = args
            .database_path
            .or_else(|| edge_config.database_path.as_ref().map(PathBuf::from))
            .or_else(|| {
                std::env::var("DATABASE_URL").ok().map(|s| {
                    // Remove sqlite:// prefix for the local file path
                    if s.starts_with("sqlite://") {
                        PathBuf::from(&s[9..])
                    } else {
                        PathBuf::from(s)
                    }
                })
            })
            .unwrap_or_else(|| PathBuf::from("soundscape.db"));

        RuntimeConfig {
            stable_id: args.stable_id,
            stall_id: args.stall_id,
            embedder_path,
            classifier_path,
            label_mapping_path,
            config_path: args.config_path,
            sample_rate: args.sample_rate,
            segment_duration: args.segment_duration,
            segment_overlap: args.segment_overlap,
            max_queue_size: args.max_queue_size,
            simulate_file: args.simulate_file,
            audio_device: args.audio_device,
            log_threshold: args.log_threshold,
            database_path,
            inference_logging: if args.inference_logging {
                true
            } else {
                edge_config.inference_logging.unwrap_or(false)
            },
            event_source: "live".to_string(),
            replay_tag: None,
            stop_on_stream_end: false,

            command: args.command,
            edge_config,
        }
    }

    pub fn load_metamodel_config(&self) -> anyhow::Result<SmartStableModelConfig> {
        use anyhow::Context;

        let content = std::fs::read_to_string(&self.config_path).with_context(|| {
            format!("Failed to read smart stable model config at {:?}", self.config_path)
        })?;
        let config: SmartStableModelConfig = toml::from_str(&content).with_context(|| {
            format!(
                "Failed to parse smart stable model config TOML at {:?}",
                self.config_path
            )
        })?;
        Ok(config)
    }

    // Kept this for backward compatibility if needed, but it should be accessed via self.edge_config now
    // Actually method was static capable before? No, it was &self.
    // Let's make it an associated function or private helper.
    fn load_edge_monitor_config() -> EdgeMonitorConfig {
        let path_str = std::env::var("EDGE_MONITOR_CONFIG_PATH")
            .unwrap_or_else(|_| "config/edge_monitor_config.toml".to_string());
        let path = PathBuf::from(&path_str);

        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(content) => match toml::from_str(&content) {
                    Ok(cfg) => {
                        log::info!("Loaded edge monitor config from {:?}", path);
                        return cfg;
                    }
                    Err(e) => log::error!("Failed to parse {:?}: {}", path, e),
                },
                Err(e) => log::error!("Failed to read {:?}: {}", path, e),
            }
        } else {
            log::info!(
                "Edge monitor config not found at {:?}, using defaults",
                path
            );
        }
        EdgeMonitorConfig::default()
    }

    // Helper to get db path logic if we need it, but now it's resolved in struct
    pub fn get_database_path(&self) -> PathBuf {
        self.database_path.clone()
    }
}

#[derive(Debug, Clone)]
struct DiscoveredModelTriplet {
    version: u32,
    stem: String,
    embedder_path: PathBuf,
    classifier_path: PathBuf,
    label_mapping_path: PathBuf,
}

#[derive(Debug, Default)]
struct PartialModelTriplet {
    embedder_path: Option<PathBuf>,
    classifier_path: Option<PathBuf>,
    label_mapping_path: Option<PathBuf>,
}

fn discover_latest_model_triplet(models_dir: &Path) -> Option<DiscoveredModelTriplet> {
    let entries = match std::fs::read_dir(models_dir) {
        Ok(entries) => entries,
        Err(e) => {
            log::warn!(
                "Could not read models directory {:?} for auto-discovery: {}",
                models_dir,
                e
            );
            return None;
        }
    };

    let mut versions: HashMap<(u32, String), PartialModelTriplet> = HashMap::new();

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let Some(file_name) = path.file_name().and_then(|f| f.to_str()) else {
            continue;
        };

        if let Some((version, stem)) = parse_versioned_stem(file_name, "_embedder.onnx") {
            let slot = versions.entry((version, stem)).or_default();
            slot.embedder_path = Some(path.clone());
            continue;
        }

        if let Some((version, stem)) = parse_versioned_stem(file_name, "_classifier.onnx") {
            let slot = versions.entry((version, stem)).or_default();
            slot.classifier_path = Some(path.clone());
            continue;
        }

        if let Some((version, stem)) = parse_versioned_stem(file_name, "_label_mapping.json") {
            let slot = versions.entry((version, stem)).or_default();
            slot.label_mapping_path = Some(path.clone());
        }
    }

    let mut discovered: Vec<DiscoveredModelTriplet> = versions
        .into_iter()
        .filter_map(|((version, stem), partial)| {
            Some(DiscoveredModelTriplet {
                version,
                stem,
                embedder_path: partial.embedder_path?,
                classifier_path: partial.classifier_path?,
                label_mapping_path: partial.label_mapping_path?,
            })
        })
        .collect();

    discovered.sort_by(|a, b| {
        a.version
            .cmp(&b.version)
            .then_with(|| a.stem.cmp(&b.stem))
    });

    let latest = discovered.pop();
    if let Some(models) = &latest {
        log::info!(
            "Auto-discovered latest model triplet '{}' (v{}): embedder={:?}, classifier={:?}, labels={:?}",
            models.stem,
            models.version,
            models.embedder_path,
            models.classifier_path,
            models.label_mapping_path
        );
    }

    latest
}

fn discover_model_triplet_from_current_dir(models_dir: &Path) -> Option<DiscoveredModelTriplet> {
    let entries = match std::fs::read_dir(models_dir) {
        Ok(entries) => entries,
        Err(e) => {
            log::warn!(
                "Could not read current model directory {:?}: {}",
                models_dir,
                e
            );
            return None;
        }
    };

    let mut embedder: Vec<PathBuf> = Vec::new();
    let mut classifier: Vec<PathBuf> = Vec::new();
    let mut label_mapping: Vec<PathBuf> = Vec::new();

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let Some(file_name) = path.file_name().and_then(|f| f.to_str()) else {
            continue;
        };

        if file_name.ends_with("_embedder.onnx") {
            embedder.push(path.clone());
            continue;
        }

        if file_name.ends_with("_classifier.onnx") {
            classifier.push(path.clone());
            continue;
        }

        if file_name.ends_with("_label_mapping.json") {
            label_mapping.push(path);
        }
    }

    embedder.sort();
    classifier.sort();
    label_mapping.sort();

    if embedder.len() != 1 || classifier.len() != 1 || label_mapping.len() != 1 {
        log::warn!(
            "Current model directory {:?} must contain exactly one *_embedder.onnx, one *_classifier.onnx, and one *_label_mapping.json (found: embedder={}, classifier={}, label_mapping={})",
            models_dir,
            embedder.len(),
            classifier.len(),
            label_mapping.len()
        );
        return None;
    }

    let embedder_path = embedder.pop()?;
    let classifier_path = classifier.pop()?;
    let label_mapping_path = label_mapping.pop()?;

    let stem = embedder_path
        .file_name()
        .and_then(|f| f.to_str())
        .and_then(|name| name.strip_suffix("_embedder.onnx"))
        .unwrap_or("current")
        .to_string();

    log::info!(
        "Auto-discovered current model triplet from {:?}: embedder={:?}, classifier={:?}, labels={:?}",
        models_dir,
        embedder_path,
        classifier_path,
        label_mapping_path
    );

    Some(DiscoveredModelTriplet {
        version: 0,
        stem,
        embedder_path,
        classifier_path,
        label_mapping_path,
    })
}

fn parse_versioned_stem(file_name: &str, suffix: &str) -> Option<(u32, String)> {
    let stem = file_name.strip_suffix(suffix)?;
    let version_pos = stem.rfind("_v")?;
    let version_str = stem.get(version_pos + 2..)?;

    if version_str.is_empty() || !version_str.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }

    let version = version_str.parse::<u32>().ok()?;
    Some((version, stem.to_string()))
}

#[derive(Debug, Deserialize, Clone)]
pub struct EdgeMonitorConfig {
    #[serde(default)]
    pub audio: AudioConfig,
    pub database_path: Option<String>,
    pub inference_logging: Option<bool>,

    // Model paths (Optional, overrides CLI defaults if present)
    pub embedder_path: Option<String>,
    pub classifier_path: Option<String>,
    pub label_mapping_path: Option<String>,

    #[serde(default)]
    pub notification: NotificationSettings,

    #[serde(default)]
    pub recording: RecordingConfig,
}

impl Default for EdgeMonitorConfig {
    fn default() -> Self {
        Self {
            audio: AudioConfig::default(),
            database_path: None,
            inference_logging: None,
            embedder_path: None,
            classifier_path: None,
            label_mapping_path: None,
            notification: NotificationSettings::default(),
            recording: RecordingConfig::default(),
        }
    }
}

/// Configuration for saving a rolling audio window to disk when a warning fires.
///
/// When `enabled`, the runner keeps the most recent `pre_seconds` of captured audio
/// in memory. As soon as a `DomainWarning` with `severity >= severity_threshold`
/// occurs, that pre-roll plus the following `post_seconds` of audio is written as a
/// 16-bit PCM WAV file into `output_dir`.
#[derive(Debug, Deserialize, Clone)]
pub struct RecordingConfig {
    /// Master switch for the alert-clip feature.
    #[serde(default)]
    pub enabled: bool,
    /// Directory where clips are written (created if missing). Relative to the working dir.
    #[serde(default = "default_clip_output_dir")]
    pub output_dir: String,
    /// Seconds of audio kept before the triggering warning.
    #[serde(default = "default_clip_pre_seconds")]
    pub pre_seconds: f32,
    /// Seconds of audio captured after the triggering warning.
    #[serde(default = "default_clip_post_seconds")]
    pub post_seconds: f32,
    /// Minimum `DomainWarning.severity` required to save a clip.
    #[serde(default = "default_clip_severity_threshold")]
    pub severity_threshold: f32,
    /// Maximum number of clips to keep. Oldest clips beyond this are deleted. 0 = unlimited.
    #[serde(default = "default_clip_max_clips")]
    pub max_clips: usize,
    /// Delete clips older than this many days. 0 = never delete by age.
    #[serde(default = "default_clip_max_age_days")]
    pub max_age_days: f32,
}

impl Default for RecordingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            output_dir: default_clip_output_dir(),
            pre_seconds: default_clip_pre_seconds(),
            post_seconds: default_clip_post_seconds(),
            severity_threshold: default_clip_severity_threshold(),
            max_clips: default_clip_max_clips(),
            max_age_days: default_clip_max_age_days(),
        }
    }
}

fn default_clip_output_dir() -> String {
    "data/alert_clips".to_string()
}

fn default_clip_pre_seconds() -> f32 {
    30.0
}

fn default_clip_post_seconds() -> f32 {
    10.0
}

fn default_clip_severity_threshold() -> f32 {
    1.0
}

fn default_clip_max_clips() -> usize {
    200
}

fn default_clip_max_age_days() -> f32 {
    0.0
}

#[derive(Debug, Deserialize, Clone)]
pub struct AudioConfig {
    #[serde(default = "default_resampler_type")]
    pub resampler_type: String,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            resampler_type: default_resampler_type(),
        }
    }
}

fn default_resampler_type() -> String {
    "linear".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_config_database_path_default() {
        let edge_config = EdgeMonitorConfig::default();
        println!("database_path value: {:?}", edge_config.database_path);
        assert_eq!(edge_config.database_path, None);
    }
}
