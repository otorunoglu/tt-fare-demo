use serde::{Deserialize, Serialize};
use sqlx::{FromRow, Type};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct InferenceEvent {
    pub timestamp: chrono::DateTime<chrono::Local>,
    #[sqlx(json)]
    pub label_probs: HashMap<String, f32>,
    pub loudness: f32,
    pub source: String,
    pub replay_tag: Option<String>,
    #[sqlx(json)]
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct LoggedEvent {
    pub timestamp: chrono::DateTime<chrono::Local>,
    pub label: String,
    pub confidence: f32,
    pub loudness: f32,
    pub source: String,
    pub replay_tag: Option<String>,
    #[sqlx(json)]
    pub secondary_labels: Vec<LabelProb>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelProb {
    pub label: String,
    pub prob: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct DomainWarning {
    pub timestamp: chrono::DateTime<chrono::Local>,
    pub label: String,
    pub warning_type: WarningType,
    pub severity: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct WarningHistoryItem {
    pub id: i64,
    pub timestamp: chrono::DateTime<chrono::Local>,
    pub label: String,
    pub warning_type: WarningType,
    pub severity: f32,
    pub confidence: f32,
    pub acknowledged: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[sqlx(type_name = "TEXT")]
pub enum WarningType {
    ALERT,
    CLUSTER,
    LOUDNESS,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoudnessBin {
    pub time: chrono::DateTime<chrono::Local>,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InactivityAlertConfig {
    /// Whether the nightly inactivity alert is enabled.
    pub enabled: bool,
    /// Local time to fire the alert in "HH:MM" format (24-hour).
    pub alert_time: String,
    /// Labels to include in the activity check (e.g. ["eating", "drinking", "snoring"]).
    pub labels: Vec<String>,
    /// Alert fires when total detections across all `labels` is below this value.
    pub min_detections: i64,
    /// How many hours back from `alert_time` to check for activity.
    pub look_back_hours: u32,
}

impl Default for InactivityAlertConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            alert_time: "08:00".to_string(),
            labels: vec![
                "eating".to_string(),
                "drinking".to_string(),
                "snoring".to_string(),
            ],
            min_detections: 1,
            look_back_hours: 12,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationSettings {
    /// Whether push notifications are enabled.
    pub enabled: bool,
    /// Minimum `DomainWarning.severity` required to trigger a push notification.
    pub severity_threshold: f32,
    /// Configuration for the scheduled nightly inactivity alert.
    #[serde(default)]
    pub inactivity_alert: InactivityAlertConfig,
}

impl Default for NotificationSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            severity_threshold: 1.0,
            inactivity_alert: InactivityAlertConfig::default(),
        }
    }
}
