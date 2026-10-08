use crate::config::SmartStableModelConfig;
use chrono::{DateTime, Local, Timelike};
use shared::{DomainWarning, InferenceEvent, WarningType};
use std::collections::{HashMap, HashSet, VecDeque};

pub struct MetaModelDecider {
    config: SmartStableModelConfig,

    // Cluster state
    last_cluster_level: HashMap<String, i32>,
    last_cluster_time: HashMap<String, DateTime<Local>>,

    // Group severity state
    last_group_level: HashMap<String, i32>,

    // Loudness state
    last_loudness_level: i32,
    loud_base_n: u64,
    loud_base_mean: f64,
    loud_base_m2: f64,
    first_event_time: Option<DateTime<Local>>,

    // Night rollover
    current_night_start: Option<DateTime<Local>>,

    // History for windowing
    history: VecDeque<HistoryItem>,
}

struct HistoryItem {
    timestamp: DateTime<Local>,
    label: String,
    confidence: f32,
    loudness: f32,
}

impl MetaModelDecider {
    pub fn new(config: SmartStableModelConfig) -> Self {
        Self {
            config,
            last_cluster_level: HashMap::new(),
            last_cluster_time: HashMap::new(),
            last_group_level: HashMap::new(),
            last_loudness_level: -1,
            loud_base_n: 0,
            loud_base_mean: 0.0,
            loud_base_m2: 0.0,
            first_event_time: None,
            current_night_start: None,
            history: VecDeque::new(),
        }
    }

    pub fn decide(&mut self, event: &InferenceEvent) -> Option<DomainWarning> {
        let now = event.timestamp;

        // Track first event time for loudness warmup
        if self.first_event_time.is_none() {
            self.first_event_time = Some(now);
        }

        // 1. Night rollover: reset all severity levels when a new night starts
        self.rollover_if_needed(now);

        // 2. Identify dominant label and apply dominant_label_floor
        let (mut label, mut confidence) = event
            .label_probs
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, v)| (k.clone(), *v))
            .unwrap_or(("unknown".to_string(), 0.0));

        let dominant_label_floor = self.get_param_as_f32("dominant_label_floor", 0.95);
        let background_label = self
            .config
            .model_parameters
            .get("background_label")
            .and_then(|v| v.as_str())
            .unwrap_or("uncertain")
            .to_string();

        if confidence < dominant_label_floor {
            label = background_label;
            confidence = 1.0;
        }

        // 3. Add to history & prune
        let burst_window = self.get_param_as_i64("burst_window_seconds", 120);
        self.history.push_back(HistoryItem {
            timestamp: now,
            label: label.clone(),
            confidence,
            loudness: event.loudness,
        });

        while let Some(front) = self.history.front() {
            if now.signed_duration_since(front.timestamp).num_seconds() > burst_window {
                self.history.pop_front();
            } else {
                break;
            }
        }

        // 4. Check Immediate Alerts
        if let Some(info) = self.config.labels.iter().find(|l| l.value == label) {
            let alert_threshold = self.get_alert_confidence_threshold(info);
            let alert_severity_score = self.get_alert_severity_score(info);

            if info.is_alert && confidence >= alert_threshold {
                return Some(DomainWarning {
                    timestamp: now,
                    warning_type: WarningType::ALERT,
                    label: label.clone(),
                    confidence,
                    severity: alert_severity_score,
                });
            }
        }

        // 5. Cluster level reset: decay cluster levels after inactivity
        let cluster_reset_seconds = self.get_param_as_i64("cluster_reset_seconds", 600);
        let groups_to_reset: Vec<String> = self
            .last_cluster_time
            .iter()
            .filter(|(_, last_time)| {
                now.signed_duration_since(**last_time).num_seconds() > cluster_reset_seconds
            })
            .map(|(key, _)| key.clone())
            .collect();

        for key in groups_to_reset {
            log::info!(
                "Cluster level reset for '{}' after {}s of inactivity",
                key,
                cluster_reset_seconds
            );
            self.last_cluster_level.remove(&key);
            self.last_cluster_time.remove(&key);
        }

        let severity_step = self.get_param_as_f32("severity_step", 1.0);
        let group_cluster_warning_min_severity =
            self.get_param_as_f32("group_cluster_warning_min_severity", 1.0);
        // 6. Per-label Cluster Logic
        if let Some(info) = self.config.labels.iter().find(|l| l.value == label) {
            if info.is_cluster {
                let floor = info.cluster_confidence_floor.unwrap_or(0.85);
                if confidence >= floor {
                    // Count covered seconds (deduplicated by second)
                    let covered_seconds = self.count_label_coverage(&label, floor);
                    let seconds_per_level = info.cluster_seconds_per_severity_level;
                    let severity_raw = if seconds_per_level > 0.0 {
                        covered_seconds / seconds_per_level
                    } else {
                        0.0
                    };
                    let level = (severity_raw / severity_step).floor() as i32;
                    let last_level =
                        *self.last_cluster_level.get(&label).unwrap_or(&-1);

                    if severity_raw >= info.cluster_warning_min_severity && level > last_level {
                        self.last_cluster_level.insert(label.clone(), level);
                        self.last_cluster_time.insert(label.clone(), now);
                        return Some(DomainWarning {
                            timestamp: now,
                            warning_type: WarningType::CLUSTER,
                            label: label.clone(),
                            confidence,
                            severity: severity_raw,
                        });
                    }
                }
            }
        }

        // 7. Group severity: union of coverage across labels in each group
        if let Some(warning) = self.check_group_severity(
            now,
            severity_step,
            group_cluster_warning_min_severity,
            &label,
            confidence,
        )
        {
            return Some(warning);
        }

        // 8. Adaptive Loudness
        if let Some(warning) = self.check_adaptive_loudness(event, now, severity_step) {
            return Some(warning);
        }

        None
    }

    /// Count covered seconds for a specific label in the history window,
    /// deduplicating by timestamp second to match Python behavior.
    fn count_label_coverage(&self, label: &str, confidence_floor: f32) -> f32 {
        let mut seen_seconds: HashSet<i64> = HashSet::new();
        for h in &self.history {
            if h.label == label && h.confidence >= confidence_floor {
                seen_seconds.insert(h.timestamp.timestamp());
            }
        }
        let snippet_duration = self.get_param_as_f32("default_snippet_duration_seconds", 1.0);
        seen_seconds.len() as f32 * snippet_duration
    }

    /// Check group severity: union of per-second coverage across all labels in each group,
    /// scaled by group_seconds_per_severity_level from config.
    fn check_group_severity(
        &mut self,
        now: DateTime<Local>,
        severity_step: f32,
        group_cluster_warning_min_severity: f32,
        _current_label: &str,
        _current_confidence: f32,
    ) -> Option<DomainWarning> {
        if self.config.group_seconds_per_severity_level.is_empty() {
            return None;
        }

        let snippet_duration = self.get_param_as_f32("default_snippet_duration_seconds", 1.0);

        // Build per-label second sets from history
        let mut per_label_secs: HashMap<String, HashSet<i64>> = HashMap::new();
        for h in &self.history {
            // For cluster labels, enforce confidence floor
            let floor = self
                .config
                .labels
                .iter()
                .find(|l| l.value == h.label)
                .and_then(|l| {
                    if l.is_cluster {
                        Some(l.cluster_confidence_floor.unwrap_or(0.85))
                    } else {
                        None
                    }
                });

            if let Some(f) = floor {
                if h.confidence < f {
                    continue;
                }
            }

            per_label_secs
                .entry(h.label.clone())
                .or_default()
                .insert(h.timestamp.timestamp());
        }

        // Build group -> label members mapping from config
        let mut group_members: HashMap<String, Vec<String>> = HashMap::new();
        for label_info in &self.config.labels {
            if !label_info.label_group.is_empty() {
                group_members
                    .entry(label_info.label_group.clone())
                    .or_default()
                    .push(label_info.value.clone());
            }
        }

        // Check each group
        for (gname, &seconds_per_level) in &self.config.group_seconds_per_severity_level {
            if seconds_per_level <= 0.0 {
                continue;
            }

            let members = match group_members.get(gname) {
                Some(m) => m,
                None => continue,
            };

            // Union of covered seconds across all group members
            let mut union_secs: HashSet<i64> = HashSet::new();
            for member in members {
                if let Some(secs) = per_label_secs.get(member) {
                    union_secs.extend(secs);
                }
            }

            if union_secs.is_empty() {
                continue;
            }

            let covered = union_secs.len() as f32 * snippet_duration;
            let severity_raw = covered / seconds_per_level;
            let level = (severity_raw / severity_step).floor() as i32;
            let last_level = *self.last_group_level.get(gname).unwrap_or(&-1);

            if severity_raw >= group_cluster_warning_min_severity && level > last_level {
                self.last_group_level.insert(gname.clone(), level);
                return Some(DomainWarning {
                    timestamp: now,
                    warning_type: WarningType::CLUSTER,
                    label: format!("group:{}", gname),
                    confidence: 1.0,
                    severity: severity_raw,
                });
            }
        }

        None
    }

    /// Adaptive loudness detection with warmup guard.
    fn check_adaptive_loudness(
        &mut self,
        event: &InferenceEvent,
        now: DateTime<Local>,
        severity_step: f32,
    ) -> Option<DomainWarning> {
        if !self.get_param_bool("enable_adaptive_loudness", true) {
            return None;
        }

        let val = event.loudness as f64;

        // Welford update
        self.loud_base_n += 1;
        let delta = val - self.loud_base_mean;
        self.loud_base_mean += delta / self.loud_base_n as f64;
        let delta2 = val - self.loud_base_mean;
        self.loud_base_m2 += delta * delta2;

        // Warmup guards (matching Python):
        // 1. Need enough baseline samples
        if self.loud_base_n < 30 {
            return None;
        }

        // 2. Need enough elapsed time since first event
        let warmup_seconds =
            self.get_loudness_param_f32("loudness_warmup_seconds", 900.0) as i64;
        if let Some(first_time) = self.first_event_time {
            let elapsed = now.signed_duration_since(first_time).num_seconds();
            if elapsed < warmup_seconds {
                return None;
            }
        }

        // 3. Need minimum window samples
        let min_window_seconds =
            self.get_loudness_param_f32("loudness_min_window_seconds", 10.0) as usize;
        let window_count = self.history.len();
        if window_count < min_window_seconds {
            return None;
        }

        let var = self.loud_base_m2 / (self.loud_base_n - 1) as f64;
        let std_dev = var.sqrt();
        let min_std = self.get_loudness_param_f32("loudness_min_std", 1.0) as f64;
        let eff_std = std_dev.max(min_std);

        let window_sum: f32 = self.history.iter().map(|h| h.loudness).sum();
        let window_mean = if window_count > 0 {
            window_sum / window_count as f32
        } else {
            0.0
        };

        let multiplier = self.get_loudness_param_f32("loudness_sigma_multiplier", 2.0) as f64;
        let z = (window_mean as f64 - self.loud_base_mean) / eff_std;

        if z > multiplier {
            let severity_scale = self.get_loudness_param_f32("loudness_severity_scale", 1.0);
            let severity_raw = (z / multiplier) as f32 * severity_scale;
            let level = (severity_raw / severity_step).floor() as i32;

            if severity_raw >= 1.0 && level > self.last_loudness_level {
                self.last_loudness_level = level;
                return Some(DomainWarning {
                    timestamp: now,
                    warning_type: WarningType::LOUDNESS,
                    label: "loudness_adaptive".to_string(),
                    confidence: 1.0,
                    severity: severity_raw,
                });
            }
        }

        None
    }

    /// Night rollover: reset severity levels when transitioning to a new night.
    /// Night is defined by night_hours (start_h, end_h) in config.
    /// Default: (20, 7) = 20:00 to 07:00.
    fn rollover_if_needed(&mut self, ts: DateTime<Local>) {
        let night_start = self.compute_night_start(ts);

        match self.current_night_start {
            None => {
                self.current_night_start = Some(night_start);
            }
            Some(current) => {
                if night_start > current {
                    log::info!(
                        "Night rollover: resetting severity levels (new night starts at {})",
                        night_start
                    );
                    self.last_cluster_level.clear();
                    self.last_cluster_time.clear();
                    self.last_group_level.clear();
                    self.last_loudness_level = -1;
                    self.current_night_start = Some(night_start);
                }
            }
        }
    }

    /// Compute the start of the night that `ts` belongs to.
    /// For wrapping nights (e.g., start=20, end=7):
    ///   - Hours >= start_h belong to the night starting that calendar day at start_h.
    ///   - Hours < end_h belong to the night that started the previous day at start_h.
    fn compute_night_start(&self, ts: DateTime<Local>) -> DateTime<Local> {
        let (start_h, end_h) = self.get_night_hours();
        let hour = ts.hour();

        if start_h > end_h {
            // Wrapping night (e.g., 20 -> 7)
            if hour >= start_h {
                // Same day at start_h
                ts.date_naive()
                    .and_hms_opt(start_h, 0, 0)
                    .map(|dt| dt.and_local_timezone(Local).unwrap())
                    .unwrap_or(ts)
            } else {
                // Early morning -> previous day at start_h
                (ts.date_naive() - chrono::Duration::days(1))
                    .and_hms_opt(start_h, 0, 0)
                    .map(|dt| dt.and_local_timezone(Local).unwrap())
                    .unwrap_or(ts)
            }
        } else {
            // Non-wrapping (e.g., 8 -> 18)
            if hour >= start_h {
                ts.date_naive()
                    .and_hms_opt(start_h, 0, 0)
                    .map(|dt| dt.and_local_timezone(Local).unwrap())
                    .unwrap_or(ts)
            } else {
                (ts.date_naive() - chrono::Duration::days(1))
                    .and_hms_opt(start_h, 0, 0)
                    .map(|dt| dt.and_local_timezone(Local).unwrap())
                    .unwrap_or(ts)
            }
        }
    }

    fn get_night_hours(&self) -> (u32, u32) {
        let start = self
            .config
            .model_parameters
            .get("night_hour_start")
            .and_then(|v| v.as_integer())
            .unwrap_or(20) as u32;
        let end = self
            .config
            .model_parameters
            .get("night_hour_end")
            .and_then(|v| v.as_integer())
            .unwrap_or(7) as u32;
        (start, end)
    }

    fn get_param_as_i64(&self, key: &str, default: i64) -> i64 {
        self.config
            .model_parameters
            .get(key)
            .and_then(|v| v.as_integer())
            .unwrap_or(default)
    }

    fn get_param_as_f32(&self, key: &str, default: f32) -> f32 {
        self.config
            .model_parameters
            .get(key)
            .and_then(|v| v.as_float())
            .map(|f| f as f32)
            .unwrap_or(default)
    }

    fn get_param_bool(&self, key: &str, default: bool) -> bool {
        self.config
            .model_parameters
            .get(key)
            .and_then(|v| v.as_bool())
            .unwrap_or(default)
    }

    fn get_loudness_param_f32(&self, key: &str, default: f32) -> f32 {
        self.config
            .loudness_parameters
            .get(key)
            .and_then(|v| v.as_float())
            .map(|f| f as f32)
            .unwrap_or(default)
    }

    fn get_alert_confidence_threshold(&self, info: &crate::config::LabelInfo) -> f32 {
        info.alert_confidence_threshold
            .unwrap_or(self.get_param_as_f32("alert_labels_confidence_floor", 0.6))
    }

    fn get_alert_severity_score(&self, info: &crate::config::LabelInfo) -> f32 {
        info.alert_severity_score.unwrap_or(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LabelInfo;
    use shared::WarningType;

    fn mock_config() -> SmartStableModelConfig {
        SmartStableModelConfig {
            labels: vec![
                LabelInfo {
                    name: "Neigh".to_string(),
                    value: "neigh".to_string(),
                    label_group: "distress".to_string(),
                    is_alert: true,
                    alert_confidence_threshold: Some(0.8),
                    alert_severity_score: Some(0.8),
                    is_cluster: true,
                    cluster_seconds_per_severity_level: 1.0,
                    cluster_warning_min_severity: 1.0,
                    cluster_confidence_floor: Some(0.6),
                    auto_thresholding: false,
                    threshold_mode: crate::config::ThresholdMode::Auto,
                    threshold_percentile: 0.80,
                    manual_threshold_db: None,
                    safety_margin_db: 3.0,
                    min_samples_for_calibration: 200,
                    calibration_mode: crate::config::CalibrationMode::Rolling,
                    rolling_window: "14d".to_string(),
                },
                LabelInfo {
                    name: "Scream".to_string(),
                    value: "scream".to_string(),
                    label_group: "distress".to_string(),
                    is_alert: true,
                    alert_confidence_threshold: Some(0.9),
                    alert_severity_score: Some(0.9),
                    is_cluster: true,
                    cluster_seconds_per_severity_level: 1.0,
                    cluster_warning_min_severity: 1.0,
                    cluster_confidence_floor: Some(0.6),
                    auto_thresholding: false,
                    threshold_mode: crate::config::ThresholdMode::Auto,
                    threshold_percentile: 0.80,
                    manual_threshold_db: None,
                    safety_margin_db: 3.0,
                    min_samples_for_calibration: 200,
                    calibration_mode: crate::config::CalibrationMode::Rolling,
                    rolling_window: "14d".to_string(),
                },
            ],
            model_parameters: HashMap::new(),
            loudness_parameters: HashMap::new(),
            group_seconds_per_severity_level: HashMap::new(),
        }
    }

    #[test]
    fn test_immediate_alert() {
        let mut config = mock_config();
        // Set dominant_label_floor low so it doesn't interfere
        config.model_parameters.insert(
            "dominant_label_floor".to_string(),
            toml::Value::Float(0.5),
        );
        let mut decider = MetaModelDecider::new(config);
        let mut label_probs = HashMap::new();
        label_probs.insert("neigh".to_string(), 0.9);

        let event = InferenceEvent {
            timestamp: Local::now(),
            label_probs,
            loudness: 0.5,
            source: "live".to_string(),
            replay_tag: None,
            embedding: None,
        };

        let warning = decider.decide(&event);
        assert!(warning.is_some());
        let w = warning.unwrap();
        assert_eq!(w.warning_type, WarningType::ALERT);
        assert_eq!(w.label, "neigh");
    }

    #[test]
    fn test_dominant_label_floor() {
        let mut config = mock_config();
        config.model_parameters.insert(
            "dominant_label_floor".to_string(),
            toml::Value::Float(0.95),
        );
        let mut decider = MetaModelDecider::new(config);

        // Low confidence prediction -> should be overridden to "uncertain"
        let mut label_probs = HashMap::new();
        label_probs.insert("neigh".to_string(), 0.6);

        let event = InferenceEvent {
            timestamp: Local::now(),
            label_probs,
            loudness: -40.0,
            source: "live".to_string(),
            replay_tag: None,
            embedding: None,
        };

        // Should NOT trigger an alert because label was overridden to "uncertain"
        let warning = decider.decide(&event);
        assert!(warning.is_none());

        // Verify history recorded "uncertain" not "neigh"
        assert_eq!(decider.history.back().unwrap().label, "uncertain");
    }

    #[test]
    fn test_group_severity() {
        let mut config = mock_config();
        // Set up group weights
        config
            .group_seconds_per_severity_level
            .insert("distress".to_string(), 1.0);
        config.model_parameters.insert(
            "dominant_label_floor".to_string(),
            toml::Value::Float(0.5),
        );
        // Disable alerts so we can test cluster/group logic
        config.labels[0].is_alert = false;
        config.labels[1].is_alert = false;

        let mut decider = MetaModelDecider::new(config);

        // Add one neigh event (1s coverage)
        let mut probs1 = HashMap::new();
        probs1.insert("neigh".to_string(), 0.9);
        let event1 = InferenceEvent {
            timestamp: Local::now(),
            label_probs: probs1,
            loudness: -30.0,
            source: "live".to_string(),
            replay_tag: None,
            embedding: None,
        };
        let w1 = decider.decide(&event1);
        // 1s / 2s threshold * 1.0 severity = 0.5 -> no cluster yet
        // But per-label cluster check might not fire. Group: 1s/2s * 1.0 = 0.5 -> no warning
        assert!(w1.is_none());

        // Add one scream event 1s later (now group has 2s coverage total)
        let mut probs2 = HashMap::new();
        probs2.insert("scream".to_string(), 0.9);
        let event2 = InferenceEvent {
            timestamp: Local::now() + chrono::Duration::seconds(1),
            label_probs: probs2,
            loudness: -30.0,
            source: "live".to_string(),
            replay_tag: None,
            embedding: None,
        };
        let w2 = decider.decide(&event2);
        // Group "distress" now has 2 seconds of coverage (neigh + scream)
        // severity = (2/2) * 1.0 = 1.0 -> should trigger group warning
        assert!(w2.is_some());
        let w = w2.unwrap();
        assert_eq!(w.warning_type, WarningType::CLUSTER);
        assert_eq!(w.label, "group:distress");
    }
}
