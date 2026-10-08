use anyhow::Result;
use chrono::{DateTime, Duration, Local};
use inference::config::{CalibrationMode, LabelInfo, SmartStableModelConfig, ThresholdMode};
use shared::{Database, LabelLoudnessEvent, LabelThresholdCalibrationState};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AutoThresholdRule {
    pub threshold_mode: ThresholdMode,
    pub manual_threshold_db: Option<f32>,
    pub threshold_percentile: f32,
    pub safety_margin_db: f32,
    pub min_samples_for_calibration: usize,
    pub calibration_mode: CalibrationMode,
    pub rolling_window: String,
}

#[derive(Debug, Clone)]
pub struct ThresholdComputation {
    pub sample_count: usize,
    pub calibrated: bool,
    pub raw_cutoff_db: Option<f32>,
    pub final_threshold_db: Option<f32>,
}

pub fn build_auto_threshold_rule_map(
    config: &SmartStableModelConfig,
) -> HashMap<String, AutoThresholdRule> {
    config
        .labels
        .iter()
        .filter_map(|label| {
            auto_threshold_rule_for_label(label).map(|rule| (label.value.clone(), rule))
        })
        .collect()
}

pub fn auto_threshold_rule_for_label(label: &LabelInfo) -> Option<AutoThresholdRule> {
    if !label.auto_thresholding {
        return None;
    }

    Some(AutoThresholdRule {
        threshold_mode: label.threshold_mode.clone(),
        manual_threshold_db: label.manual_threshold_db,
        threshold_percentile: label.threshold_percentile,
        safety_margin_db: label.safety_margin_db,
        min_samples_for_calibration: label.min_samples_for_calibration,
        calibration_mode: label.calibration_mode.clone(),
        rolling_window: label.rolling_window.clone(),
    })
}

pub async fn compute_effective_label_thresholds(
    db: &Database,
    labels: &[String],
    rules: &HashMap<String, AutoThresholdRule>,
    now: DateTime<Local>,
) -> Result<HashMap<String, Option<f32>>> {
    let mut thresholds = HashMap::with_capacity(labels.len());

    for label in labels {
        let threshold = match rules.get(label) {
            Some(rule) => resolve_threshold_for_label(db, label, rule, now).await?,
            None => None,
        };

        thresholds.insert(label.clone(), threshold);
    }

    Ok(thresholds)
}

#[allow(dead_code)]
pub async fn get_label_counts_with_auto_thresholds(
    db: &Database,
    labels: &[String],
    start: DateTime<Local>,
    end: DateTime<Local>,
    rules: &HashMap<String, AutoThresholdRule>,
) -> Result<HashMap<String, i64>> {
    get_label_counts_with_auto_thresholds_audited(db, labels, start, end, rules, None).await
}

pub async fn get_label_counts_with_auto_thresholds_audited(
    db: &Database,
    labels: &[String],
    start: DateTime<Local>,
    end: DateTime<Local>,
    rules: &HashMap<String, AutoThresholdRule>,
    decision_type: Option<&str>,
) -> Result<HashMap<String, i64>> {
    let thresholds = compute_effective_label_thresholds(db, labels, rules, end).await?;
    let mut counts = HashMap::with_capacity(labels.len());

    for label in labels {
        let min_loudness = thresholds.get(label).and_then(|v| *v);
        let count = if let Some(threshold_db) = min_loudness {
            let events = db.get_label_events_for_window(label, start, end).await?;
            let (included_count, excluded_events) = split_by_threshold(&events, threshold_db);

            if let Some(kind) = decision_type {
                db.log_alert_audit_exclusions(
                    kind,
                    label,
                    start,
                    end,
                    threshold_db,
                    &excluded_events,
                )
                .await?;
            }

            included_count
        } else {
            db.get_label_count_for_window(label, start, end, None).await?
        };

        counts.insert(label.clone(), count);
    }

    Ok(counts)
}

fn split_by_threshold(events: &[LabelLoudnessEvent], threshold_db: f32) -> (i64, Vec<LabelLoudnessEvent>) {
    let mut included = 0_i64;
    let mut excluded = Vec::new();

    for event in events {
        if event.loudness >= threshold_db {
            included += 1;
        } else {
            excluded.push(event.clone());
        }
    }

    (included, excluded)
}

pub async fn resolve_threshold_for_label(
    db: &Database,
    label: &str,
    rule: &AutoThresholdRule,
    now: DateTime<Local>,
) -> Result<Option<f32>> {
    if matches!(rule.threshold_mode, ThresholdMode::Manual) {
        let manual_threshold = rule.manual_threshold_db.unwrap_or(-18.0);
        let state = LabelThresholdCalibrationState {
            label: label.to_string(),
            calibration_mode: "manual".to_string(),
            threshold_percentile: clamp_percentile(rule.threshold_percentile),
            safety_margin_db: rule.safety_margin_db,
            min_samples_for_calibration: rule.min_samples_for_calibration as i64,
            rolling_window: Some(rule.rolling_window.clone()),
            sample_count: 0,
            raw_cutoff_db: None,
            final_threshold_db: Some(manual_threshold),
            calibrated: true,
            updated_at: now,
        };
        db.upsert_label_threshold_state(&state).await?;
        return Ok(Some(manual_threshold));
    }

    let existing = db.get_label_threshold_state(label).await?;

    if matches!(rule.calibration_mode, CalibrationMode::FixedOnce)
        && state_matches_rule(existing.as_ref(), rule)
    {
        if let Some(existing_state) = existing {
            if existing_state.calibrated {
                return Ok(existing_state.final_threshold_db);
            }
            return Ok(None);
        }
    }

    let window_start = match rule.calibration_mode {
        CalibrationMode::Rolling => Some(now - parse_rolling_window(&rule.rolling_window)),
        CalibrationMode::FixedOnce => None,
    };

    let history = db.get_label_loudness_history(label, window_start, now).await?;
    let computation = compute_threshold_from_history(
        &history,
        rule.threshold_percentile,
        rule.safety_margin_db,
        rule.min_samples_for_calibration,
    );

    let state = LabelThresholdCalibrationState {
        label: label.to_string(),
        calibration_mode: calibration_mode_name(&rule.calibration_mode).to_string(),
        threshold_percentile: clamp_percentile(rule.threshold_percentile),
        safety_margin_db: rule.safety_margin_db,
        min_samples_for_calibration: rule.min_samples_for_calibration as i64,
        rolling_window: Some(rule.rolling_window.clone()),
        sample_count: computation.sample_count as i64,
        raw_cutoff_db: computation.raw_cutoff_db,
        final_threshold_db: computation.final_threshold_db,
        calibrated: computation.calibrated,
        updated_at: now,
    };
    db.upsert_label_threshold_state(&state).await?;

    Ok(computation.final_threshold_db)
}

pub fn compute_threshold_from_history(
    history_db: &[f32],
    threshold_percentile: f32,
    safety_margin_db: f32,
    min_samples_for_calibration: usize,
) -> ThresholdComputation {
    let sample_count = history_db.len();
    if sample_count < min_samples_for_calibration {
        return ThresholdComputation {
            sample_count,
            calibrated: false,
            raw_cutoff_db: None,
            final_threshold_db: None,
        };
    }

    let raw_cutoff_db = percentile_nearest_rank(history_db, clamp_percentile(threshold_percentile));
    let final_threshold_db = raw_cutoff_db - safety_margin_db;

    ThresholdComputation {
        sample_count,
        calibrated: true,
        raw_cutoff_db: Some(raw_cutoff_db),
        final_threshold_db: Some(final_threshold_db),
    }
}

fn clamp_percentile(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn state_matches_rule(
    existing: Option<&LabelThresholdCalibrationState>,
    rule: &AutoThresholdRule,
) -> bool {
    let Some(existing) = existing else {
        return false;
    };

    existing.calibration_mode == calibration_mode_name(&rule.calibration_mode)
        && (existing.threshold_percentile - clamp_percentile(rule.threshold_percentile)).abs() < 1e-6
        && (existing.safety_margin_db - rule.safety_margin_db).abs() < 1e-6
        && existing.min_samples_for_calibration == rule.min_samples_for_calibration as i64
        && existing.rolling_window.as_deref() == Some(rule.rolling_window.as_str())
}

fn calibration_mode_name(mode: &CalibrationMode) -> &'static str {
    match mode {
        CalibrationMode::Rolling => "rolling",
        CalibrationMode::FixedOnce => "fixed_once",
    }
}

fn parse_rolling_window(raw: &str) -> Duration {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Duration::days(14);
    }

    let (num_part, unit_part) = trimmed.split_at(trimmed.len() - 1);
    let Ok(value) = num_part.parse::<i64>() else {
        return Duration::days(14);
    };

    if value <= 0 {
        return Duration::days(14);
    }

    match unit_part {
        "d" | "D" => Duration::days(value),
        "h" | "H" => Duration::hours(value),
        "m" | "M" => Duration::minutes(value),
        _ => Duration::days(14),
    }
}

fn percentile_nearest_rank(values: &[f32], percentile: f32) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }

    let p = clamp_percentile(percentile);
    let rank = ((p * n as f32).ceil() as isize - 1).clamp(0, (n - 1) as isize) as usize;
    sorted[rank]
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::InferenceEvent;
    use std::collections::HashMap;

    #[test]
    fn test_min_samples_gating_uncalibrated() {
        let history = vec![-48.0, -42.0, -37.0];
        let computed = compute_threshold_from_history(&history, 0.80, 3.0, 5);

        assert_eq!(computed.sample_count, 3);
        assert!(!computed.calibrated);
        assert_eq!(computed.raw_cutoff_db, None);
        assert_eq!(computed.final_threshold_db, None);
    }

    #[test]
    fn test_percentile_plus_margin_deterministic() {
        let history = vec![-60.0, -40.0, -20.0, -10.0, -5.0];
        let computed = compute_threshold_from_history(&history, 0.80, 3.0, 5);

        assert!(computed.calibrated);
        assert_eq!(computed.raw_cutoff_db, Some(-10.0));
        assert_eq!(computed.final_threshold_db, Some(-13.0));
    }

    #[tokio::test]
    async fn test_rolling_recalibration_reads_raw_history_not_prior_state() {
        let db = Database::new("sqlite::memory:", false)
            .await
            .expect("db should initialize");
        let base = Local::now();

        for (idx, loudness) in [1.0_f32, 2.0, 100.0, 101.0].iter().enumerate() {
            let mut probs = HashMap::new();
            probs.insert("eating".to_string(), 0.99);
            let event = InferenceEvent {
                timestamp: base + Duration::seconds(idx as i64),
                label_probs: probs,
                loudness: *loudness,
                source: "live".to_string(),
                replay_tag: None,
                embedding: None,
            };
            db.log_event(&event).await.expect("event insert should work");
        }

        db.upsert_label_threshold_state(&LabelThresholdCalibrationState {
            label: "eating".to_string(),
            calibration_mode: "rolling".to_string(),
            threshold_percentile: 0.5,
            safety_margin_db: 0.0,
            min_samples_for_calibration: 1,
            rolling_window: Some("14d".to_string()),
            sample_count: 1,
            raw_cutoff_db: Some(999.0),
            final_threshold_db: Some(999.0),
            calibrated: true,
            updated_at: base,
        })
        .await
        .expect("state insert should work");

        let rule = AutoThresholdRule {
            threshold_mode: ThresholdMode::Auto,
            manual_threshold_db: None,
            threshold_percentile: 0.5,
            safety_margin_db: 0.0,
            min_samples_for_calibration: 1,
            calibration_mode: CalibrationMode::Rolling,
            rolling_window: "14d".to_string(),
        };

        let resolved = resolve_threshold_for_label(&db, "eating", &rule, base + Duration::seconds(10))
            .await
            .expect("threshold resolve should work");

        assert_eq!(resolved, Some(2.0));

        let persisted = db
            .get_label_threshold_state("eating")
            .await
            .expect("state query should work")
            .expect("state should exist");

        assert_eq!(persisted.sample_count, 4);
        assert_eq!(persisted.raw_cutoff_db, Some(2.0));
    }

    #[tokio::test]
    async fn test_audited_count_logs_excluded_detections() {
        let db = Database::new("sqlite::memory:", false)
            .await
            .expect("db should initialize");
        let base = Local::now();

        for (idx, loudness) in [1.0_f32, 10.0].iter().enumerate() {
            let mut probs = HashMap::new();
            probs.insert("eating".to_string(), 0.99);
            let event = InferenceEvent {
                timestamp: base + Duration::seconds(idx as i64),
                label_probs: probs,
                loudness: *loudness,
                source: "live".to_string(),
                replay_tag: None,
                embedding: None,
            };
            db.log_event(&event).await.expect("event insert should work");
        }

        let mut rules = HashMap::new();
        rules.insert(
            "eating".to_string(),
            AutoThresholdRule {
                threshold_mode: ThresholdMode::Auto,
                manual_threshold_db: None,
                threshold_percentile: 1.0,
                safety_margin_db: 0.0,
                min_samples_for_calibration: 1,
                calibration_mode: CalibrationMode::Rolling,
                rolling_window: "14d".to_string(),
            },
        );

        let labels = vec!["eating".to_string()];
        let counts = get_label_counts_with_auto_thresholds_audited(
            &db,
            &labels,
            base,
            base + Duration::seconds(10),
            &rules,
            Some("inactivity_alert"),
        )
        .await
        .expect("audited count should succeed");

        assert_eq!(counts.get("eating"), Some(&1));

        let audit_count = db
            .get_alert_audit_count_for_decision("inactivity_alert")
            .await
            .expect("audit count query should succeed");
        assert_eq!(audit_count, 1);
    }

    #[tokio::test]
    async fn test_manual_threshold_mode_bypasses_calibration() {
        let db = Database::new("sqlite::memory:", false)
            .await
            .expect("db should initialize");

        let rule = AutoThresholdRule {
            threshold_mode: ThresholdMode::Manual,
            manual_threshold_db: Some(-18.0),
            threshold_percentile: 0.8,
            safety_margin_db: 3.0,
            min_samples_for_calibration: 999,
            calibration_mode: CalibrationMode::Rolling,
            rolling_window: "14d".to_string(),
        };

        let resolved = resolve_threshold_for_label(&db, "eating", &rule, Local::now())
            .await
            .expect("manual resolve should succeed");

        assert_eq!(resolved, Some(-18.0));

        let persisted = db
            .get_label_threshold_state("eating")
            .await
            .expect("state read should succeed")
            .expect("state should exist");

        assert_eq!(persisted.calibration_mode, "manual");
        assert_eq!(persisted.final_threshold_db, Some(-18.0));
        assert_eq!(persisted.sample_count, 0);
    }

    #[tokio::test]
    async fn test_calibration_uses_live_source_only() {
        let db = Database::new("sqlite::memory:", false)
            .await
            .expect("db should initialize");
        let base = Local::now();

        let mut live_probs = HashMap::new();
        live_probs.insert("eating".to_string(), 0.9);
        db.log_event(&InferenceEvent {
            timestamp: base,
            label_probs: live_probs,
            loudness: -20.0,
            source: "live".to_string(),
            replay_tag: None,
            embedding: None,
        })
        .await
        .expect("live event insert should succeed");

        let mut replay_probs = HashMap::new();
        replay_probs.insert("eating".to_string(), 0.9);
        db.log_event(&InferenceEvent {
            timestamp: base + Duration::seconds(1),
            label_probs: replay_probs,
            loudness: -1.0,
            source: "file_replay".to_string(),
            replay_tag: Some("cone_v2".to_string()),
            embedding: None,
        })
        .await
        .expect("replay event insert should succeed");

        let rule = AutoThresholdRule {
            threshold_mode: ThresholdMode::Auto,
            manual_threshold_db: None,
            threshold_percentile: 1.0,
            safety_margin_db: 0.0,
            min_samples_for_calibration: 1,
            calibration_mode: CalibrationMode::Rolling,
            rolling_window: "14d".to_string(),
        };

        let resolved = resolve_threshold_for_label(&db, "eating", &rule, base + Duration::seconds(10))
            .await
            .expect("auto resolve should succeed");

        assert_eq!(resolved, Some(-20.0));
    }
}
