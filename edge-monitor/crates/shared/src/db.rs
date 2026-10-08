use crate::domain::{
    DomainWarning, InferenceEvent, LabelProb, LoggedEvent, LoudnessBin, WarningHistoryItem,
};
use anyhow::Result;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{FromRow, Pool, Sqlite};
use std::collections::HashMap;
use std::str::FromStr;

#[derive(Debug, Clone, FromRow)]
pub struct LabelThresholdCalibrationState {
    pub label: String,
    pub calibration_mode: String,
    pub threshold_percentile: f32,
    pub safety_margin_db: f32,
    pub min_samples_for_calibration: i64,
    pub rolling_window: Option<String>,
    pub sample_count: i64,
    pub raw_cutoff_db: Option<f32>,
    pub final_threshold_db: Option<f32>,
    pub calibrated: bool,
    pub updated_at: chrono::DateTime<chrono::Local>,
}

#[derive(Debug, Clone, FromRow)]
pub struct LabelLoudnessEvent {
    pub timestamp: chrono::DateTime<chrono::Local>,
    pub loudness: f32,
}

#[derive(Debug, Clone, serde::Serialize, FromRow)]
pub struct AlertAuditItem {
    pub id: i64,
    pub created_at: chrono::DateTime<chrono::Local>,
    pub decision_type: String,
    pub label: String,
    pub window_start: chrono::DateTime<chrono::Local>,
    pub window_end: chrono::DateTime<chrono::Local>,
    pub threshold_db: f32,
    pub detection_timestamp: chrono::DateTime<chrono::Local>,
    pub detection_loudness: f32,
    pub reason: String,
}

#[derive(Clone)]
pub struct Database {
    pool: Pool<Sqlite>,
}

impl Database {
    pub async fn new(db_url: &str, read_only: bool) -> Result<Self> {
        if !read_only {
            if let Some(db_path) = sqlite_file_path_from_url(db_url) {
                if let Some(parent) = db_path.parent() {
                    if !parent.as_os_str().is_empty() {
                        std::fs::create_dir_all(parent)?;
                    }
                }
            }
        }

        let mut options = SqliteConnectOptions::from_str(db_url)?
            .create_if_missing(!read_only)
            .read_only(read_only);

        if !read_only {
            options = options.journal_mode(sqlx::sqlite::SqliteJournalMode::Wal);
        }

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await?;

        let db = Self { pool };

        if !read_only {
            db.init().await?;
        }

        Ok(db)
    }

    pub fn pool(&self) -> &Pool<Sqlite> {
        &self.pool
    }

    async fn init(&self) -> Result<()> {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS events (
                timestamp TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                loudness REAL NOT NULL,
                source TEXT NOT NULL DEFAULT 'live',
                replay_tag TEXT,
                secondary_labels TEXT
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
            .execute(&self.pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_events_label ON events(label)")
            .execute(&self.pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_events_label_timestamp ON events(label, timestamp)")
            .execute(&self.pool)
            .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS warnings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                label TEXT NOT NULL,
                warning_type TEXT NOT NULL,
                severity REAL NOT NULL,
                confidence REAL NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_warnings_timestamp ON warnings(timestamp)")
            .execute(&self.pool)
            .await?;

        // Lightweight migration for existing DBs: add acknowledge tracking.
        let has_ack_col: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM pragma_table_info('warnings') WHERE name = 'acknowledged'",
        )
        .fetch_one(&self.pool)
        .await?;
        if has_ack_col == 0 {
            sqlx::query("ALTER TABLE warnings ADD COLUMN acknowledged INTEGER NOT NULL DEFAULT 0")
                .execute(&self.pool)
                .await?;
        }

        let has_source_col: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM pragma_table_info('events') WHERE name = 'source'",
        )
        .fetch_one(&self.pool)
        .await?;
        if has_source_col == 0 {
            sqlx::query("ALTER TABLE events ADD COLUMN source TEXT NOT NULL DEFAULT 'live'")
                .execute(&self.pool)
                .await?;
        }

        let has_replay_tag_col: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM pragma_table_info('events') WHERE name = 'replay_tag'",
        )
        .fetch_one(&self.pool)
        .await?;
        if has_replay_tag_col == 0 {
            sqlx::query("ALTER TABLE events ADD COLUMN replay_tag TEXT")
                .execute(&self.pool)
                .await?;
        }

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_events_source_timestamp ON events(source, timestamp)")
            .execute(&self.pool)
            .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS label_thresholds (
                label TEXT PRIMARY KEY,
                calibration_mode TEXT NOT NULL,
                threshold_percentile REAL NOT NULL,
                safety_margin_db REAL NOT NULL,
                min_samples_for_calibration INTEGER NOT NULL,
                rolling_window TEXT,
                sample_count INTEGER NOT NULL,
                raw_cutoff_db REAL,
                final_threshold_db REAL,
                calibrated INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS alert_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                decision_type TEXT NOT NULL,
                label TEXT NOT NULL,
                window_start TEXT NOT NULL,
                window_end TEXT NOT NULL,
                threshold_db REAL NOT NULL,
                detection_timestamp TEXT NOT NULL,
                detection_loudness REAL NOT NULL,
                reason TEXT NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_alert_audit_decision_window ON alert_audit(decision_type, window_start, window_end)",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_alert_audit_detection_timestamp ON alert_audit(detection_timestamp)",
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn log_event(&self, event: &InferenceEvent) -> Result<()> {
        // Find dominant label and top 2 secondary labels
        let mut sorted_probs: Vec<(&String, &f32)> = event.label_probs.iter().collect();
        sorted_probs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Extract dominant label for flat columns (compatibility and indexing)
        let (label, confidence) = if sorted_probs.is_empty() {
            ("unknown".to_string(), 0.0)
        } else {
            let top = sorted_probs[0];
            (top.0.clone(), *top.1)
        };

        let mut secondary_labels: Vec<LabelProb> = Vec::new();
        for (l, p) in sorted_probs.into_iter().skip(1).take(2) {
            let rounded_prob = (*p * 100.0).round() / 100.0;
            if rounded_prob > 0.0 {
                secondary_labels.push(LabelProb {
                    label: l.clone(),
                    prob: rounded_prob,
                });
            }
        }

        let secondary_labels_json = serde_json::to_string(&secondary_labels)?;
        let insert_result = sqlx::query(
            "INSERT OR REPLACE INTO events (timestamp, label, confidence, loudness, source, replay_tag, secondary_labels)
             VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(event.timestamp)
        .bind(&label)
        .bind(confidence)
        .bind(event.loudness)
        .bind(&event.source)
        .bind(&event.replay_tag)
        .bind(&secondary_labels_json)
        .execute(&self.pool)
        .await;

        match insert_result {
            Ok(_) => {}
            Err(err) if is_missing_source_column_error(&err) => {
                log::warn!(
                    "Legacy events schema detected (missing source/replay_tag); writing compatibility event row."
                );
                sqlx::query(
                    "INSERT OR REPLACE INTO events (timestamp, label, confidence, loudness, secondary_labels)
                     VALUES (?, ?, ?, ?, ?)"
                )
                .bind(event.timestamp)
                .bind(&label)
                .bind(confidence)
                .bind(event.loudness)
                .bind(&secondary_labels_json)
                .execute(&self.pool)
                .await?;
            }
            Err(err) => return Err(err.into()),
        }
        Ok(())
    }

    pub async fn log_warning(&self, warning: &DomainWarning) -> Result<()> {
        sqlx::query(
            "INSERT INTO warnings (timestamp, label, warning_type, severity, confidence)
             VALUES (?, ?, ?, ?, ?)",
        )
        .bind(warning.timestamp)
        .bind(&warning.label)
        .bind(format!("{:?}", warning.warning_type))
        .bind(warning.severity)
        .bind(warning.confidence)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn get_warning_history(
        &self,
        start: chrono::DateTime<chrono::Local>,
        end: chrono::DateTime<chrono::Local>,
    ) -> Result<Vec<WarningHistoryItem>> {
        let warnings = sqlx::query_as::<_, WarningHistoryItem>(
            "SELECT id, timestamp, label, warning_type, severity, confidence, acknowledged FROM warnings 
             WHERE timestamp >= ? AND timestamp <= ? 
             ORDER BY timestamp ASC",
        )
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await?;
        Ok(warnings)
    }

    pub async fn get_last_n_warnings(&self, n: usize) -> Result<Vec<WarningHistoryItem>> {
        let mut warnings = sqlx::query_as::<_, WarningHistoryItem>(
            "SELECT id, timestamp, label, warning_type, severity, confidence, acknowledged FROM warnings 
             ORDER BY timestamp DESC LIMIT ?",
        )
        .bind(n as i64)
        .fetch_all(&self.pool)
        .await?;

        // Reverse to ensure chronological order like the history query
        warnings.reverse();
        Ok(warnings)
    }

    pub async fn acknowledge_warning(&self, id: i64) -> Result<bool> {
        let update = sqlx::query("UPDATE warnings SET acknowledged = 1 WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;

        if update.rows_affected() > 0 {
            return Ok(true);
        }

        let exists: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM warnings WHERE id = ?")
            .bind(id)
            .fetch_one(&self.pool)
            .await?;
        Ok(exists > 0)
    }
    
    pub async fn get_unacknowledged_warnings(&self) -> Result<Vec<WarningHistoryItem>> {
        let warnings = sqlx::query_as::<_, WarningHistoryItem>(
            "SELECT id, timestamp, label, warning_type, severity, confidence, acknowledged FROM warnings 
             WHERE acknowledged = 0
             ORDER BY timestamp ASC",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(warnings)
    }

    pub async fn get_last_n_labels(&self, n: usize) -> Result<Vec<LoggedEvent>> {
        let labels_query = sqlx::query_as::<_, LoggedEvent>(
            "SELECT timestamp, label, confidence, loudness, source, replay_tag, secondary_labels FROM events 
             ORDER BY timestamp DESC LIMIT ?",
        )
        .bind(n as i64)
        .fetch_all(&self.pool)
        .await;

        let mut labels = match labels_query {
            Ok(rows) => rows,
            Err(err) if is_missing_source_column_error(&err) => {
                log::warn!(
                    "Legacy events schema detected (missing source/replay_tag); falling back to compatibility label query."
                );
                sqlx::query_as::<_, (chrono::DateTime<chrono::Local>, String, f32, f32, String)>(
                    "SELECT timestamp, label, confidence, loudness, COALESCE(secondary_labels, '[]') FROM events
                     ORDER BY timestamp DESC LIMIT ?",
                )
                .bind(n as i64)
                .fetch_all(&self.pool)
                .await?
                .into_iter()
                .map(|(timestamp, label, confidence, loudness, secondary_labels_json)| {
                    let secondary_labels = serde_json::from_str(&secondary_labels_json)
                        .unwrap_or_else(|_| Vec::new());
                    LoggedEvent {
                        timestamp,
                        label,
                        confidence,
                        loudness,
                        source: "live".to_string(),
                        replay_tag: None,
                        secondary_labels,
                    }
                })
                .collect()
            }
            Err(err) => return Err(err.into()),
        };

        // Reverse to ensure chronological order
        labels.reverse();
        Ok(labels)
    }

    pub async fn get_labels_by_source_tag_window(
        &self,
        source: &str,
        replay_tag: Option<&str>,
        start: chrono::DateTime<chrono::Local>,
        end: chrono::DateTime<chrono::Local>,
    ) -> Result<Vec<LoggedEvent>> {
        let compatibility_requested = source == "live" && replay_tag.is_none();

        let rows = if let Some(tag) = replay_tag {
            let query_result = sqlx::query_as::<_, LoggedEvent>(
                "SELECT timestamp, label, confidence, loudness, source, replay_tag, secondary_labels FROM events
                 WHERE source = ? AND replay_tag = ? AND timestamp >= ? AND timestamp <= ?
                 ORDER BY timestamp ASC",
            )
            .bind(source)
            .bind(tag)
            .bind(start)
            .bind(end)
            .fetch_all(&self.pool)
            .await;

            match query_result {
                Ok(rows) => rows,
                Err(err) if is_missing_source_column_error(&err) && compatibility_requested => {
                    log::warn!(
                        "Legacy events schema detected (missing source/replay_tag); falling back to compatibility source query."
                    );
                    sqlx::query_as::<_, (chrono::DateTime<chrono::Local>, String, f32, f32, String)>(
                        "SELECT timestamp, label, confidence, loudness, COALESCE(secondary_labels, '[]') FROM events
                         WHERE timestamp >= ? AND timestamp <= ?
                         ORDER BY timestamp ASC",
                    )
                    .bind(start)
                    .bind(end)
                    .fetch_all(&self.pool)
                    .await?
                    .into_iter()
                    .map(|(timestamp, label, confidence, loudness, secondary_labels_json)| {
                        let secondary_labels = serde_json::from_str(&secondary_labels_json)
                            .unwrap_or_else(|_| Vec::new());
                        LoggedEvent {
                            timestamp,
                            label,
                            confidence,
                            loudness,
                            source: "live".to_string(),
                            replay_tag: None,
                            secondary_labels,
                        }
                    })
                    .collect()
                }
                Err(err) => return Err(err.into()),
            }
        } else {
            let query_result = sqlx::query_as::<_, LoggedEvent>(
                "SELECT timestamp, label, confidence, loudness, source, replay_tag, secondary_labels FROM events
                 WHERE source = ? AND timestamp >= ? AND timestamp <= ?
                 ORDER BY timestamp ASC",
            )
            .bind(source)
            .bind(start)
            .bind(end)
            .fetch_all(&self.pool)
            .await;

            match query_result {
                Ok(rows) => rows,
                Err(err) if is_missing_source_column_error(&err) && compatibility_requested => {
                    log::warn!(
                        "Legacy events schema detected (missing source/replay_tag); falling back to compatibility source query."
                    );
                    sqlx::query_as::<_, (chrono::DateTime<chrono::Local>, String, f32, f32, String)>(
                        "SELECT timestamp, label, confidence, loudness, COALESCE(secondary_labels, '[]') FROM events
                         WHERE timestamp >= ? AND timestamp <= ?
                         ORDER BY timestamp ASC",
                    )
                    .bind(start)
                    .bind(end)
                    .fetch_all(&self.pool)
                    .await?
                    .into_iter()
                    .map(|(timestamp, label, confidence, loudness, secondary_labels_json)| {
                        let secondary_labels = serde_json::from_str(&secondary_labels_json)
                            .unwrap_or_else(|_| Vec::new());
                        LoggedEvent {
                            timestamp,
                            label,
                            confidence,
                            loudness,
                            source: "live".to_string(),
                            replay_tag: None,
                            secondary_labels,
                        }
                    })
                    .collect()
                }
                Err(err) => return Err(err.into()),
            }
        };

        Ok(rows)
    }

    pub async fn get_loudness_history(
        &self,
        start: chrono::DateTime<chrono::Local>,
        end: chrono::DateTime<chrono::Local>,
        bin_size_seconds: i64,
    ) -> Result<Vec<LoudnessBin>> {
        let rows = sqlx::query_as::<_, (i64, f64)>(
            "SELECT 
                CAST((strftime('%s', timestamp) - strftime('%s', ?)) / ? AS INTEGER) as bin_idx,
                AVG(loudness) as avg_loudness
            FROM events 
            WHERE timestamp >= ? AND timestamp < ?
            GROUP BY bin_idx
            ORDER BY bin_idx ASC",
        )
        .bind(start)
        .bind(bin_size_seconds)
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await?;

        // Determine total bins
        let duration = end.timestamp() - start.timestamp();
        let num_bins = ((duration as f64 / bin_size_seconds as f64).ceil() as usize).max(1);

        let mut bins: Vec<LoudnessBin> = Vec::with_capacity(num_bins);
        for i in 0..num_bins {
            let bin_start = start
                + chrono::Duration::try_seconds((i as i64) * bin_size_seconds).unwrap_or_default();
            bins.push(LoudnessBin {
                time: bin_start,
                value: 0.0,
            });
        }

        for (bin_idx, avg_loudness) in rows {
            if bin_idx >= 0 && (bin_idx as usize) < num_bins {
                bins[bin_idx as usize].value = avg_loudness;
            }
        }

        Ok(bins)
    }

    pub async fn get_label_counts_for_window(
        &self,
        start: chrono::DateTime<chrono::Local>,
        end: chrono::DateTime<chrono::Local>,
    ) -> Result<HashMap<String, i64>> {
        let rows = sqlx::query_as::<_, (String, i64)>(
            "SELECT label, COUNT(*) as count
            FROM events
            WHERE timestamp >= ? AND timestamp < ?
            GROUP BY label",
        )
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await?;

        let mut counts = HashMap::new();
        for (label, count) in rows {
            counts.insert(label, count);
        }
        Ok(counts)
    }

    pub async fn get_label_count_for_window(
        &self,
        label: &str,
        start: chrono::DateTime<chrono::Local>,
        end: chrono::DateTime<chrono::Local>,
        min_loudness_db: Option<f32>,
    ) -> Result<i64> {
        let count = match min_loudness_db {
            Some(threshold) => {
                sqlx::query_scalar::<_, i64>(
                    "SELECT COUNT(*) FROM events
                    WHERE label = ? AND timestamp >= ? AND timestamp < ? AND loudness >= ?",
                )
                .bind(label)
                .bind(start)
                .bind(end)
                .bind(threshold)
                .fetch_one(&self.pool)
                .await?
            }
            None => {
                sqlx::query_scalar::<_, i64>(
                    "SELECT COUNT(*) FROM events
                    WHERE label = ? AND timestamp >= ? AND timestamp < ?",
                )
                .bind(label)
                .bind(start)
                .bind(end)
                .fetch_one(&self.pool)
                .await?
            }
        };

        Ok(count)
    }

    pub async fn get_label_events_for_window(
        &self,
        label: &str,
        start: chrono::DateTime<chrono::Local>,
        end: chrono::DateTime<chrono::Local>,
    ) -> Result<Vec<LabelLoudnessEvent>> {
        let rows = sqlx::query_as::<_, LabelLoudnessEvent>(
            "SELECT timestamp, loudness FROM events
            WHERE label = ? AND timestamp >= ? AND timestamp < ?
            ORDER BY timestamp ASC",
        )
        .bind(label)
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    pub async fn get_label_loudness_history(
        &self,
        label: &str,
        start: Option<chrono::DateTime<chrono::Local>>,
        end: chrono::DateTime<chrono::Local>,
    ) -> Result<Vec<f32>> {
        let rows_result: std::result::Result<Vec<f32>, sqlx::Error> = match start {
            Some(start_ts) => {
                sqlx::query_scalar::<_, f32>(
                    "SELECT loudness FROM events
                    WHERE label = ? AND source = 'live' AND timestamp >= ? AND timestamp < ?
                    ORDER BY timestamp ASC",
                )
                .bind(label)
                .bind(start_ts)
                .bind(end)
                .fetch_all(&self.pool)
                .await
            }
            None => {
                sqlx::query_scalar::<_, f32>(
                    "SELECT loudness FROM events
                    WHERE label = ? AND source = 'live' AND timestamp < ?
                    ORDER BY timestamp ASC",
                )
                .bind(label)
                .bind(end)
                .fetch_all(&self.pool)
                .await
            }
        };

        let rows = match rows_result {
            Ok(rows) => rows,
            Err(err) if is_missing_source_column_error(&err) => {
                log::warn!(
                    "Legacy events schema detected (missing source/replay_tag); calibrating from untagged history."
                );
                match start {
                    Some(start_ts) => {
                        sqlx::query_scalar::<_, f32>(
                            "SELECT loudness FROM events
                            WHERE label = ? AND timestamp >= ? AND timestamp < ?
                            ORDER BY timestamp ASC",
                        )
                        .bind(label)
                        .bind(start_ts)
                        .bind(end)
                        .fetch_all(&self.pool)
                        .await?
                    }
                    None => {
                        sqlx::query_scalar::<_, f32>(
                            "SELECT loudness FROM events
                            WHERE label = ? AND timestamp < ?
                            ORDER BY timestamp ASC",
                        )
                        .bind(label)
                        .bind(end)
                        .fetch_all(&self.pool)
                        .await?
                    }
                }
            }
            Err(err) => return Err(err.into()),
        };

        Ok(rows)
    }

    pub async fn get_label_threshold_state(
        &self,
        label: &str,
    ) -> Result<Option<LabelThresholdCalibrationState>> {
        let state = sqlx::query_as::<_, LabelThresholdCalibrationState>(
            "SELECT
                label,
                calibration_mode,
                threshold_percentile,
                safety_margin_db,
                min_samples_for_calibration,
                rolling_window,
                sample_count,
                raw_cutoff_db,
                final_threshold_db,
                calibrated,
                updated_at
             FROM label_thresholds
             WHERE label = ?",
        )
        .bind(label)
        .fetch_optional(&self.pool)
        .await?;

        Ok(state)
    }

    pub async fn upsert_label_threshold_state(
        &self,
        state: &LabelThresholdCalibrationState,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO label_thresholds (
                label,
                calibration_mode,
                threshold_percentile,
                safety_margin_db,
                min_samples_for_calibration,
                rolling_window,
                sample_count,
                raw_cutoff_db,
                final_threshold_db,
                calibrated,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(label) DO UPDATE SET
                calibration_mode = excluded.calibration_mode,
                threshold_percentile = excluded.threshold_percentile,
                safety_margin_db = excluded.safety_margin_db,
                min_samples_for_calibration = excluded.min_samples_for_calibration,
                rolling_window = excluded.rolling_window,
                sample_count = excluded.sample_count,
                raw_cutoff_db = excluded.raw_cutoff_db,
                final_threshold_db = excluded.final_threshold_db,
                calibrated = excluded.calibrated,
                updated_at = excluded.updated_at",
        )
        .bind(&state.label)
        .bind(&state.calibration_mode)
        .bind(state.threshold_percentile)
        .bind(state.safety_margin_db)
        .bind(state.min_samples_for_calibration)
        .bind(&state.rolling_window)
        .bind(state.sample_count)
        .bind(state.raw_cutoff_db)
        .bind(state.final_threshold_db)
        .bind(state.calibrated)
        .bind(state.updated_at)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn reset_label_threshold_state(&self, label: &str) -> Result<u64> {
        let result = sqlx::query("DELETE FROM label_thresholds WHERE label = ?")
            .bind(label)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected())
    }

    pub async fn reset_all_label_threshold_states(&self) -> Result<u64> {
        let result = sqlx::query("DELETE FROM label_thresholds")
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected())
    }

    pub async fn log_alert_audit_exclusions(
        &self,
        decision_type: &str,
        label: &str,
        window_start: chrono::DateTime<chrono::Local>,
        window_end: chrono::DateTime<chrono::Local>,
        threshold_db: f32,
        excluded_events: &[LabelLoudnessEvent],
    ) -> Result<()> {
        if excluded_events.is_empty() {
            return Ok(());
        }

        let created_at = chrono::Local::now();
        for event in excluded_events {
            sqlx::query(
                "INSERT INTO alert_audit (
                    created_at,
                    decision_type,
                    label,
                    window_start,
                    window_end,
                    threshold_db,
                    detection_timestamp,
                    detection_loudness,
                    reason
                 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(created_at)
            .bind(decision_type)
            .bind(label)
            .bind(window_start)
            .bind(window_end)
            .bind(threshold_db)
            .bind(event.timestamp)
            .bind(event.loudness)
            .bind("below_label_threshold")
            .execute(&self.pool)
            .await?;
        }

        Ok(())
    }

    pub async fn get_alert_audit_count_for_decision(&self, decision_type: &str) -> Result<i64> {
        let count = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM alert_audit WHERE decision_type = ?",
        )
        .bind(decision_type)
        .fetch_one(&self.pool)
        .await?;
        Ok(count)
    }

    pub async fn get_alert_audit_entries(
        &self,
        start: Option<chrono::DateTime<chrono::Local>>,
        end: Option<chrono::DateTime<chrono::Local>>,
        decision_type: Option<&str>,
        label: Option<&str>,
        limit: usize,
    ) -> Result<Vec<AlertAuditItem>> {
        let mut sql = String::from(
            "SELECT
                id,
                created_at,
                decision_type,
                label,
                window_start,
                window_end,
                threshold_db,
                detection_timestamp,
                detection_loudness,
                reason
             FROM alert_audit",
        );

        let mut clauses: Vec<&str> = Vec::new();
        if start.is_some() {
            clauses.push("created_at >= ?");
        }
        if end.is_some() {
            clauses.push("created_at <= ?");
        }
        if decision_type.is_some() {
            clauses.push("decision_type = ?");
        }
        if label.is_some() {
            clauses.push("label = ?");
        }

        if !clauses.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&clauses.join(" AND "));
        }

        sql.push_str(" ORDER BY created_at DESC LIMIT ?");

        let mut query = sqlx::query_as::<_, AlertAuditItem>(&sql);
        if let Some(start) = start {
            query = query.bind(start);
        }
        if let Some(end) = end {
            query = query.bind(end);
        }
        if let Some(decision_type) = decision_type {
            query = query.bind(decision_type);
        }
        if let Some(label) = label {
            query = query.bind(label);
        }

        query = query.bind(limit as i64);

        let rows = query.fetch_all(&self.pool).await?;
        Ok(rows)
    }

    pub async fn get_label_history_binned(
        &self,
        start: chrono::DateTime<chrono::Local>,
        end: chrono::DateTime<chrono::Local>,
        bin_size_seconds: i64,
    ) -> Result<Vec<HashMap<String, i64>>> {
        let rows = sqlx::query_as::<_, (i64, String, i64)>(
            "SELECT 
                CAST((strftime('%s', timestamp) - strftime('%s', ?)) / ? AS INTEGER) as bin_idx,
                label,
                COUNT(*) as count
            FROM events 
            WHERE timestamp >= ? AND timestamp < ?
            GROUP BY bin_idx, label
            ORDER BY bin_idx ASC",
        )
        .bind(start)
        .bind(bin_size_seconds)
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await?;

        // Determine total bins
        let duration = end.timestamp() - start.timestamp();
        let num_bins = ((duration as f64 / bin_size_seconds as f64).ceil() as usize).max(1);

        let mut bins: Vec<HashMap<String, i64>> = vec![HashMap::new(); num_bins];

        for (bin_idx, label, count) in rows {
            if bin_idx >= 0 && (bin_idx as usize) < num_bins {
                bins[bin_idx as usize].insert(label, count);
            }
        }

        Ok(bins)
    }
}

fn sqlite_file_path_from_url(db_url: &str) -> Option<std::path::PathBuf> {
    if db_url == "sqlite::memory:" {
        return None;
    }

    let path_part = db_url
        .strip_prefix("sqlite://")
        .or_else(|| db_url.strip_prefix("sqlite:"))
        .unwrap_or(db_url);
    let path_part = path_part.split('?').next().unwrap_or(path_part);

    if path_part.is_empty() || path_part == ":memory:" {
        None
    } else {
        Some(std::path::PathBuf::from(path_part))
    }
}

fn is_missing_source_column_error(err: &sqlx::Error) -> bool {
    let text = err.to_string().to_ascii_lowercase();
    text.contains("no such column: source")
        || text.contains("table events has no column named source")
}
