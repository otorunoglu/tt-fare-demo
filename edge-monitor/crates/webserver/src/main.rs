use axum::{
    extract::{Path as AxumPath, Query, State},
    routing::{get, post, put},
    Json, Router,
};
use axum_server::tls_rustls::RustlsConfig;
use chrono::{Local, NaiveDate, NaiveDateTime, NaiveTime, TimeZone};
use clap::Parser;
use inference::{get_audio_devices, run_inference, CliArgs, RuntimeConfig};
use rustls::crypto::aws_lc_rs;
use serde::{Deserialize, Serialize};
use shared::domain::LoggedEvent;
use shared::{AlertAuditItem, Database, DomainWarning, NotificationSettings, WarningHistoryItem};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::process::Command;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

mod notifier;
mod thresholding;

struct InferenceHandle {
    running: Arc<AtomicBool>,
    handle: tokio::task::JoinHandle<()>,
}

#[derive(Clone)]
struct AppState {
    db: Database,
    inference: Arc<Mutex<Option<InferenceHandle>>>,
    config: Arc<Mutex<RuntimeConfig>>,
    label_groups: Arc<HashMap<String, String>>,
    auto_threshold_rules: Arc<HashMap<String, thresholding::AutoThresholdRule>>,
    notification_settings: Arc<Mutex<NotificationSettings>>,
    /// Cloned into each inference run to forward warnings to the notifier worker.
    warning_tx: Arc<tokio::sync::mpsc::Sender<DomainWarning>>,
    injection_jobs: Arc<Mutex<HashMap<String, InjectAudioJobState>>>,
}

#[derive(Deserialize)]
struct HistoryQuery {
    start: Option<String>,
    end: Option<String>,
    n_warnings: Option<usize>,
}

#[derive(Deserialize)]
struct AlertAuditQuery {
    start: Option<String>,
    end: Option<String>,
    decision_type: Option<String>,
    label: Option<String>,
    limit: Option<usize>,
}

#[derive(Deserialize)]
struct LabelHistoryQuery {
    start: Option<String>,
    end: Option<String>,
    bin_size: Option<i64>,
    n_labels: Option<usize>,
    #[serde(default)]
    grouped: bool,
    #[serde(default)]
    show_uncertain: bool,
}

#[derive(Deserialize)]
struct LoudnessHistoryQuery {
    start: Option<String>,
    end: Option<String>,
    bin_size: Option<i64>, // in seconds
    n_labels: Option<usize>,
}

#[derive(Serialize)]
struct InferenceStatus {
    running: bool,
}

#[derive(Deserialize)]
struct LabelSumRequest {
    date_from: String,
    date_until: String,
    start_time: String,
    end_time: String,
    sum_labels: Vec<String>,
}

#[derive(Serialize)]
struct LabelDurationInfo {
    seconds: f64,
    display: String,
}

#[derive(Serialize)]
struct NightlyLabelSumItem {
    date: String,
    time_window: String,
    label_sums: HashMap<String, LabelDurationInfo>,
}

#[derive(Serialize)]
struct LabelSumResponse {
    items: Vec<NightlyLabelSumItem>,
}

#[derive(Serialize)]
struct PushTestResponse {
    status: &'static str,
    message: String,
}

#[derive(Serialize)]
struct UpdateTriggerResponse {
    status: &'static str,
    message: String,
}

#[derive(Deserialize)]
struct SyncModelsRequest {
    remote: Option<String>,
    zip_name: Option<String>,
    models_dir: Option<String>,
}

#[derive(Serialize)]
struct SyncModelsResponse {
    status: &'static str,
    message: String,
    remote: String,
    zip_name: String,
    models_dir: String,
}

#[derive(Deserialize)]
struct ResetThresholdsRequest {
    label: Option<String>,
}

#[derive(Serialize)]
struct ResetThresholdsResponse {
    status: &'static str,
    message: String,
    removed: u64,
}

#[derive(Deserialize)]
struct InjectAudioRequest {
    file_path: String,
    tag: String,
    labelstudio_json_path: Option<String>,
}

#[derive(Deserialize)]
struct InjectAudioJobsQuery {
    status: Option<String>,
    limit: Option<usize>,
}

#[derive(Serialize)]
struct InjectAudioStartResponse {
    status: &'static str,
    job_id: String,
}

#[derive(Debug, Clone, Serialize)]
struct InjectedDetection {
    timestamp: String,
    label: String,
    confidence: f32,
    loudness: f32,
    source: String,
    tag: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
enum InjectAudioJobStatus {
    Queued,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize)]
struct InjectAudioJobState {
    job_id: String,
    file_path: String,
    tag: String,
    labelstudio_json_path: Option<String>,
    status: InjectAudioJobStatus,
    started_at: Option<String>,
    finished_at: Option<String>,
    error: Option<String>,
    detections: Option<Vec<InjectedDetection>>,
    comparison_summary: Option<InjectComparisonSummary>,
    comparison_error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct InjectComparisonSummary {
    method: String,
    annotation_source: String,
    total_detected: usize,
    total_annotated: usize,
    overlap_count: usize,
    precision: Option<f32>,
    recall: Option<f32>,
    detected_per_label: HashMap<String, usize>,
    annotated_per_label: HashMap<String, usize>,
    overlap_per_label: HashMap<String, usize>,
    notes: Vec<String>,
}

#[derive(serde::Deserialize)]
struct PushAlertQuery {
    name: String,
    stable: String,
    stall: String,
}

fn format_seconds(total_seconds: f64) -> String {
    let total = total_seconds as u64;
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let seconds = total % 60;
    let mut parts: Vec<String> = Vec::new();
    if hours > 0 {
        parts.push(format!("{}h", hours));
    }
    if minutes > 0 {
        parts.push(format!("{}m", minutes));
    }
    if seconds > 0 || parts.is_empty() {
        parts.push(format!("{}s", seconds));
    }
    parts.join(" ")
}

fn parse_flex_naive(s: &str) -> Result<NaiveDateTime, String> {
    // Try ISO 8601 (T) first
    NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S")
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S"))
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f"))
        .map_err(|e| {
            format!(
                "Invalid date format '{}': {}. Use YYYY-MM-DDTHH:MM:SS",
                s, e
            )
        })
}

fn generate_job_id() -> String {
    let now = chrono::Utc::now();
    format!(
        "inject_{}_{}",
        now.timestamp(),
        now.timestamp_subsec_micros()
    )
}

fn is_supported_inject_audio_extension(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase());

    matches!(
        ext.as_deref(),
        Some("wav") | Some("flac") | Some("mp3") | Some("ogg") | Some("m4a") | Some("aac")
    )
}

fn resolve_inject_audio_path(input_path: &str) -> Result<PathBuf, String> {
    if input_path.trim().is_empty() {
        return Err("file_path cannot be empty".to_string());
    }

    let requested = PathBuf::from(input_path);
    let requested = std::fs::canonicalize(&requested)
        .map_err(|e| format!("Failed to resolve file_path '{}': {}", input_path, e))?;

    if !requested.is_file() {
        return Err("file_path is not a file".to_string());
    }

    if !is_supported_inject_audio_extension(&requested) {
        return Err("Unsupported audio format; allowed: wav, flac, mp3, ogg, m4a, aac".to_string());
    }

    let allowed_base =
        std::env::var("INJECT_AUDIO_BASE_DIR").unwrap_or_else(|_| "data".to_string());
    let allowed_base_path = std::fs::canonicalize(&allowed_base).map_err(|e| {
        format!(
            "Allowed base directory '{}' is not accessible: {}",
            allowed_base, e
        )
    })?;

    if !requested.starts_with(&allowed_base_path) {
        return Err(format!(
            "file_path must be inside allowed base directory '{}'",
            allowed_base_path.display()
        ));
    }

    Ok(requested)
}

fn resolve_inject_json_path(input_path: &str) -> Result<PathBuf, String> {
    if input_path.trim().is_empty() {
        return Err("labelstudio_json_path cannot be empty".to_string());
    }

    let requested = PathBuf::from(input_path);
    let requested = std::fs::canonicalize(&requested).map_err(|e| {
        format!(
            "Failed to resolve labelstudio_json_path '{}': {}",
            input_path, e
        )
    })?;

    if !requested.is_file() {
        return Err("labelstudio_json_path is not a file".to_string());
    }

    let is_json = requested
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.eq_ignore_ascii_case("json"))
        .unwrap_or(false);
    if !is_json {
        return Err("labelstudio_json_path must point to a .json file".to_string());
    }

    let allowed_base = std::env::var("INJECT_JSON_BASE_DIR").unwrap_or_else(|_| "data".to_string());
    let allowed_base_path = std::fs::canonicalize(&allowed_base).map_err(|e| {
        format!(
            "Allowed JSON base directory '{}' is not accessible: {}",
            allowed_base, e
        )
    })?;

    if !requested.starts_with(&allowed_base_path) {
        return Err(format!(
            "labelstudio_json_path must be inside allowed base directory '{}'",
            allowed_base_path.display()
        ));
    }

    Ok(requested)
}

fn extract_label_counts_from_task(task: &serde_json::Value, counts: &mut HashMap<String, usize>) {
    let Some(annotations) = task.get("annotations").and_then(|v| v.as_array()) else {
        return;
    };

    for annotation in annotations {
        let Some(results) = annotation.get("result").and_then(|v| v.as_array()) else {
            continue;
        };
        for result in results {
            let Some(kind) = result.get("type").and_then(|v| v.as_str()) else {
                continue;
            };
            if kind != "labels" {
                continue;
            }

            let Some(labels) = result
                .get("value")
                .and_then(|v| v.get("labels"))
                .and_then(|v| v.as_array())
            else {
                continue;
            };

            for label in labels {
                if let Some(label) = label.as_str() {
                    *counts.entry(label.to_string()).or_insert(0) += 1;
                }
            }
        }
    }
}

fn load_labelstudio_annotation_counts(json_path: &Path) -> Result<HashMap<String, usize>, String> {
    let raw = std::fs::read_to_string(json_path).map_err(|e| {
        format!(
            "Failed to read Label Studio json '{}': {}",
            json_path.display(),
            e
        )
    })?;
    let payload: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
        format!(
            "Failed to parse Label Studio json '{}': {}",
            json_path.display(),
            e
        )
    })?;

    let mut counts = HashMap::new();
    match &payload {
        serde_json::Value::Object(_) => extract_label_counts_from_task(&payload, &mut counts),
        serde_json::Value::Array(items) => {
            for item in items {
                extract_label_counts_from_task(item, &mut counts);
            }
        }
        _ => {
            return Err(
                "Label Studio json must be a task object or array of task objects".to_string(),
            )
        }
    }

    Ok(counts)
}

fn build_detection_label_counts(detections: &[InjectedDetection]) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for detection in detections {
        *counts.entry(detection.label.clone()).or_insert(0) += 1;
    }
    counts
}

fn build_inject_comparison_summary(
    annotation_source: &str,
    detections: &[InjectedDetection],
    annotated_counts: &HashMap<String, usize>,
) -> InjectComparisonSummary {
    let detected_counts = build_detection_label_counts(detections);
    let mut overlap_per_label: HashMap<String, usize> = HashMap::new();
    let mut overlap_count = 0usize;

    for (label, &detected) in &detected_counts {
        if let Some(&annotated) = annotated_counts.get(label) {
            let overlap = detected.min(annotated);
            if overlap > 0 {
                overlap_per_label.insert(label.clone(), overlap);
                overlap_count += overlap;
            }
        }
    }

    let total_detected = detected_counts.values().sum();
    let total_annotated = annotated_counts.values().sum();
    let precision = if total_detected > 0 {
        Some(overlap_count as f32 / total_detected as f32)
    } else {
        None
    };
    let recall = if total_annotated > 0 {
        Some(overlap_count as f32 / total_annotated as f32)
    } else {
        None
    };

    let mut notes = Vec::new();
    notes.push("count_based_summary_no_time_alignment".to_string());
    if total_annotated == 0 {
        notes.push("no_annotation_labels_found_in_json".to_string());
    }

    InjectComparisonSummary {
        method: "count_based_label_overlap".to_string(),
        annotation_source: annotation_source.to_string(),
        total_detected,
        total_annotated,
        overlap_count,
        precision,
        recall,
        detected_per_label: detected_counts,
        annotated_per_label: annotated_counts.clone(),
        overlap_per_label,
        notes,
    }
}

fn build_label_group_map(config: &RuntimeConfig) -> HashMap<String, String> {
    match config.load_metamodel_config() {
        Ok(meta) => meta
            .labels
            .into_iter()
            .filter(|label| !label.value.is_empty() && !label.label_group.is_empty())
            .map(|label| (label.value, label.label_group))
            .collect(),
        Err(e) => {
            log::warn!(
                "Failed to load smart stable model config for label grouping: {}",
                e
            );
            HashMap::new()
        }
    }
}

fn build_auto_threshold_rules(
    config: &RuntimeConfig,
) -> HashMap<String, thresholding::AutoThresholdRule> {
    match config.load_metamodel_config() {
        Ok(meta) => thresholding::build_auto_threshold_rule_map(&meta),
        Err(e) => {
            log::warn!(
                "Failed to load smart stable model config for auto-thresholding: {}",
                e
            );
            HashMap::new()
        }
    }
}

fn resolve_label_group(label: &str, label_groups: &HashMap<String, String>) -> String {
    if label == "uncertain" {
        return "uncertain".to_string();
    }

    label_groups
        .get(label)
        .cloned()
        .unwrap_or_else(|| "ungrouped".to_string())
}

fn should_include_label(label: &str, show_uncertain: bool) -> bool {
    show_uncertain || label != "uncertain"
}

fn group_logged_events(
    events: Vec<LoggedEvent>,
    hop_seconds: f32,
    label_groups: &HashMap<String, String>,
) -> HashMap<String, f32> {
    let mut grouped: HashMap<String, f32> = HashMap::new();
    for event in events {
        let group = resolve_label_group(&event.label, label_groups);
        *grouped.entry(group).or_insert(0.0) += hop_seconds;
    }
    grouped
}

fn group_binned_history(
    bins: Vec<HashMap<String, f32>>,
    label_groups: &HashMap<String, String>,
) -> Vec<HashMap<String, f32>> {
    bins.into_iter()
        .map(|bin| {
            let mut grouped_bin: HashMap<String, f32> = HashMap::new();
            for (label, value) in bin {
                let group = resolve_label_group(&label, label_groups);
                *grouped_bin.entry(group).or_insert(0.0) += value;
            }
            grouped_bin
        })
        .collect()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    aws_lc_rs::default_provider()
        .install_default()
        .expect("failed to install rustls crypto provider");

    // Initialize ORT environment globally for the webserver as well
    let _ = ort::init().with_name("stable-monitor-web").commit();

    // Parse config from environment/args for inference defaults
    // Since we don't assume we control CLI args here, we try parse or default
    let args = CliArgs::try_parse().unwrap_or_else(|_| CliArgs::parse_from(["webserver"]));
    let config = RuntimeConfig::from_args(args);
    let db_path = config.get_database_path();
    let db_url = format!("sqlite://{}", db_path.to_string_lossy());
    log::info!("Opening database at {:?} (url: {})", db_path, db_url);

    // Connect in read-write mode to run lightweight schema migrations on startup.
    let db = Database::new(&db_url, false).await?;

    let label_groups = Arc::new(build_label_group_map(&config));
    let auto_threshold_rules = Arc::new(build_auto_threshold_rules(&config));

    // Load notification settings persisted in edge_monitor_config.toml.
    let notification_settings = Arc::new(Mutex::new(config.edge_config.notification.clone()));

    // Warning channel: sender is cloned into each inference run; receiver drives the notifier.
    let (warning_tx, warning_rx) = tokio::sync::mpsc::channel::<DomainWarning>(64);

    let state = AppState {
        db,
        inference: Arc::new(Mutex::new(None)),
        config: Arc::new(Mutex::new(config)),
        label_groups,
        auto_threshold_rules: auto_threshold_rules.clone(),
        notification_settings: notification_settings.clone(),
        warning_tx: Arc::new(warning_tx),
        injection_jobs: Arc::new(Mutex::new(HashMap::new())),
    };

    // Spawn the persistent notification worker.
    // Clone the db pool (cheap — it is Arc-backed) so the notifier can run DB queries
    // independently of the request-handling state.
    let notifier_db = state.db.clone();
    tokio::spawn(notifier::run_notifier(
        warning_rx,
        notification_settings,
        notifier_db,
        auto_threshold_rules,
    ));

    // Auto-start inference on boot so that it resumes after a reboot without manual intervention.
    {
        let running = Arc::new(AtomicBool::new(true));
        let r_clone = running.clone();
        let config = state.config.lock().await.clone();
        let warning_tx = (*state.warning_tx).clone();
        let handle = tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Handle::current();
            rt.block_on(async {
                if let Err(e) = run_inference(config, Some(r_clone), Some(warning_tx)).await {
                    log::error!("Inference task failed: {}", e);
                }
            });
        });
        *state.inference.lock().await = Some(InferenceHandle { running, handle });
        log::info!("Inference auto-started on boot.");
    }

    let app = Router::new()
        .route("/", get(|| async { "Horse Stable Monitor API" }))
        .route("/api/data/warning-history", get(get_warning_history))
        .route(
            "/api/data/acknowledge-warning-by-id",
            post(post_acknowledge_warning),
        )
        .route(
            "/api/data/unacknowledged-warnings",
            get(get_unacknowledged_warnings),
        )
        .route("/api/data/label-history", get(get_label_history))
        .route("/api/data/loudness-history", get(get_loudness_history))
        .route("/api/data/label-sum", post(post_label_sum))
        .route("/api/data/alert-audit", get(get_alert_audit))
        .route("/api/inference/status", get(get_inference_status))
        .route("/api/inference/start", post(start_inference))
        .route("/api/inference/stop", post(stop_inference))
        .route("/api/inference/restart", post(restart_inference))
        .route("/api/inference/audio/list", get(get_audio_device_list))
        .route("/api/inference/audio/select", post(select_audio_device))
        .route("/api/notification/settings", get(get_notification_settings))
        .route(
            "/api/notification/settings",
            put(update_notification_settings),
        )
        .route("/api/clips/list", get(get_clips_list))
        .route("/api/clips/download", get(download_clip))
        .route("/api/clips/push_alert_file", post(post_push_alert_file))
        .route("/api/test/push", post(test_push_notification))
        .route("/api/admin/update", post(trigger_update))
        .route("/api/admin/models/sync", post(sync_models_from_remote))
        .route("/api/admin/pull_weights", get(pull_weights))
        .route("/api/admin/thresholds/reset", post(reset_thresholds))
        .route("/api/test/audio/inject", post(post_inject_audio))
        .route("/api/test/audio/inject/jobs", get(list_inject_audio_jobs))
        .route("/api/test/audio/inject/:job_id", get(get_inject_audio_job))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    log::info!("Webserver listening on https://{}", addr);
    let cert_path = std::env::var("TLS_CERT_PATH").unwrap_or_else(|_| "cert/cert.pem".to_string());
    let key_path = std::env::var("TLS_KEY_PATH").unwrap_or_else(|_| "cert/key.pem".to_string());

    let config = RustlsConfig::from_pem_file(&cert_path, &key_path)
        .await
        .unwrap_or_else(|e| {
            panic!(
                "Failed to load TLS config from {}/{}! {}",
                cert_path, key_path, e
            )
        });
    axum_server::bind_rustls(addr, config)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}

async fn get_warning_history(
    State(state): State<AppState>,
    Query(params): Query<HistoryQuery>,
) -> Result<Json<Vec<WarningHistoryItem>>, (axum::http::StatusCode, String)> {
    if params.start.is_none() || params.end.is_none() {
        let n = params.n_warnings.unwrap_or(5);
        return match state.db.get_last_n_warnings(n).await {
            Ok(history) => Ok(Json(history)),
            Err(e) => Err((
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Database error: {}", e),
            )),
        };
    }

    let start_str = params.start.as_ref().unwrap();
    let end_str = params.end.as_ref().unwrap();

    let s_naive =
        parse_flex_naive(start_str).map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))?;
    let start = Local.from_local_datetime(&s_naive).single().ok_or((
        axum::http::StatusCode::BAD_REQUEST,
        "Invalid start time (ambiguous or invalid)".to_string(),
    ))?;
    let e_naive =
        parse_flex_naive(end_str).map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))?;
    let end = Local.from_local_datetime(&e_naive).single().ok_or((
        axum::http::StatusCode::BAD_REQUEST,
        "Invalid end time (ambiguous or invalid)".to_string(),
    ))?;

    match state.db.get_warning_history(start, end).await {
        Ok(history) => Ok(Json(history)),
        Err(e) => Err((
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Database error: {}", e),
        )),
    }
}

async fn get_alert_audit(
    State(state): State<AppState>,
    Query(params): Query<AlertAuditQuery>,
) -> Result<Json<Vec<AlertAuditItem>>, (axum::http::StatusCode, String)> {
    let start = if let Some(start_str) = params.start.as_ref() {
        let s_naive =
            parse_flex_naive(start_str).map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))?;
        Some(Local.from_local_datetime(&s_naive).single().ok_or((
            axum::http::StatusCode::BAD_REQUEST,
            "Invalid start time (ambiguous or invalid)".to_string(),
        ))?)
    } else {
        None
    };

    let end = if let Some(end_str) = params.end.as_ref() {
        let e_naive =
            parse_flex_naive(end_str).map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))?;
        Some(Local.from_local_datetime(&e_naive).single().ok_or((
            axum::http::StatusCode::BAD_REQUEST,
            "Invalid end time (ambiguous or invalid)".to_string(),
        ))?)
    } else {
        None
    };

    let limit = params.limit.unwrap_or(200).clamp(1, 5000);

    state
        .db
        .get_alert_audit_entries(
            start,
            end,
            params.decision_type.as_deref(),
            params.label.as_deref(),
            limit,
        )
        .await
        .map(Json)
        .map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Database error: {}", e),
            )
        })
}

async fn get_loudness_history(
    State(state): State<AppState>,
    Query(params): Query<LoudnessHistoryQuery>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    if params.start.is_none() || params.end.is_none() {
        let n = params.n_labels.unwrap_or(5);
        return match state.db.get_last_n_labels(n).await {
            Ok(labels) => Ok(Json(serde_json::to_value(labels).map_err(|e| {
                (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Serialization error: {}", e),
                )
            })?)),
            Err(e) => Err((
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Database error: {}", e),
            )),
        };
    }

    // divide time span into time bins of size bin_size and acquire averaged loudness values
    let bin_size = params.bin_size.unwrap_or(60);
    let start_str = params.start.as_ref().unwrap();
    let end_str = params.end.as_ref().unwrap();

    let s_naive =
        parse_flex_naive(start_str).map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))?;
    let start = Local.from_local_datetime(&s_naive).single().ok_or((
        axum::http::StatusCode::BAD_REQUEST,
        "Invalid start time (ambiguous or invalid)".to_string(),
    ))?;
    let e_naive =
        parse_flex_naive(end_str).map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))?;
    let end = Local.from_local_datetime(&e_naive).single().ok_or((
        axum::http::StatusCode::BAD_REQUEST,
        "Invalid end time (ambiguous or invalid)".to_string(),
    ))?;

    match state.db.get_loudness_history(start, end, bin_size).await {
        Ok(history) => {
            let val = serde_json::to_value(history).map_err(|e| {
                (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Serialization error: {}", e),
                )
            })?;
            Ok(Json(val))
        }
        Err(e) => Err((
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Database error: {}", e),
        )),
    }
}

#[derive(Deserialize)]
struct AcknowledgeWarningRequest {
    id: i64,
}

async fn post_acknowledge_warning(
    State(state): State<AppState>,
    Json(payload): Json<AcknowledgeWarningRequest>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    match state.db.acknowledge_warning(payload.id).await {
        Ok(true) => Ok(Json(
            serde_json::json!({"status": "success", "id": payload.id}),
        )),
        Ok(false) => Err((
            axum::http::StatusCode::NOT_FOUND,
            format!("Warning with id {} not found", payload.id),
        )),
        Err(e) => Err((
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Database error: {}", e),
        )),
    }
}

async fn get_unacknowledged_warnings(
    State(state): State<AppState>,
) -> Result<Json<Vec<WarningHistoryItem>>, (axum::http::StatusCode, String)> {
    match state.db.get_unacknowledged_warnings().await {
        Ok(warnings) => Ok(Json(warnings)),
        Err(e) => Err((
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Database error: {}", e),
        )),
    }
}

async fn get_label_history(
    State(state): State<AppState>,
    Query(params): Query<LabelHistoryQuery>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    let hop_seconds = {
        let config = state.config.lock().await;
        config.segment_duration * (1.0 - config.segment_overlap)
    };

    if params.start.is_none() || params.end.is_none() {
        let n = params.n_labels.unwrap_or(5);
        return match state.db.get_last_n_labels(n).await {
            Ok(labels) => {
                let labels: Vec<LoggedEvent> = labels
                    .into_iter()
                    .filter(|event| should_include_label(&event.label, params.show_uncertain))
                    .collect();

                let response = if params.grouped {
                    serde_json::to_value(group_logged_events(
                        labels,
                        hop_seconds,
                        state.label_groups.as_ref(),
                    ))
                } else {
                    serde_json::to_value(labels)
                };

                Ok(Json(response.map_err(|e| {
                    (
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Serialization error: {}", e),
                    )
                })?))
            }
            Err(e) => Err((
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Database error: {}", e),
            )),
        };
    }

    let start_str = params.start.as_ref().unwrap();
    let end_str = params.end.as_ref().unwrap();

    let s_naive =
        parse_flex_naive(start_str).map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))?;
    let e_naive =
        parse_flex_naive(end_str).map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))?;

    let start = Local.from_local_datetime(&s_naive).single().ok_or((
        axum::http::StatusCode::BAD_REQUEST,
        "Invalid start time (ambiguous or invalid)".to_string(),
    ))?;
    let end = Local.from_local_datetime(&e_naive).single().ok_or((
        axum::http::StatusCode::BAD_REQUEST,
        "Invalid end time (ambiguous or invalid)".to_string(),
    ))?;

    let bin_size = params.bin_size.unwrap_or(60);

    match state
        .db
        .get_label_history_binned(start, end, bin_size)
        .await
    {
        Ok(history) => {
            let converted: Vec<HashMap<String, f32>> = history
                .into_iter()
                .map(|bin| {
                    bin.into_iter()
                        .filter(|(label, _)| should_include_label(label, params.show_uncertain))
                        .map(|(k, v)| (k, v as f32 * hop_seconds))
                        .collect()
                })
                .collect();

            let response = if params.grouped {
                serde_json::to_value(group_binned_history(converted, state.label_groups.as_ref()))
            } else {
                serde_json::to_value(converted)
            };

            Ok(Json(response.map_err(|e| {
                (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Serialization error: {}", e),
                )
            })?))
        }
        Err(e) => Err((
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Database error: {}", e),
        )),
    }
}

async fn get_inference_status(State(state): State<AppState>) -> Json<InferenceStatus> {
    let mut inf = state.inference.lock().await;

    // Check if the task is actually still running
    let running = if let Some(handle) = inf.as_ref() {
        if handle.handle.is_finished() {
            // Task died or finished on its own
            *inf = None;
            false
        } else {
            true
        }
    } else {
        false
    };

    Json(InferenceStatus { running })
}

async fn start_inference(
    State(state): State<AppState>,
) -> Result<Json<InferenceStatus>, (axum::http::StatusCode, String)> {
    let mut inf = state.inference.lock().await;

    if inf.is_some() && !inf.as_ref().unwrap().handle.is_finished() {
        return Err((
            axum::http::StatusCode::CONFLICT,
            "Inference is already running".to_string(),
        ));
    }

    let running = Arc::new(AtomicBool::new(true));
    let r_clone = running.clone();
    let config = state.config.lock().await.clone();
    let warning_tx = (*state.warning_tx).clone();

    let handle = tokio::task::spawn_blocking(move || {
        // Since run_inference returns a future, and spawn_blocking expects synchronous closure,
        // we realistically have to use a local block_on here, or since the caller can be async:
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async {
            if let Err(e) = run_inference(config, Some(r_clone), Some(warning_tx)).await {
                log::error!("Inference task failed: {}", e);
            }
        });
    });

    *inf = Some(InferenceHandle { running, handle });

    Ok(Json(InferenceStatus { running: true }))
}

async fn stop_inference(State(state): State<AppState>) -> Json<InferenceStatus> {
    let mut inf = state.inference.lock().await;

    if let Some(handle) = inf.take() {
        log::info!("Stopping inference task...");
        handle.running.store(false, Ordering::SeqCst);
        let _ = handle.handle.await;
        log::info!("Inference task stopped.");
    }

    Json(InferenceStatus { running: false })
}

async fn restart_inference(
    state: State<AppState>,
) -> Result<Json<InferenceStatus>, (axum::http::StatusCode, String)> {
    let _ = stop_inference(state.clone()).await;
    start_inference(state).await
}

#[derive(Serialize)]
struct AudioDeviceList {
    devices: Vec<String>,
    current: Option<String>,
}

#[derive(Deserialize)]
struct SelectAudioDeviceRequest {
    device: String,
}

async fn get_audio_device_list(
    State(state): State<AppState>,
) -> Result<Json<AudioDeviceList>, (axum::http::StatusCode, String)> {
    let devices = match get_audio_devices() {
        Ok(d) => d,
        Err(e) => {
            return Err((
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to get devices: {}", e),
            ))
        }
    };

    let config = state.config.lock().await;
    let current = config.audio_device.clone();

    Ok(Json(AudioDeviceList { devices, current }))
}

fn update_env_audio_device(device: &str) -> anyhow::Result<()> {
    let env_path = PathBuf::from(".env");
    let content = std::fs::read_to_string(&env_path).unwrap_or_default();
    let escaped = device.replace('\\', "\\\\").replace('"', "\\\"");
    let new_line = format!("AUDIO_DEVICE=\"{}\"", escaped);

    let mut updated = false;
    let mut lines = Vec::new();

    for line in content.lines() {
        if line.trim_start().starts_with("AUDIO_DEVICE=") {
            lines.push(new_line.clone());
            updated = true;
        } else {
            lines.push(line.to_string());
        }
    }

    if !updated {
        lines.push(new_line);
    }

    let mut output = lines.join("\n");
    if !output.is_empty() {
        output.push('\n');
    }

    std::fs::write(env_path, output)?;
    Ok(())
}

async fn select_audio_device(
    State(state): State<AppState>,
    Json(payload): Json<SelectAudioDeviceRequest>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    if let Err(e) = update_env_audio_device(&payload.device) {
        return Err((
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to update .env audio device: {}", e),
        ));
    }

    let mut config = state.config.lock().await;
    config.audio_device = Some(payload.device.clone());

    Ok(Json(
        serde_json::json!({"status": "success", "device": payload.device}),
    ))
}

// ── Notification settings ─────────────────────────────────────────────────────

async fn get_notification_settings(State(state): State<AppState>) -> Json<NotificationSettings> {
    let settings = state.notification_settings.lock().await.clone();
    Json(settings)
}

#[derive(Deserialize)]
struct UpdateNotificationSettingsRequest {
    enabled: Option<bool>,
    severity_threshold: Option<f32>,
}

async fn update_notification_settings(
    State(state): State<AppState>,
    Json(payload): Json<UpdateNotificationSettingsRequest>,
) -> Result<Json<NotificationSettings>, (axum::http::StatusCode, String)> {
    let updated = {
        let mut settings = state.notification_settings.lock().await;
        if let Some(enabled) = payload.enabled {
            settings.enabled = enabled;
        }
        if let Some(threshold) = payload.severity_threshold {
            settings.severity_threshold = threshold;
        }
        settings.clone()
    };

    if let Err(e) = update_toml_notification_settings(&updated) {
        return Err((
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to persist notification settings: {}", e),
        ));
    }

    Ok(Json(updated))
}

// ── Alert clips ───────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ClipInfo {
    /// Bare filename (also the value to pass to the download endpoint).
    name: String,
    size_bytes: u64,
    /// Last-modified time as RFC 3339, when available.
    modified: Option<String>,
}

#[derive(Deserialize)]
struct ClipDownloadQuery {
    name: String,
}

/// Resolve the configured alert-clip directory from the live runtime config.
async fn clips_dir(state: &AppState) -> PathBuf {
    let config = state.config.lock().await;
    PathBuf::from(&config.edge_config.recording.output_dir)
}

/// Reject anything that isn't a plain `*.wav` basename, to prevent path traversal.
fn is_safe_clip_name(name: &str) -> bool {
    !name.is_empty()
        && name.ends_with(".wav")
        && !name.contains('/')
        && !name.contains('\\')
        && !name.contains("..")
}

async fn get_clips_list(
    State(state): State<AppState>,
) -> Result<Json<Vec<ClipInfo>>, (axum::http::StatusCode, String)> {
    let dir = clips_dir(&state).await;

    // The directory may not exist yet (recording disabled or no clip saved) — that's fine.
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Json(Vec::new())),
        Err(e) => {
            return Err((
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to read clips dir {:?}: {}", dir, e),
            ))
        }
    };

    let mut clips: Vec<ClipInfo> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("wav") {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let meta = entry.metadata().ok();
        let size_bytes = meta.as_ref().map(|m| m.len()).unwrap_or(0);
        let modified = meta
            .and_then(|m| m.modified().ok())
            .map(|t| chrono::DateTime::<Local>::from(t).to_rfc3339());
        clips.push(ClipInfo {
            name: name.to_string(),
            size_bytes,
            modified,
        });
    }

    // Newest first. Filenames are timestamp-prefixed, so a descending name sort is chronological.
    clips.sort_by(|a, b| b.name.cmp(&a.name));
    Ok(Json(clips))
}

async fn download_clip(
    State(state): State<AppState>,
    Query(params): Query<ClipDownloadQuery>,
) -> Result<axum::response::Response, (axum::http::StatusCode, String)> {
    if !is_safe_clip_name(&params.name) {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "Invalid clip name".to_string(),
        ));
    }

    let path = clips_dir(&state).await.join(&params.name);
    let bytes = tokio::fs::read(&path).await.map_err(|_| {
        (
            axum::http::StatusCode::NOT_FOUND,
            format!("Clip not found: {}", params.name),
        )
    })?;

    axum::response::Response::builder()
        .header(axum::http::header::CONTENT_TYPE, "audio/wav")
        .header(
            axum::http::header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{}\"", params.name),
        )
        .body(axum::body::Body::from(bytes))
        .map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to build response: {}", e),
            )
        })
}

// stable/stall end up in a remote path, so allow only safe chars.
fn is_safe_label(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

async fn pull_weights(
    State(_state): State<AppState>,
) -> Result<String, (axum::http::StatusCode, String)> {
    let output = tokio::process::Command::new("scripts/get_champion_weights.sh")
        .output()
        .await
        .map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("run failed: {e}"),
            )
        })?;
    if !output.status.success() {
        return Err((
            axum::http::StatusCode::BAD_GATEWAY,
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

async fn post_push_alert_file(
    State(state): State<AppState>,
    Json(params): Json<PushAlertQuery>, // was Query(params): Query<PushAlertQuery>
) -> Result<axum::response::Response, (axum::http::StatusCode, String)> {
    use axum::http::StatusCode;

    // Same name guard as download_clip, plus stable/stall validation.
    if !is_safe_clip_name(&params.name) {
        return Err((StatusCode::BAD_REQUEST, "Invalid clip name".to_string()));
    }
    if !is_safe_label(&params.stable) || !is_safe_label(&params.stall) {
        return Err((StatusCode::BAD_REQUEST, "Invalid stable/stall".to_string()));
    }

    // Resolve the clip inside the clips dir (mirrors download_clip).
    let path = clips_dir(&state).await.join(&params.name);
    if !tokio::fs::try_exists(&path).await.unwrap_or(false) {
        return Err((
            StatusCode::NOT_FOUND,
            format!("Clip not found: {}", params.name),
        ));
    }

    // Run the push script: push_alert.sh <stable> <stall> <file>
    // Each value is a separate arg — no shell, so no injection.
    let output = tokio::process::Command::new("scripts/push_alert_file.sh")
        .arg(&params.stable)
        .arg(&params.stall)
        .arg(&path)
        .output()
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to run push script: {e}"),
            )
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err((StatusCode::BAD_GATEWAY, format!("Push failed: {stderr}")));
    }

    Ok(axum::response::Response::builder()
        .status(StatusCode::OK)
        .body(axum::body::Body::from(format!(
            "Pushed {} to {}_{}",
            params.name, params.stable, params.stall
        )))
        .unwrap())
}

async fn test_push_notification() -> Result<Json<PushTestResponse>, (axum::http::StatusCode, String)>
{
    notifier::send_test_push_notification().await.map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to send test push notification: {}", e),
        )
    })?;

    Ok(Json(PushTestResponse {
        status: "ok",
        message: "Test push notification sent".to_string(),
    }))
}

async fn reset_thresholds(
    State(state): State<AppState>,
    Json(payload): Json<ResetThresholdsRequest>,
) -> Result<Json<ResetThresholdsResponse>, (axum::http::StatusCode, String)> {
    let removed = match payload.label.as_deref() {
        Some(label) => state
            .db
            .reset_label_threshold_state(label)
            .await
            .map_err(|e| {
                (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to reset threshold state for '{}': {}", label, e),
                )
            })?,
        None => state
            .db
            .reset_all_label_threshold_states()
            .await
            .map_err(|e| {
                (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to reset threshold states: {}", e),
                )
            })?,
    };

    let message = match payload.label {
        Some(label) => format!("Reset calibration state for label '{}'", label),
        None => "Reset calibration state for all labels".to_string(),
    };

    Ok(Json(ResetThresholdsResponse {
        status: "ok",
        message,
        removed,
    }))
}

async fn post_inject_audio(
    State(state): State<AppState>,
    Json(payload): Json<InjectAudioRequest>,
) -> Result<Json<InjectAudioStartResponse>, (axum::http::StatusCode, String)> {
    if !is_safe_label(&payload.tag) {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "Invalid tag; use only [A-Za-z0-9_-] and max length 64".to_string(),
        ));
    }

    let resolved_path = resolve_inject_audio_path(&payload.file_path)
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))?;
    let resolved_labelstudio_json_path = payload
        .labelstudio_json_path
        .as_deref()
        .map(resolve_inject_json_path)
        .transpose()
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))?;

    let job_id = generate_job_id();
    let initial_state = InjectAudioJobState {
        job_id: job_id.clone(),
        file_path: resolved_path.display().to_string(),
        tag: payload.tag.clone(),
        labelstudio_json_path: resolved_labelstudio_json_path
            .as_ref()
            .map(|p| p.display().to_string()),
        status: InjectAudioJobStatus::Queued,
        started_at: None,
        finished_at: None,
        error: None,
        detections: None,
        comparison_summary: None,
        comparison_error: None,
    };

    {
        let mut jobs = state.injection_jobs.lock().await;
        jobs.insert(job_id.clone(), initial_state);
    }

    let job_state = state.injection_jobs.clone();
    let config_state = state.config.clone();
    let db = state.db.clone();
    let job_id_for_task = job_id.clone();
    let tag = payload.tag.clone();
    let labelstudio_json_path = resolved_labelstudio_json_path.clone();

    tokio::spawn(async move {
        let started_at = chrono::Local::now();

        {
            let mut jobs = job_state.lock().await;
            if let Some(job) = jobs.get_mut(&job_id_for_task) {
                job.status = InjectAudioJobStatus::Running;
                job.started_at = Some(started_at.to_rfc3339());
            }
        }

        let run_result = {
            let mut replay_config = config_state.lock().await.clone();
            replay_config.simulate_file = Some(resolved_path.clone());
            replay_config.event_source = "file_replay".to_string();
            replay_config.replay_tag = Some(tag.clone());
            replay_config.stop_on_stream_end = true;

            let running = Arc::new(AtomicBool::new(true));
            run_inference(replay_config, Some(running), None).await
        };

        let finished_at = chrono::Local::now();

        match run_result {
            Ok(()) => {
                let detections = db
                    .get_labels_by_source_tag_window(
                        "file_replay",
                        Some(&tag),
                        started_at,
                        finished_at + chrono::Duration::seconds(1),
                    )
                    .await;

                let mut jobs = job_state.lock().await;
                if let Some(job) = jobs.get_mut(&job_id_for_task) {
                    match detections {
                        Ok(rows) => {
                            let mapped_detections: Vec<InjectedDetection> = rows
                                .into_iter()
                                .map(|row| InjectedDetection {
                                    timestamp: row.timestamp.to_rfc3339(),
                                    label: row.label,
                                    confidence: row.confidence,
                                    loudness: row.loudness,
                                    source: row.source,
                                    tag: row.replay_tag,
                                })
                                .collect();

                            let mut comparison_summary = None;
                            let mut comparison_error = None;
                            if let Some(json_path) = labelstudio_json_path.as_ref() {
                                match load_labelstudio_annotation_counts(json_path) {
                                    Ok(annotation_counts) => {
                                        comparison_summary = Some(build_inject_comparison_summary(
                                            &json_path.display().to_string(),
                                            &mapped_detections,
                                            &annotation_counts,
                                        ));
                                    }
                                    Err(e) => {
                                        comparison_error = Some(e);
                                    }
                                }
                            }

                            job.status = InjectAudioJobStatus::Completed;
                            job.finished_at = Some(finished_at.to_rfc3339());
                            job.detections = Some(mapped_detections);
                            job.comparison_summary = comparison_summary;
                            job.comparison_error = comparison_error;
                        }
                        Err(e) => {
                            job.status = InjectAudioJobStatus::Failed;
                            job.finished_at = Some(finished_at.to_rfc3339());
                            job.error = Some(format!(
                                "Replay succeeded but detection fetch failed: {}",
                                e
                            ));
                        }
                    }
                }
            }
            Err(e) => {
                let mut jobs = job_state.lock().await;
                if let Some(job) = jobs.get_mut(&job_id_for_task) {
                    job.status = InjectAudioJobStatus::Failed;
                    job.finished_at = Some(finished_at.to_rfc3339());
                    job.error = Some(format!("Replay job failed: {}", e));
                }
            }
        }
    });

    Ok(Json(InjectAudioStartResponse {
        status: "accepted",
        job_id,
    }))
}

async fn get_inject_audio_job(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> Result<Json<InjectAudioJobState>, (axum::http::StatusCode, String)> {
    let jobs = state.injection_jobs.lock().await;
    let Some(job) = jobs.get(&job_id) else {
        return Err((
            axum::http::StatusCode::NOT_FOUND,
            format!("No inject-audio job found for id '{}'", job_id),
        ));
    };
    Ok(Json(job.clone()))
}

fn inject_status_name(status: &InjectAudioJobStatus) -> &'static str {
    match status {
        InjectAudioJobStatus::Queued => "queued",
        InjectAudioJobStatus::Running => "running",
        InjectAudioJobStatus::Completed => "completed",
        InjectAudioJobStatus::Failed => "failed",
    }
}

async fn list_inject_audio_jobs(
    State(state): State<AppState>,
    Query(params): Query<InjectAudioJobsQuery>,
) -> Result<Json<Vec<InjectAudioJobState>>, (axum::http::StatusCode, String)> {
    let status_filter = params
        .status
        .as_ref()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty());

    if let Some(status) = status_filter.as_deref() {
        let valid = matches!(status, "queued" | "running" | "completed" | "failed");
        if !valid {
            return Err((
                axum::http::StatusCode::BAD_REQUEST,
                "Invalid status filter. Use one of: queued, running, completed, failed".to_string(),
            ));
        }
    }

    let limit = params.limit.unwrap_or(200).clamp(1, 1000);

    let jobs = state.injection_jobs.lock().await;
    let mut items: Vec<InjectAudioJobState> = jobs.values().cloned().collect();

    if let Some(status) = status_filter.as_deref() {
        items.retain(|job| inject_status_name(&job.status) == status);
    }

    // job_id starts with inject_<timestamp>_<micros>, so desc gives newest first.
    items.sort_by(|a, b| b.job_id.cmp(&a.job_id));
    items.truncate(limit);

    Ok(Json(items))
}

async fn trigger_update(
) -> Result<(axum::http::StatusCode, Json<UpdateTriggerResponse>), (axum::http::StatusCode, String)>
{
    let mut command = Command::new("sudo");
    command
        .arg("/bin/systemctl")
        .arg("start")
        .arg("stable-edge-monitor-update.service")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    command.spawn().map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to start update service: {}", e),
        )
    })?;

    Ok((
        axum::http::StatusCode::ACCEPTED,
        Json(UpdateTriggerResponse {
            status: "accepted",
            message: "Update service triggered".to_string(),
        }),
    ))
}

fn parse_champion_onnx_version(filename: &str) -> Option<u64> {
    let prefix = "champion_v";
    let suffix = "_onnx.zip";
    if !(filename.starts_with(prefix) && filename.ends_with(suffix)) {
        return None;
    }
    let ver = &filename[prefix.len()..filename.len() - suffix.len()];
    ver.parse::<u64>().ok()
}

async fn rclone_list_onnx_archives(
    remote: &str,
) -> Result<Vec<String>, (axum::http::StatusCode, String)> {
    let output = Command::new("rclone")
        .arg("lsf")
        .arg(remote)
        .arg("--files-only")
        .arg("--include")
        .arg("*_onnx.zip")
        .output()
        .await
        .map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to run rclone lsf: {}", e),
            )
        })?;

    if !output.status.success() {
        return Err((
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "rclone lsf failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        ));
    }

    let files = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();

    Ok(files)
}

fn pick_latest_archive(mut files: Vec<String>) -> Option<String> {
    if files.is_empty() {
        return None;
    }

    files.sort();
    files
        .iter()
        .filter_map(|name| parse_champion_onnx_version(name).map(|version| (version, name)))
        .max_by_key(|(version, _)| *version)
        .map(|(_, name)| name.clone())
        .or_else(|| files.last().cloned())
}

async fn sync_models_from_remote(
    Json(payload): Json<SyncModelsRequest>,
) -> Result<(axum::http::StatusCode, Json<SyncModelsResponse>), (axum::http::StatusCode, String)> {
    let remote = payload
        .remote
        .unwrap_or_else(|| {
            std::env::var("MODEL_WEIGHTS_REMOTE")
                .unwrap_or_else(|_| "model_weights:weights".to_string())
        })
        .trim()
        .to_string();

    if remote.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "Remote must not be empty".to_string(),
        ));
    }

    let zip_name = if let Some(name) = payload.zip_name {
        let clean = name.trim().to_string();
        if clean.is_empty() {
            return Err((
                axum::http::StatusCode::BAD_REQUEST,
                "zip_name must not be empty".to_string(),
            ));
        }
        clean
    } else {
        let files = rclone_list_onnx_archives(&remote).await?;
        pick_latest_archive(files).ok_or((
            axum::http::StatusCode::NOT_FOUND,
            "No ONNX archive found on remote".to_string(),
        ))?
    };

    let models_dir = payload
        .models_dir
        .unwrap_or_else(|| {
            std::env::var("MODEL_WEIGHTS_DIR").unwrap_or_else(|_| "data/models".to_string())
        })
        .trim()
        .to_string();

    if models_dir.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "models_dir must not be empty".to_string(),
        ));
    }

    let temp_zip_path: PathBuf = std::env::temp_dir().join(&zip_name);
    let remote_zip_path = format!("{}/{}", remote.trim_end_matches('/'), zip_name);

    let copy_status = Command::new("rclone")
        .arg("copyto")
        .arg(&remote_zip_path)
        .arg(&temp_zip_path)
        .output()
        .await
        .map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to run rclone copyto: {}", e),
            )
        })?;

    if !copy_status.status.success() {
        return Err((
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "rclone copyto failed: {}",
                String::from_utf8_lossy(&copy_status.stderr).trim()
            ),
        ));
    }

    std::fs::create_dir_all(&models_dir).map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to create models dir '{}': {}", models_dir, e),
        )
    })?;

    let unzip_status = Command::new("unzip")
        .arg("-o")
        .arg(&temp_zip_path)
        .arg("-d")
        .arg(&models_dir)
        .output()
        .await
        .map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to run unzip: {}", e),
            )
        })?;

    if !unzip_status.status.success() {
        return Err((
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "unzip failed: {}",
                String::from_utf8_lossy(&unzip_status.stderr).trim()
            ),
        ));
    }

    if let Err(e) = std::fs::remove_file(&temp_zip_path) {
        log::warn!(
            "Failed to remove temporary archive {:?}: {}",
            temp_zip_path,
            e
        );
    }

    Ok((
        axum::http::StatusCode::OK,
        Json(SyncModelsResponse {
            status: "ok",
            message: "Model archive downloaded and extracted".to_string(),
            remote,
            zip_name,
            models_dir,
        }),
    ))
}

fn update_toml_notification_settings(settings: &NotificationSettings) -> anyhow::Result<()> {
    let path_str = std::env::var("EDGE_MONITOR_CONFIG_PATH")
        .unwrap_or_else(|_| "config/edge_monitor_config.toml".to_string());
    let config_path = std::path::Path::new(&path_str);
    let content = if config_path.exists() {
        std::fs::read_to_string(config_path)?
    } else {
        String::new()
    };

    let mut doc = content.parse::<toml_edit::DocumentMut>()?;

    if !doc.contains_key("notification") {
        doc.insert(
            "notification",
            toml_edit::Item::Table(toml_edit::Table::new()),
        );
    }

    if let Some(notification) = doc["notification"].as_table_mut() {
        notification.insert("enabled", toml_edit::value(settings.enabled));
        notification.insert(
            "severity_threshold",
            toml_edit::value(settings.severity_threshold as f64),
        );
    }

    if let Some(parent) = config_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(config_path, doc.to_string())?;
    Ok(())
}

async fn post_label_sum(
    State(state): State<AppState>,
    Json(payload): Json<LabelSumRequest>,
) -> Result<Json<LabelSumResponse>, (axum::http::StatusCode, String)> {
    if payload.sum_labels.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "sum_labels must not be empty".to_string(),
        ));
    }

    let date_from = NaiveDate::parse_from_str(&payload.date_from, "%Y-%m-%d").map_err(|e| {
        (
            axum::http::StatusCode::BAD_REQUEST,
            format!("Invalid date_from '{}': {}", payload.date_from, e),
        )
    })?;
    let date_until = NaiveDate::parse_from_str(&payload.date_until, "%Y-%m-%d").map_err(|e| {
        (
            axum::http::StatusCode::BAD_REQUEST,
            format!("Invalid date_until '{}': {}", payload.date_until, e),
        )
    })?;

    if date_until < date_from {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "date_until must be on or after date_from".to_string(),
        ));
    }

    let start_time = NaiveTime::parse_from_str(&payload.start_time, "%H:%M").map_err(|e| {
        (
            axum::http::StatusCode::BAD_REQUEST,
            format!("Invalid start_time '{}': {}", payload.start_time, e),
        )
    })?;
    let end_time = NaiveTime::parse_from_str(&payload.end_time, "%H:%M").map_err(|e| {
        (
            axum::http::StatusCode::BAD_REQUEST,
            format!("Invalid end_time '{}': {}", payload.end_time, e),
        )
    })?;

    let hop_seconds = {
        let config = state.config.lock().await;
        f64::from(config.segment_duration * (1.0 - config.segment_overlap))
    };

    let time_window = format!("{}-{}", payload.start_time, payload.end_time);
    let mut items: Vec<NightlyLabelSumItem> = Vec::new();
    let mut day = date_from;

    while day <= date_until {
        let window_start_naive = NaiveDateTime::new(day, start_time);
        // Overnight window: if end_time <= start_time, the window ends the next day
        let window_end_naive = if end_time <= start_time {
            let next_day = day.succ_opt().ok_or((
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "Date overflow computing window end".to_string(),
            ))?;
            NaiveDateTime::new(next_day, end_time)
        } else {
            NaiveDateTime::new(day, end_time)
        };

        let window_start = Local
            .from_local_datetime(&window_start_naive)
            .single()
            .ok_or((
                axum::http::StatusCode::BAD_REQUEST,
                "Ambiguous or invalid window start time".to_string(),
            ))?;
        let window_end = Local
            .from_local_datetime(&window_end_naive)
            .single()
            .ok_or((
                axum::http::StatusCode::BAD_REQUEST,
                "Ambiguous or invalid window end time".to_string(),
            ))?;

        let counts = thresholding::get_label_counts_with_auto_thresholds_audited(
            &state.db,
            &payload.sum_labels,
            window_start,
            window_end,
            &state.auto_threshold_rules,
            Some("label_sum"),
        )
        .await
        .map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Database error: {}", e),
            )
        })?;

        let mut label_sums: HashMap<String, LabelDurationInfo> = HashMap::new();
        for label in &payload.sum_labels {
            let count = counts.get(label.as_str()).copied().unwrap_or(0);
            let total_seconds = count as f64 * hop_seconds;
            label_sums.insert(
                label.clone(),
                LabelDurationInfo {
                    seconds: total_seconds,
                    display: format_seconds(total_seconds),
                },
            );
        }

        items.push(NightlyLabelSumItem {
            date: day.format("%Y-%m-%d").to_string(),
            time_window: time_window.clone(),
            label_sums,
        });

        day = day.succ_opt().ok_or((
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "Date overflow iterating days".to_string(),
        ))?;
    }

    log::info!(
        "Label sum computed for {} nights ({} to {})",
        items.len(),
        payload.date_from,
        payload.date_until
    );

    Ok(Json(LabelSumResponse { items }))
}
