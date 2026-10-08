//! Notification worker: receives `DomainWarning` events from inference, applies the live
//! notification policy, and sends FCM push notifications for qualifying warnings.
//!
//! FCM credentials must be provided through env vars:
//!   SERVICE_ACCOUNT_KEY  – path to the Google service-account JSON file
//!   FCM_TOPIC            – FCM topic to publish to (default: "horse-warnings")

use chrono::Timelike;
use crate::thresholding::{self, AutoThresholdRule};
use shared::{Database, DomainWarning, NotificationSettings, WarningType};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{interval, Duration, MissedTickBehavior};

// ── Service-account JSON shape ────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct ServiceAccountKey {
    project_id: String,
    private_key: String,
    client_email: String,
}

// ── JWT claims for Google OAuth2 token exchange ───────────────────────────────

#[derive(serde::Serialize)]
struct JwtClaims {
    iss: String,
    scope: String,
    aud: String,
    iat: i64,
    exp: i64,
}

// ── Worker entry point ────────────────────────────────────────────────────────

/// Long-running task that does two things concurrently:
///
/// 1. **Warning channel** — reads `DomainWarning` events from `rx`, filters by the
///    live notification policy, and sends FCM push notifications for qualifying warnings.
///
/// 2. **Inactivity timer** — checks once per minute whether the clock matches
///    `inactivity_alert.alert_time`.  When it does (and has not already fired today),
///    it queries the database for label activity over the configured look-back window
///    and sends a push notification if the total count is below `min_detections`.
pub async fn run_notifier(
    mut rx: tokio::sync::mpsc::Receiver<DomainWarning>,
    notification_settings: Arc<Mutex<NotificationSettings>>,
    db: Database,
    auto_threshold_rules: Arc<HashMap<String, AutoThresholdRule>>,
) {
    let client = reqwest::Client::new();

    let mut tick = interval(Duration::from_secs(60));
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);

    // Track the last date on which the inactivity alert fired to prevent duplicates.
    let mut last_inactivity_date: Option<chrono::NaiveDate> = None;

    loop {
        tokio::select! {
            // ── Branch 1: incoming warning ────────────────────────────────
            maybe_warning = rx.recv() => {
                let Some(warning) = maybe_warning else { break; };

                let settings = {
                    let guard = notification_settings.lock().await;
                    guard.clone()
                };

                if !settings.enabled {
                    log::debug!(
                        "Notification suppressed (disabled): {:?} label={}",
                        warning.warning_type,
                        warning.label
                    );
                    continue;
                }

                if warning.severity < settings.severity_threshold {
                    log::debug!(
                        "Notification suppressed (severity {:.2} < threshold {:.2}): {:?} label={}",
                        warning.severity,
                        settings.severity_threshold,
                        warning.warning_type,
                        warning.label
                    );
                    continue;
                }

                match send_push_notification(&client, &warning).await {
                    Ok(()) => log::info!(
                        "Push notification sent: {:?} label={} severity={:.2}",
                        warning.warning_type,
                        warning.label,
                        warning.severity
                    ),
                    Err(e) => log::error!("Failed to send push notification: {}", e),
                }
            }

            // ── Branch 2: per-minute inactivity check ────────────────────
            _ = tick.tick() => {
                let settings = {
                    let guard = notification_settings.lock().await;
                    guard.clone()
                };

                let cfg = &settings.inactivity_alert;
                if !settings.enabled || !cfg.enabled {
                    continue;
                }

                // Parse "HH:MM"
                let (alert_hour, alert_minute) = match parse_hhmm(&cfg.alert_time) {
                    Some(t) => t,
                    None => {
                        log::warn!(
                            "inactivity_alert.alert_time '{}' is not valid HH:MM — skipping check",
                            cfg.alert_time
                        );
                        continue;
                    }
                };

                let now = chrono::Local::now();
                if now.hour() != alert_hour || now.minute() != alert_minute {
                    continue;
                }

                let today = now.date_naive();
                if last_inactivity_date == Some(today) {
                    continue; // already fired today
                }

                // Mark as fired before the async query so a slow query can't double-fire.
                last_inactivity_date = Some(today);

                let window_start = now - chrono::Duration::hours(cfg.look_back_hours as i64);
                match thresholding::get_label_counts_with_auto_thresholds_audited(
                    &db,
                    &cfg.labels,
                    window_start,
                    now,
                    &auto_threshold_rules,
                    Some("inactivity_alert"),
                ).await {
                    Ok(counts) => {
                        let total: i64 = cfg
                            .labels
                            .iter()
                            .map(|l| counts.get(l).copied().unwrap_or(0))
                            .sum();

                        if total < cfg.min_detections {
                            let absent: Vec<&str> = cfg
                                .labels
                                .iter()
                                .filter(|l| counts.get(*l).copied().unwrap_or(0) == 0)
                                .map(String::as_str)
                                .collect();

                            let title = "Inactivity Alert".to_string();
                            let body = if absent.is_empty() {
                                format!(
                                    "Low activity: only {} detection(s) in the past {} hours.",
                                    total, cfg.look_back_hours
                                )
                            } else {
                                format!(
                                    "No {} detected in the past {} hours.",
                                    absent.join(", "),
                                    cfg.look_back_hours
                                )
                            };

                            match send_inactivity_notification(&client, &title, &body).await {
                                Ok(()) => log::info!("Inactivity alert sent: {}", body),
                                Err(e) => log::error!("Failed to send inactivity alert: {}", e),
                            }
                        } else {
                            log::debug!(
                                "Inactivity check passed: {} detection(s) in past {} hours",
                                total, cfg.look_back_hours
                            );
                        }
                    }
                    Err(e) => log::error!("Inactivity check DB query failed: {}", e),
                }
            }
        }
    }

    log::info!("Notifier worker stopped.");
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Parse a "HH:MM" string into `(hour, minute)`.  Returns `None` on any parse error.
fn parse_hhmm(s: &str) -> Option<(u32, u32)> {
    let (h, m) = s.split_once(':')?;
    let hour = h.parse::<u32>().ok().filter(|&v| v < 24)?;
    let minute = m.parse::<u32>().ok().filter(|&v| v < 60)?;
    Some((hour, minute))
}

// ── FCM delivery ──────────────────────────────────────────────────────────────

async fn send_push_notification(
    client: &reqwest::Client,
    warning: &DomainWarning,
) -> anyhow::Result<()> {
    let info = warning_to_notification_text(warning);
    send_topic_notification(client, &info, Some(serde_json::json!({
        "warning_type": format!("{:?}", warning.warning_type),
        "label": warning.label,
        "severity": warning.severity.to_string(),
        "confidence": warning.confidence.to_string(),
        "timestamp": warning.timestamp.to_rfc3339(),
    }))).await
}

pub async fn send_test_push_notification() -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let warning = DomainWarning {
        warning_type: WarningType::ALERT,
        label: "TEST_LABEL".to_string(),
        severity: 0.9,
        confidence: 0.95,
        timestamp: chrono::Utc::now().into(),
    };

    let info = warning_to_notification_text(&warning);
    send_topic_notification(&client, &info, Some(serde_json::json!({
        "warning_type": format!("{:?}", warning.warning_type),
        "label": warning.label,
        "severity": warning.severity.to_string(),
        "confidence": warning.confidence.to_string(),
        "timestamp": warning.timestamp.to_rfc3339(),
    }))).await
}

async fn send_topic_notification(
    client: &reqwest::Client,
    info: &str,
    data: Option<serde_json::Value>,
) -> anyhow::Result<()> {
    let key_path = std::env::var("SERVICE_ACCOUNT_KEY")
        .map_err(|_| anyhow::anyhow!("SERVICE_ACCOUNT_KEY env var not set"))?;

    let key_json = std::fs::read_to_string(&key_path)
        .map_err(|e| anyhow::anyhow!("Failed to read service account key '{}': {}", key_path, e))?;

    let service_account: ServiceAccountKey = serde_json::from_str(&key_json)
        .map_err(|e| anyhow::anyhow!("Failed to parse service account key: {}", e))?;

    let project_id = service_account.project_id.clone();
    let topic = std::env::var("FCM_TOPIC").unwrap_or_else(|_| "horse-warnings".to_string());

    let access_token = get_access_token(client, &service_account).await?;

    // Build the data payload: start with {"info": ...} then merge any extra fields.
    let mut data_map = serde_json::Map::new();
    data_map.insert("info".to_string(), serde_json::Value::String(info.to_string()));
    if let Some(serde_json::Value::Object(extra)) = data {
        data_map.extend(extra);
    }

    let message = serde_json::json!({
        "message": {
            "topic": topic,
            "android": {
                "priority": "high",
            },
            "data": data_map
        }
    });

    let url = format!(
        "https://fcm.googleapis.com/v1/projects/{}/messages:send",
        project_id
    );

    let response = client
        .post(&url)
        .bearer_auth(&access_token)
        .json(&message)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("FCM request failed: {}", e))?;

    let status = response.status();
    if !status.is_success() {
        let body_text = response.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!(
            "FCM returned {}: {}",
            status,
            body_text
        ));
    }

    Ok(())
}

// ── Inactivity alert FCM delivery ─────────────────────────────────────────────

async fn send_inactivity_notification(
    client: &reqwest::Client,
    title: &str,
    body: &str,
) -> anyhow::Result<()> {
    let key_path = std::env::var("SERVICE_ACCOUNT_KEY")
        .map_err(|_| anyhow::anyhow!("SERVICE_ACCOUNT_KEY env var not set"))?;

    let key_json = std::fs::read_to_string(&key_path)
        .map_err(|e| anyhow::anyhow!("Failed to read service account key '{}': {}", key_path, e))?;

    let service_account: ServiceAccountKey = serde_json::from_str(&key_json)
        .map_err(|e| anyhow::anyhow!("Failed to parse service account key: {}", e))?;

    let project_id = service_account.project_id.clone();
    let topic = std::env::var("FCM_TOPIC").unwrap_or_else(|_| "horse-warnings".to_string());

    let access_token = get_access_token(client, &service_account).await?;
    let info_combined = format!("{}: {}", title, body);
    let message = serde_json::json!({
        "message": {
            "topic": topic,
            "android": {
                "priority": "high",
            },
            "data": {
                "info": info_combined,
                "warning_type": "INACTIVITY",
                "timestamp": chrono::Local::now().to_rfc3339(),
            }
        }
    });

    let url = format!(
        "https://fcm.googleapis.com/v1/projects/{}/messages:send",
        project_id
    );

    let response = client
        .post(&url)
        .bearer_auth(&access_token)
        .json(&message)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("FCM request failed: {}", e))?;

    let status = response.status();
    if !status.is_success() {
        let body_text = response.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!("FCM returned {}: {}", status, body_text));
    }

    Ok(())
}

// ── Google OAuth2 JWT token exchange ─────────────────────────────────────────

async fn get_access_token(
    client: &reqwest::Client,
    service_account: &ServiceAccountKey,
) -> anyhow::Result<String> {
    let now = chrono::Utc::now().timestamp();
    let claims = JwtClaims {
        iss: service_account.client_email.clone(),
        scope: "https://www.googleapis.com/auth/firebase.messaging".to_string(),
        aud: "https://oauth2.googleapis.com/token".to_string(),
        iat: now,
        exp: now + 3600,
    };

    let header = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::RS256);
    let encoding_key =
        jsonwebtoken::EncodingKey::from_rsa_pem(service_account.private_key.as_bytes())
            .map_err(|e| anyhow::anyhow!("Invalid RSA private key in service account: {}", e))?;

    let jwt = jsonwebtoken::encode(&header, &claims, &encoding_key)
        .map_err(|e| anyhow::anyhow!("Failed to sign JWT: {}", e))?;

    #[derive(serde::Deserialize)]
    struct TokenResponse {
        access_token: String,
    }

    let response: TokenResponse = client
        .post("https://oauth2.googleapis.com/token")
        .form(&[
            ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
            ("assertion", jwt.as_str()),
        ])
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("Token exchange request failed: {}", e))?
        .json()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to parse token response: {}", e))?;

    Ok(response.access_token)
}

// ── Warning → human-readable text ────────────────────────────────────────────

fn warning_to_notification_text(warning: &DomainWarning) -> String {
    match warning.warning_type {
        WarningType::ALERT => format!(
            "Horse warning:Alert: {} detected (confidence {:.0}%)",
            warning.label,
            warning.confidence * 100.0
        ),
        WarningType::CLUSTER => format!(
            "Horse warning: Cluster activity: {} (severity {:.1})",
            warning.label, warning.severity
        ),
        WarningType::LOUDNESS => format!(
            "Horse warning: Unusual loudness detected (severity {:.1})",
            warning.severity
        ),
    }
}
