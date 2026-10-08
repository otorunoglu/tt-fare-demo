use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::{Pool, Row, Sqlite};

#[derive(Debug)]
pub struct EventSummary {
    pub timestamp: DateTime<Utc>,
    pub label: String,
    pub confidence: f32,
}

pub struct Verifier;

impl Verifier {
    pub async fn fetch_events(pool: &Pool<Sqlite>) -> Result<Vec<EventSummary>> {
        let rows = sqlx::query(
            "SELECT timestamp, label, confidence FROM events ORDER BY timestamp",
        )
        .fetch_all(pool)
        .await?;

        let mut events = Vec::new();
        for row in rows {
            let ts_str: String = row.get("timestamp");
            let timestamp = match DateTime::parse_from_rfc3339(&ts_str) {
                Ok(dt) => dt.with_timezone(&Utc),
                Err(_) => {
                    // Try parsing as naive ISO format (Python style: 2026-02-10T09:13:29.921665)
                    use chrono::NaiveDateTime;
                    match NaiveDateTime::parse_from_str(&ts_str, "%Y-%m-%dT%H:%M:%S%.f") {
                        Ok(ndt) => DateTime::from_naive_utc_and_offset(ndt, Utc),
                        Err(_) => {
                            // Try another format if needed, or fallback to error
                            log::warn!("Failed to parse timestamp: {}", ts_str);
                            continue;
                        }
                    }
                }
            };

            events.push(EventSummary {
                timestamp,
                label: row.get("label"),
                confidence: row.get("confidence"),
            });
        }
        Ok(events)
    }

    pub fn compare(
        rust: &[EventSummary],
        reference: &[EventSummary],
        tolerance_sec: f32,
    ) -> ComparisonReport {
        let mut matched = 0;
        let mut mismatches = Vec::new();
        let mut rust_idx = 0;
        let mut ref_idx = 0;

        if rust.is_empty() || reference.is_empty() {
            return ComparisonReport {
                total_rust: rust.len(),
                total_ref: reference.len(),
                matched: 0,
                accuracy: 0.0,
                mismatches: vec![],
            };
        }

        let t0_rust = rust[0].timestamp;
        let t0_ref = reference[0].timestamp;

        while rust_idx < rust.len() && ref_idx < reference.len() {
            let r = &rust[rust_idx];
            let p = &reference[ref_idx];

            let rel_r = (r.timestamp - t0_rust).num_milliseconds() as f32 / 1000.0;
            let rel_p = (p.timestamp - t0_ref).num_milliseconds() as f32 / 1000.0;

            let diff = rel_r - rel_p;

            if diff.abs() <= tolerance_sec {
                // Check label
                if r.label == p.label {
                    matched += 1;
                } else {
                    mismatches.push(format!(
                        "T+{:.2}s: Rust='{}'({:.2}) vs Ref='{}'({:.2})",
                        rel_r, r.label, r.confidence, p.label, p.confidence
                    ));
                }
                rust_idx += 1;
                ref_idx += 1;
            } else if diff < 0.0 {
                rust_idx += 1;
            } else {
                ref_idx += 1;
            }
        }

        ComparisonReport {
            total_rust: rust.len(),
            total_ref: reference.len(),
            matched,
            accuracy: if matched + mismatches.len() > 0 {
                matched as f32 / (matched + mismatches.len()) as f32
            } else {
                1.0
            },
            mismatches,
        }
    }
}

pub struct ComparisonReport {
    pub total_rust: usize,
    pub total_ref: usize,
    pub matched: usize,
    pub accuracy: f32,
    pub mismatches: Vec<String>,
}
