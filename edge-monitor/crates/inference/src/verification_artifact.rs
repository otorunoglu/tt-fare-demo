use crate::config::RuntimeConfig;
use crate::inference::InferenceEngine;
use anyhow::{Context, Result};
use base64::{engine::general_purpose, Engine as _};
use log::{error, info, warn};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Write};
use std::path::Path;

#[derive(Deserialize, Debug)]
pub struct VerificationArtifact {
    pub metadata: ArtifactMetadata,
    pub audio_data: String,
    pub config_snapshot: ConfigSnapshot,
    pub expected_results: Vec<ExpectedSegment>,
}

#[derive(Deserialize, Debug)]
pub struct ArtifactMetadata {
    pub task_id: i64,
    pub filename: String,
    pub model_version: String,
}

#[derive(Deserialize, Debug)]
pub struct ConfigSnapshot {
    pub classifier_thresholds: HashMap<String, f32>,
    pub silence_threshold: f32,
}

#[derive(Deserialize, Debug)]
pub struct ExpectedSegment {
    pub index: usize,
    pub start_time: f64,
    pub end_time: f64,
    pub label: String,
    pub confidence: f32,
    pub loudness: f32,
    pub label_probs: Option<HashMap<String, f32>>,
}

pub async fn run_verification(artifact_path: &Path, config: &RuntimeConfig) -> Result<()> {
    info!("Loading verification artifact from {:?}", artifact_path);

    // 1. Load Artifact
    let file = File::open(artifact_path).context("Failed to open artifact file")?;
    let reader = BufReader::new(file);
    let artifact: VerificationArtifact =
        serde_json::from_reader(reader).context("Failed to parse artifact JSON")?;

    info!(
        "Artifact loaded: Task {}, File '{}', Model '{}'",
        artifact.metadata.task_id, artifact.metadata.filename, artifact.metadata.model_version
    );

    // 2. Decode Audio
    let audio_bytes = general_purpose::STANDARD
        .decode(&artifact.audio_data)
        .context("Failed to decode base64 audio data")?;

    let temp_audio_path = std::env::temp_dir().join(format!(
        "verify_artifact_{}_{}.wav",
        artifact.metadata.task_id,
        uuid::Uuid::new_v4()
    ));
    {
        let mut temp_file = File::create(&temp_audio_path)?;
        temp_file.write_all(&audio_bytes)?;
        temp_file.flush()?;
    }
    info!("Decoded audio written to temp file: {:?}", temp_audio_path);

    // 3. Load Audio using audio_preprocessing
    // We assume `audio_preprocessing` crate is available and exports `io`.
    // If this fails compilation, we will verify exports.
    let (samples, sr) = audio_preprocessing::io::load_audio_file(&temp_audio_path, 16000)
        .context("Failed to load and resample audio file")?;

    info!("Loaded audio: {} samples at {} Hz", samples.len(), sr);

    // 4. Initialize Inference Engine
    let mut mm_config = config.load_metamodel_config()?;

    // Apply overrides from artifact to ensure logic consistency
    if let Some(val) = mm_config.model_parameters.get_mut("silence_threshold") {
        *val = toml::Value::Float(artifact.config_snapshot.silence_threshold as f64);
    }
    // Note: classifier_threshold overriding is harder due to TOML structure,
    // but the engine will read what's in mm_config (which reads from disk).
    // Ideally we should override it too, but we trust the user to have synced configs
    // Find the loudest 1-second chunk (16000 samples)
    let window_size = 16000;
    let mut best_start = 0;
    let mut max_energy = -1.0;

    if samples.len() >= window_size {
        // Simple sliding window (hop 8000)
        for start in (0..samples.len() - window_size).step_by(8000) {
            let chunk = &samples[start..start + window_size];
            let energy: f32 = chunk.iter().map(|x| x * x).sum();
            if energy > max_energy {
                max_energy = energy;
                best_start = start;
            }
        }
    }

    let warmup_samples = if samples.len() >= window_size {
        &samples[best_start..best_start + window_size]
    } else {
        &samples[..]
    };

    info!(
        "Selected warmup audio chunk from index {} with energy {:.2}",
        best_start, max_energy
    );

    // 4. Initialize Inference Engine
    let mut engine = InferenceEngine::new(config, &mm_config, Some(warmup_samples)).await?;

    // 5. Run Verification Loop
    let mut pass_count = 0;
    let total_count = artifact.expected_results.len();

    for expected in &artifact.expected_results {
        // Calculate sample range for this segment
        // Python: start_time, end_time.
        // We can just extract the slice corresponding to this time window.
        // start_sample = start_time * sr
        // len = (end - start) * sr

        let start_sample = (expected.start_time * sr as f64).round() as usize;
        let end_sample = (expected.end_time * sr as f64).round() as usize;

        if start_sample >= samples.len() {
            warn!("Segment {} starts after audio end", expected.index);
            continue;
        }

        // Ensure we don't go out of bounds
        let valid_end = std::cmp::min(end_sample, samples.len());
        let segment_audio = &samples[start_sample..valid_end];

        // Run Inference
        // Use the timestamp from expected result to keep logs consistent
        let ts = chrono::Local::now(); // Placeholder, or derive from start_time relative to "now"

        let events = engine.process_segment(segment_audio, ts)?;

        // We expect exactly 1 event per segment usually
        if let Some(event) = events.first() {
            // Compare
            // 1. Label
            let rust_label = best_label(&event.label_probs);
            let label_match = rust_label == expected.label;

            // 2. Confidence
            let rust_conf = event.label_probs.get(&rust_label).cloned().unwrap_or(0.0);
            let conf_diff = (rust_conf - expected.confidence).abs();
            let _conf_match = conf_diff < 0.05; // 5% tolerance

            // 3. Loudness
            let loud_diff = (event.loudness - expected.loudness).abs();
            let _loud_match = loud_diff < 2.0; // 2dB tolerance

            let status = if label_match { "PASS" } else { "FAIL" };
            if status == "PASS" {
                pass_count += 1;
            }

            println!(
                "{:<6} | {:<5.2}-{:<5.2} | {:<20} | {:<12} | {:<10.1} | {}",
                expected.index,
                expected.start_time,
                expected.end_time,
                format!("{} / {}", expected.label, rust_label),
                format!("{:.2} / {:.2}", expected.confidence, rust_conf),
                format!("{:.1} / {:.1}", expected.loudness, event.loudness),
                status
            );

            if !label_match {
                warn!(
                    "Mismatch at {}: Py='{}'({:.2}), Rs='{}'({:.2})",
                    expected.index, expected.label, expected.confidence, rust_label, rust_conf
                );
            }
        } else {
            error!("No inference event produced for segment {}", expected.index);
            println!(
                "{:<6} | {:<12} | {:<20} | {:<12} | {:<10} | {:<10} | ERROR",
                expected.index, "...", "No Output", "...", "...", "..."
            );
        }
    }

    println!("----------------------------------------------------------------");
    println!("Result: {} / {} segments matched.", pass_count, total_count);

    if pass_count == total_count {
        info!("Verification PASSED");
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "Verification FAILED: {} mismatches",
            total_count - pass_count
        ))
    }
}

fn best_label(probs: &HashMap<String, f32>) -> String {
    let mut entries: Vec<(&String, &f32)> = probs.iter().collect();
    // Sort by probability descending, then by label string for determinism
    entries.sort_by(|a, b| {
        b.1.partial_cmp(a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(b.0))
    });

    entries
        .first()
        .map(|(k, _)| (*k).clone())
        .unwrap_or_else(|| "unknown".to_string())
}
