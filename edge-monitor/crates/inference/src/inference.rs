use anyhow::{Context, Result};
use log::info;
use ndarray::ArrayView2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

use crate::config::RuntimeConfig;
use shared::InferenceEvent;

pub struct InferenceEngine {
    embedder_session: Session,
    classifier_session: Session,
    label_mapping: HashMap<usize, String>,
    silence_threshold: f32,
    classifier_threshold: f32,
}

impl InferenceEngine {
    pub async fn new(
        config: &RuntimeConfig,
        mm_config: &crate::config::SmartStableModelConfig,
        _sanity_check_audio: Option<&[f32]>,
    ) -> Result<Self> {
        // Initialize ORT environment should be done in main.rs

        let embedder_path = &config.embedder_path;
        info!("Loading embedder from {:?}", embedder_path);

        let session_builder =
            Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;

        let embedder_session = session_builder.commit_from_file(embedder_path)?;

        let classifier_path = &config.classifier_path;
        info!("Loading classifier from {:?}", classifier_path);
        let classifier_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Disable)?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_file(classifier_path)?;
        println!("Loading label mapping");
        let label_mapping_path = &config.label_mapping_path;
        info!("Loading label mapping from {:?}", label_mapping_path);
        let file = File::open(label_mapping_path)
            .with_context(|| format!("Failed to open label mapping at {:?}", label_mapping_path))?;
        let reader = BufReader::new(file);

        let mapping_json: serde_json::Value = serde_json::from_reader(reader)?;
        let mut label_mapping = HashMap::new();

        if let Some(obj) = mapping_json.get("idx_to_label") {
            if let Some(map) = obj.as_object() {
                for (k, v) in map {
                    if let Ok(idx) = k.parse::<usize>() {
                        if let Some(s) = v.as_str() {
                            label_mapping.insert(idx, s.to_string());
                        }
                    }
                }
            }
        }

        let silence_threshold = mm_config
            .model_parameters
            .get("silence_threshold")
            .and_then(|v| v.as_float())
            .map(|f| f as f32)
            .unwrap_or(-60.0);

        // Get classifier threshold (default to 0.9 for PANNs if not specified)
        // We assume "panns" architecture for now as per Python default
        // TODO: Read model architecture from config if variable
        let classifier_threshold = mm_config
            .model_parameters
            .get("classifier_thresholds")
            .and_then(|v| v.as_table())
            .and_then(|t| t.get("panns")) // Defaulting to panns
            .and_then(|v| v.as_float())
            .map(|f| f as f32)
            .unwrap_or(0.9);

        info!(
            "Inference initialized. Silence threshold: {:.1} dB, Classifier threshold: {:.2}",
            silence_threshold, classifier_threshold
        );

        Ok(Self {
            embedder_session,
            classifier_session,
            label_mapping,
            silence_threshold,
            classifier_threshold,
        })
    }

    pub fn process_segment(
        &mut self,
        audio: &[f32],
        ts: chrono::DateTime<chrono::Local>,
    ) -> Result<Vec<InferenceEvent>> {
        // 1. Compute Loudness (dBFS)
        let db_fs = calculate_loudness_dbfs(audio);

        // 2. Silence Gate
        if db_fs < self.silence_threshold {
            // Return "silence" event
            return Ok(vec![InferenceEvent {
                timestamp: ts,
                label_probs: HashMap::from([("silence".to_string(), 1.0)]),
                loudness: db_fs,
                source: "live".to_string(),
                replay_tag: None,
                embedding: None,
            }]);
        }

        // Diagnostic: Audio Sanity
        if audio.iter().any(|&x| !x.is_finite()) {
            log::error!("Audio contains non-finite values (NaN/Inf)!");
        }
        if audio.len() >= 5 {
            log::debug!("AUDIO DATA [0..5]: {:?}", &audio[0..5]);
        }

        let audio_sum: f32 = audio.iter().sum();
        let audio_sq_sum: f32 = audio.iter().map(|&x| x * x).sum();

        // Create input tensor using tuple for proper shape communication (bypasses ndarray trait issues)
        // Clamp to [-1, 1] to ensure input sanity
        let audio_vec: Vec<f32> = audio.iter().map(|&x| x.clamp(-1.0, 1.0)).collect();
        let input_tensor = Value::from_array(([1usize, audio_vec.len()], audio_vec))?;

        // Run embedder
        let outputs_embed = self
            .embedder_session
            .run(ort::inputs!["input_audio" => &input_tensor])?;

        // Extract "embeddings": [1, 2048]
        let (embed_shape, embed_data) = outputs_embed["embeddings"].try_extract_tensor::<f32>()?;

        // Diagnostic: Embedding Sanity
        if embed_data.iter().any(|&x| !x.is_finite()) {
            log::error!("Embedder produced non-finite values in embedding!");
        }

        let embed_sq_sum: f32 = embed_data.iter().map(|&x| x * x).sum();
        let embed_norm = embed_sq_sum.sqrt();

        // --- Step 2: Classifier ---
        // We need to create a new tensor for the classifier input
        let batch_size = embed_shape[0] as usize;
        let num_features = embed_shape[1] as usize;
        let embed_vec = embed_data.to_vec();

        let classifier_input = Value::from_array(([batch_size, num_features], embed_vec.clone()))?;

        // Run classifier
        let outputs_class = self
            .classifier_session
            .run(ort::inputs!["embeddings" => &classifier_input])?;

        // Extract Probabilities (output 0)
        let (out_shape, out_data) = outputs_class[0].try_extract_tensor::<f32>()?;

        let batch_size = out_shape[0] as usize;
        let num_classes = if out_shape.len() > 1 {
            out_shape[1] as usize
        } else {
            1
        };

        // Create ArrayView to slice easily
        let out_view = ArrayView2::from_shape((batch_size, num_classes), out_data)?;

        // Get prob vector (batch 0)
        let probs_slice = out_view.index_axis(ndarray::Axis(0), 0);

        // Map to labels
        let mut label_probs = HashMap::new();
        let mut top_label = "unknown".to_string();
        let mut top_p = -1.0;

        for (idx, &prob) in probs_slice.iter().enumerate() {
            let label = self
                .label_mapping
                .get(&idx)
                .cloned()
                .unwrap_or_else(|| format!("class_{}", idx));
            if prob > top_p {
                top_p = prob;
                top_label = label.clone();
            }
            label_probs.insert(label, prob);
        }

        // Apply low-confidence fallback label.
        // If the top prediction confidence is below threshold, force "uncertain".
        if top_p < self.classifier_threshold {
            log::debug!(
                "Top label '{}' ({:.4}) below threshold {:.2}. Forcing 'uncertain'.",
                top_label,
                top_p,
                self.classifier_threshold
            );

            // "Uncertain" is a confidence flag label.
            // We set it to 1.0 - prob of predicted class.
            // This matches Python: `accepted_label = "uncertain"`.

            // Rebuild label_probs to reflect this override
            let uncertain_prob = 1.0 - top_p; // lower top confidence means higher uncertainty
            label_probs.clear();
            label_probs.insert("uncertain".to_string(), uncertain_prob);

            // Update top_label/top_p for logging below (optional, but good for consistency)
            top_label = "uncertain".to_string();
            top_p = uncertain_prob;
        }

        // Log diagnostics
        log::debug!(
            "T={:?}, audio_sum={:.4}, audio_rms={:.4}, embed_norm={:.4}, top_label={}, p={:.4}",
            ts,
            audio_sum,
            (audio_sq_sum / audio.len() as f32).sqrt(),
            embed_norm,
            top_label,
            top_p
        );

        if embed_vec.iter().any(|v| !v.is_finite()) {
            log::error!("Embedder produced non-finite values in embedding (check after clone)!");
        }

        Ok(vec![InferenceEvent {
            timestamp: ts,
            label_probs,
            loudness: db_fs,
            source: "live".to_string(),
            replay_tag: None,
            embedding: Some(embed_vec),
        }])
    }
}

fn calculate_loudness_dbfs(audio: &[f32]) -> f32 {
    if audio.is_empty() {
        return -100.0;
    }
    let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
    if rms > 1e-9 {
        (20.0 * rms.log10()).max(-100.0)
    } else {
        -100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loudness_calculation() {
        // Test 1: Full scale sine wave (approx -3 dBFS for RMS of sine)
        // Amplitude 1.0 sine wave has RMS of 1/sqrt(2) = 0.707
        // 20 * log10(0.707) = -3.01 dB
        let sine: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let loud = calculate_loudness_dbfs(&sine);
        // It won't be exactly -3.01 due to sampling but close
        assert!(
            loud > -4.0 && loud < -2.0,
            "Sine wave loudness {} should be approx -3dB",
            loud
        );

        // Test 2: Silence
        let silence = vec![0.0; 100];
        let loud_silence = calculate_loudness_dbfs(&silence);
        assert_eq!(loud_silence, -100.0);

        // Test 3: Specific small value
        // RMS = 0.01 -> 20 * log10(0.01) = -40.0
        let small: Vec<f32> = vec![0.01; 100];
        let loud_small = calculate_loudness_dbfs(&small);
        assert!(
            (loud_small - -40.0).abs() < 0.1,
            "0.01 amplitude should be -40dB, got {}",
            loud_small
        );
    }
}
