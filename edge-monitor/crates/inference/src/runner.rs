use crate::audio;
use crate::clip_recorder::ClipRecorder;
use crate::config::RuntimeConfig;
use crate::inference::InferenceEngine;
use crate::metamodel::MetaModelDecider;
use log::{error, info, warn};
use shared::{Database, DomainWarning};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub async fn run_inference(
    config: RuntimeConfig,
    external_running: Option<Arc<AtomicBool>>,
    warning_tx: Option<tokio::sync::mpsc::Sender<DomainWarning>>,
) -> anyhow::Result<()> {
    info!(
        "Configuration: Stable={}, Stall={}",
        config.stable_id, config.stall_id
    );

    // Load MetaModel config
    let mm_config = config.load_metamodel_config()?;

    let (audio_tx, mut audio_rx) = tokio::sync::mpsc::channel::<(
        chrono::DateTime<chrono::Local>,
        Vec<f32>,
    )>(config.max_queue_size);

    info!("Initializing Inference Engine...");
    let mut inference_engine = InferenceEngine::new(&config, &mm_config, None).await?;

    let mut decider = MetaModelDecider::new(mm_config);

    // Optional rolling-window clip recorder: saves audio around qualifying warnings.
    let mut clip_recorder = ClipRecorder::from_config(&config.edge_config.recording, config.sample_rate);

    let db_path = config.get_database_path();
    let db_url = format!("sqlite://{}", db_path.to_string_lossy());
    let db = Database::new(&db_url, false).await?;
    info!("Database initialized at {:?}", db_path);

    // Start audio streamer AFTER initialization is complete
    let streamer_running = Arc::new(AtomicBool::new(true));
    let mut streamer_handle = Some(audio::start_audio_stream(
        config.clone(),
        audio_tx,
        streamer_running.clone(),
    )?);

    let running = external_running.unwrap_or_else(|| {
        let r = Arc::new(AtomicBool::new(true));
        let rr = r.clone();
        // Only set handler if we are in standalone mode
        let _ = ctrlc::set_handler(move || {
            info!("\nReceived Ctrl-C, shutting down...");
            rr.store(false, Ordering::SeqCst);
        });
        r
    });

    let mut buffer: Vec<f32> = Vec::new();
    let event_source = config.event_source.clone();
    let replay_tag = config.replay_tag.clone();
    let stop_on_stream_end = config.stop_on_stream_end || config.simulate_file.is_some();
    let segment_samples = (config.sample_rate as f32 * config.segment_duration) as usize;
    let hop_samples = (config.sample_rate as f32
        * (config.segment_duration * (1.0 - config.segment_overlap)))
        as usize;

    info!(
        "Entering main loop. Req segment samples: {}, Hop: {}",
        segment_samples, hop_samples
    );

    // Track samples for precise timestamping
    let mut total_samples_processed: u64 = 0;
    let mut stream_start_ts: Option<chrono::DateTime<chrono::Local>> = None;

    while running.load(Ordering::SeqCst) {
        match tokio::time::timeout(Duration::from_millis(500), audio_rx.recv()).await {
            Ok(Some((chunk_ts, chunk))) => {
                // Initialize start time from the very first chunk received
                if stream_start_ts.is_none() {
                    stream_start_ts = Some(chunk_ts);
                    info!("First audio chunk received at {}", chunk_ts);
                }

                // Feed the rolling-window recorder before processing so the pre-roll
                // includes the audio that produced any warning decided below.
                if let Some(recorder) = clip_recorder.as_mut() {
                    recorder.push_audio(&chunk);
                }

                buffer.extend_from_slice(&chunk);

                while buffer.len() >= segment_samples {
                    let segment = &buffer[0..segment_samples];

                    // Calculate timestamp for the MIDDLE of the segment (matches Python/librosa tagging)
                    let center_offset_samples =
                        total_samples_processed + (segment_samples as u64 / 2);
                    let event_ts = stream_start_ts.unwrap()
                        + chrono::Duration::milliseconds(
                            (center_offset_samples as f64 / config.sample_rate as f64 * 1000.0)
                                as i64,
                        );

                    let start = std::time::Instant::now();
                    match inference_engine.process_segment(segment, event_ts) {
                        Ok(events) => {
                            for event in events.iter() {
                                let mut event_to_log = event.clone();
                                event_to_log.source = event_source.clone();
                                event_to_log.replay_tag = replay_tag.clone();

                                if let Err(e) = db.log_event(&event_to_log).await {
                                    error!("Failed to log event to DB: {}", e);
                                }
                                let (label, prob) = event_to_log
                                    .label_probs
                                    .iter()
                                    .max_by(|a, b| {
                                        a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)
                                    })
                                    .map(|(k, v)| (k.clone(), *v))
                                    .unwrap_or(("unknown".to_string(), 0.0));

                                if prob >= config.log_threshold && config.inference_logging {
                                    let json_out = serde_json::json!({
                                        "timestamp": event_to_log.timestamp.to_rfc3339(),
                                        "label": label,
                                        "confidence": prob,
                                        "loudness": event_to_log.loudness,
                                        "source": event_to_log.source,
                                        "replay_tag": event_to_log.replay_tag
                                    });
                                    println!("{}", json_out);
                                }

                                // Decide on Warnings
                                if let Some(warning) = decider.decide(&event_to_log) {
                                    // Save a rolling-window audio clip for qualifying warnings.
                                    if let Some(recorder) = clip_recorder.as_mut() {
                                        recorder.on_warning(&warning);
                                    }

                                    // Log warning to DB
                                    if let Err(e) = db.log_warning(&warning).await {
                                        error!("Failed to log warning to DB: {}", e);
                                    }

                                    // Forward to the notification worker if one is wired up
                                    if let Some(tx) = &warning_tx {
                                        if let Err(e) = tx.try_send(warning.clone()) {
                                            warn!("Warning channel full or closed, notification dropped: {}", e);
                                        }
                                    }

                                    warn!(
                                        "WARNING: {:?} - {} ({:.2})",
                                        warning.warning_type, warning.label, warning.confidence
                                    );

                                    let json_out = serde_json::json!({
                                        "timestamp": warning.timestamp.to_rfc3339(),
                                        "type": "warning",
                                        "warning_type": warning.warning_type,
                                        "label": warning.label,
                                        "confidence": warning.confidence,
                                        "severity": warning.severity
                                    });
                                    println!("{}", json_out);
                                }
                            }
                        }
                        Err(e) => {
                            error!("Inference failed: {}", e);
                        }
                    }
                    let inference_time = start.elapsed().as_millis();

                    if inference_time > (hop_samples as u128 / config.sample_rate as u128 * 1000) {
                        warn!(
                            "Inference took longer than hop_samples time: {} ms",
                            inference_time
                        );
                    }

                    // Drain hop_samples and update our sample tracker
                    if hop_samples < buffer.len() {
                        buffer.drain(0..hop_samples);
                    } else {
                        buffer.clear();
                    }
                    total_samples_processed += hop_samples as u64;
                }
            }
            Ok(None) => {
                // Channel closed — if we're still supposed to be running, the audio thread died
                if running.load(Ordering::SeqCst) {
                    if stop_on_stream_end {
                        info!("Audio stream ended naturally.");
                        streamer_running.store(false, Ordering::SeqCst);
                        if let Some(handle) = streamer_handle.take() {
                            match handle.join() {
                                Ok(Err(e)) => return Err(e.context("Audio stream failed")),
                                Err(_) => return Err(anyhow::anyhow!("Audio thread panicked")),
                                Ok(Ok(())) => {}
                            }
                        }
                        break;
                    }

                    error!("Audio channel closed unexpectedly — audio thread may have failed.");
                    // Join the audio thread to retrieve its error
                    streamer_running.store(false, Ordering::SeqCst);
                    if let Some(handle) = streamer_handle.take() {
                        match handle.join() {
                            Ok(Err(e)) => return Err(e.context("Audio stream failed")),
                            Ok(Ok(())) => {
                                return Err(anyhow::anyhow!(
                                    "Audio stream exited unexpectedly without error"
                                ))
                            }
                            Err(_) => return Err(anyhow::anyhow!("Audio thread panicked")),
                        }
                    }
                }
                info!("Audio channel closed.");
                break;
            }
            Err(_) => {
                // Timeout, continue loop to check running flag
                continue;
            }
        }
    }

    streamer_running.store(false, Ordering::SeqCst);

    // Flush any clip still collecting its post-roll so it isn't lost on shutdown.
    if let Some(recorder) = clip_recorder.as_mut() {
        recorder.finish();
    }

    std::mem::drop(audio_rx);

    if let Some(handle) = streamer_handle.take() {
        match handle.join() {
            Ok(Err(e)) => error!("Audio thread returned error: {}", e),
            Err(_) => error!("Audio thread panicked"),
            _ => {}
        }
    }
    info!("Shutdown complete.");

    Ok(())
}
