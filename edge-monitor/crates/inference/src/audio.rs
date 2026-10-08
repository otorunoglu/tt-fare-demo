use crate::config::RuntimeConfig;
use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use log::{error, info, warn};
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use tokio::sync::mpsc::Sender;

use audio_preprocessing::resample::create_resampler;
/// Spawns a new thread to capture audio, either from a file (simulation) or the default input device.
///
/// # Arguments
/// * `config` - Runtime configuration containing audio settings and paths.
/// * `tx` - Channel sender to dispatch audio chunks to the main processing loop.
/// * `running` - Atomic flag to control the stream loop; stopping it cleanly.
pub fn start_audio_stream(
    config: RuntimeConfig,
    tx: Sender<(chrono::DateTime<chrono::Local>, Vec<f32>)>,
    running: Arc<AtomicBool>,
) -> Result<std::thread::JoinHandle<anyhow::Result<()>>> {
    let handle = std::thread::spawn(move || -> anyhow::Result<()> {
        if let Some(file_path) = &config.simulate_file {
            info!("Simulating audio from file: {:?}", file_path);
            stream_from_file(file_path, &config, tx, running).map_err(|e| {
                error!("File simulation error: {}", e);
                e
            })
        } else {
            // Realtime CPAL
            if config.audio_device.is_none() {
                info!("Listing available audio fallback devices...");
                if let Err(e) = list_audio_devices() {
                    warn!("Failed to list audio devices: {}", e);
                }
            }

            stream_from_device(config, tx, running).map_err(|e| {
                error!("Audio device error: {}", e);
                e
            })
        }
    });
    Ok(handle)
}

/// Helper to resolve human-readable names for ALSA devices on Linux.
/// Maps things like "hw:CARD=II" to "RODE VideoMic GO II" by reading /proc/asound/cards.
fn resolve_alsa_name(alsa_name: &str) -> String {
    #[cfg(target_os = "linux")]
    {
        // Try to extract the card ID. Look for "CARD=ID"
        let card_id = if let Some(start) = alsa_name.find("CARD=") {
            let rest = &alsa_name[start + 5..];
            let end = rest
                .find(',')
                .or_else(|| rest.find(' '))
                .unwrap_or(rest.len());
            Some(&rest[..end])
        } else {
            None
        };

        if let Some(id) = card_id {
            // Read /proc/asound/cards to find the mapping from ID to Name
            if let Ok(content) = std::fs::read_to_string("/proc/asound/cards") {
                let mut lines = content.lines().peekable();
                while let Some(line) = lines.next() {
                    // ALSA IDs in /proc/asound/cards are often padded: " 1 [II             ]: ..."
                    if let (Some(b_start), Some(b_end)) = (line.find('['), line.find(']')) {
                        let inner = &line[b_start + 1..b_end].trim();
                        if inner == &id {
                            let short_name = line.split("]: ").nth(1).unwrap_or("").trim().to_string();
                            
                            let mut long_name = String::new();
                            if let Some(next_line) = lines.peek() {
                                let trimmed = next_line.trim();
                                if let Some(at_idx) = trimmed.find(" at ") {
                                    long_name = trimmed[..at_idx].to_string();
                                } else {
                                    long_name = trimmed.to_string();
                                }
                            }
                            
                            let best_name = if !long_name.is_empty() {
                                long_name
                            } else if !short_name.is_empty() {
                                short_name
                            } else {
                                alsa_name.to_string()
                            };
                            
                            // Remove generic USB-Audio prefix to reveal true device string if present
                            let clean_name = best_name.replace("USB-Audio - ", "");
                            
                            return format!("{} [{}]", clean_name, alsa_name);
                        }
                    }
                }
            }
        }
    }
    alsa_name.to_string()
}

/// Helper to filter out unhelpful duplicate ALSA pseudo-devices from the output.
/// Returns true if the device name seems like a primary useful device (e.g. hw, plughw, pulse, default).
fn is_useful_alsa_device(name: &str) -> bool {
    #[cfg(target_os = "linux")]
    {
        let lower = name.to_lowercase();
        // Skip common ALSA surround/specialty pseudo-devices that just duplicate the main hardware
        if lower.starts_with("front:")
            || lower.starts_with("surround")
            || lower.starts_with("iec958:")
            || lower.starts_with("dsnoop:")
            || lower.starts_with("dmix:")
            || lower.starts_with("sysdefault:")
            || lower.starts_with("default:")
        {
            return false;
        }
    }
    true
}

/// Lists all available audio input devices, summarizing their capture capabilities.
pub fn list_audio_devices() -> Result<()> {
    let host = cpal::default_host();
    let devices = host.input_devices()?;

    info!("Available Audio Input Devices:");
    let mut display_idx = 0;
    for device in devices {
        let raw_name = device
            .name()
            .unwrap_or_else(|_| "Unknown Device".to_string());
            
        if !is_useful_alsa_device(&raw_name) {
            continue;
        }
            
        let name = resolve_alsa_name(&raw_name);

        match device.supported_input_configs() {
            Ok(configs) => {
                let configs: Vec<_> = configs.collect();
                if configs.is_empty() {
                    info!("  [{}] (Unsupported): {}", display_idx, name);
                } else {
                    let min_rate = configs
                        .iter()
                        .map(|c| c.min_sample_rate().0)
                        .min()
                        .unwrap_or(0);
                    let max_rate = configs
                        .iter()
                        .map(|c| c.max_sample_rate().0)
                        .max()
                        .unwrap_or(0);
                    let mut channels: Vec<_> = configs.iter().map(|c| c.channels()).collect();
                    channels.sort();
                    channels.dedup();

                    info!(
                        "  [{}]: {} — CAPTURE OK (Rates: {}-{}Hz, Channels: {:?})",
                        display_idx, name, min_rate, max_rate, channels
                    );
                }
            }
            Err(_) => {
                info!("  [{}] (Unsupported): {}", display_idx, name);
            }
        }
        display_idx += 1;
    }
    Ok(())
}

pub fn get_audio_devices() -> Result<Vec<String>> {
    let host = cpal::default_host();
    let devices = host.input_devices()?;
    let mut device_names = Vec::new();

    for device in devices {
        if let Ok(raw_name) = device.name() {
            if !is_useful_alsa_device(&raw_name) {
                continue;
            }
            
            // Only include devices that actually support capture
            if device
                .supported_input_configs()
                .map(|mut c| c.next().is_some())
                .unwrap_or(false)
            {
                let human_name = resolve_alsa_name(&raw_name);
                device_names.push(human_name);
            }
        }
    }
    Ok(device_names)
}

/// Simulates real-time audio input by reading from a file.
/// Resamples the file audio to the target sample rate and sends chunks at the correct real-time pace.
fn stream_from_file(
    path: &Path,
    config: &RuntimeConfig,
    tx: Sender<(chrono::DateTime<chrono::Local>, Vec<f32>)>,
    running: Arc<AtomicBool>,
) -> Result<()> {
    let src = File::open(path)?;
    warn!("Initializing stream for file: {:?}", path);
    // Explicit config load removed, relying on log side-effect if any, or just proper logging.
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    let hint = Hint::new();

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;
    let mut format = probed.format;

    let (track_id, source_rate, channels) = {
        let track = format
            .default_track()
            .ok_or_else(|| anyhow::anyhow!("No track found"))?;
        (
            track.id,
            track.codec_params.sample_rate.unwrap_or(config.sample_rate),
            track.codec_params.channels.map(|c| c.count()).unwrap_or(1),
        )
    };

    let dec_opts: DecoderOptions = Default::default();
    let mut decoder = {
        let track = format.tracks().iter().find(|t| t.id == track_id).unwrap();
        symphonia::default::get_codecs().make(&track.codec_params, &dec_opts)?
    };

    let target_rate = config.sample_rate;
    let start_time = chrono::Local::now();
    let mut samples_processed: u64 = 0;

    let mut processor = create_resampler(source_rate, target_rate, None)?;
    let mut chunk_buffer: Vec<f32> = Vec::new();

    info!("File: {:?}, Rate: {} -> {}", path, source_rate, target_rate);

    while running.load(Ordering::SeqCst) {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(_)) => break,
            Err(e) => return Err(e.into()),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let frames = decoded.frames();
                let mut pcm = Vec::with_capacity(frames);

                match decoded {
                    // VERIFY NORMALIZATION: Standard float conversion
                    AudioBufferRef::F32(buf) => {
                        for i in 0..frames {
                            let mut sum = 0.0;
                            for c in 0..channels {
                                sum += buf.chan(c)[i];
                            }
                            pcm.push(sum / channels as f32);
                        }
                    }
                    AudioBufferRef::S32(buf) => {
                        for i in 0..frames {
                            let mut sum = 0.0;
                            for c in 0..channels {
                                sum += buf.chan(c)[i] as f32 / 2147483648.0;
                            }
                            pcm.push(sum / channels as f32);
                        }
                    }
                    AudioBufferRef::S24(buf) => {
                        for i in 0..frames {
                            let mut sum = 0.0;
                            for c in 0..channels {
                                sum += buf.chan(c)[i].0 as f32 / 8388608.0;
                            }
                            pcm.push(sum / channels as f32);
                        }
                    }
                    AudioBufferRef::S16(buf) => {
                        for i in 0..frames {
                            let mut sum = 0.0;
                            for c in 0..channels {
                                sum += buf.chan(c)[i] as f32 / 32768.0;
                            }
                            pcm.push(sum / channels as f32);
                        }
                    }
                    _ => continue,
                }

                let mut resampled = Vec::new();
                if let Err(e) = processor.process(&pcm, &mut resampled) {
                    error!("Resampling error: {}", e);
                    continue;
                }

                chunk_buffer.extend(resampled);

                while chunk_buffer.len() >= 1600 {
                    let chunk: Vec<f32> = chunk_buffer.drain(0..1600).collect();
                    let ts = start_time
                        + chrono::Duration::milliseconds(
                            (samples_processed as f64 / target_rate as f64 * 1000.0) as i64,
                        );

                    if tx.blocking_send((ts, chunk)).is_err() {
                        return Ok(());
                    }

                    samples_processed += 1600;
                }
            }
            Err(_) => break,
        }
    }
    Ok(())
}

/// Captures audio from the system's default input device using CPAL.
/// Handles different sample formats (F32, I16) and resamples to the target rate if necessary.
/// When no AUDIO_DEVICE is configured, tries the default input device first,
/// then falls back to the first device that actually supports capture.
fn stream_from_device(
    config: RuntimeConfig,
    tx: Sender<(chrono::DateTime<chrono::Local>, Vec<f32>)>,
    running: Arc<AtomicBool>,
) -> Result<()> {
    let host = cpal::default_host();
    let device = if let Some(ref name) = config.audio_device {
        let name_lower = name.to_lowercase();
        
        // Find all devices whose raw or friendly name contains the search string
        let mut matched_devices = Vec::new();
        for d in host.input_devices()? {
            let raw_name = d.name().unwrap_or_default();
            let friendly_name = resolve_alsa_name(&raw_name);
            
            if raw_name.to_lowercase().contains(&name_lower) || friendly_name.to_lowercase().contains(&name_lower) {
                // Score: prefer "hw:" (raw hardware, no redundant ALSA resampling since we
                // already do software resampling), then "plughw:", then anything else.
                let raw_lower = raw_name.to_lowercase();
                let score = if raw_lower.starts_with("hw:") {
                    3
                } else if raw_lower.starts_with("plughw:") {
                    2
                } else {
                    1
                };
                matched_devices.push((score, d));
            }
        }
        
        // Pick the device with the highest score
        let best = matched_devices.into_iter()
            .max_by_key(|(score, _d)| *score)
            .map(|(_score, d)| d)
            .ok_or_else(|| anyhow::anyhow!("Device not found: {}", name))?;

        info!("Selected audio device: {}", best.name().unwrap_or_default());
        best
    } else {
        // Try the default input device first
        let default = host.default_input_device();
        let use_default = default.as_ref().is_some_and(|d| {
            d.supported_input_configs()
                .map(|mut c| c.next().is_some())
                .unwrap_or(false)
        });
        if use_default {
            info!("Using default input device");
            default.unwrap()
        } else {
            // Default device doesn't support capture; find the first one that does
            warn!("Default input device does not support capture, searching for a capture-capable device...");
            let fallback = host.input_devices()?.find(|d| {
                d.supported_input_configs()
                    .map(|mut c| c.next().is_some())
                    .unwrap_or(false)
            });
            match fallback {
                Some(d) => {
                    let fallback_name = d.name().unwrap_or_default();
                    info!("Auto-selected fallback capture device: {}", fallback_name);
                    info!("HINT: To lock this device, set AUDIO_DEVICE=\"{}\" in your .env file!", resolve_alsa_name(&fallback_name));
                    d
                }
                None => return Err(anyhow::anyhow!(
                    "No capture-capable audio device found. Available devices listed above. \
                     Set AUDIO_DEVICE in .env to one of the capture-capable devices (e.g. 'plughw:CARD=II,DEV=0')."
                )),
            }
        }
    };

    let target_rate = config.sample_rate;
    
    // Prefer to use the device's native default input config to avoid CPAL ALSA "Invalid argument" errors
    // when forcing unsupported arbitrary sample rates. The internal resampler handles the rate conversion natively.
    let supported_config = device
        .default_input_config()
        .or_else(|_| {
            // Fallback: Use the very first supported configuration if no default is specified
            match device.supported_input_configs() {
                Ok(mut c) => c
                    .next()
                    .map(|range| range.with_max_sample_rate())
                    .ok_or(cpal::DefaultStreamConfigError::DeviceNotAvailable),
                Err(_) => Err(cpal::DefaultStreamConfigError::DeviceNotAvailable),
            }
        })
        .map_err(|e| anyhow::anyhow!("No supported config found: {}", e))?;

    let sample_format = supported_config.sample_format();
    let config_channels = supported_config.channels() as usize;
    let config_rate = supported_config.sample_rate().0;

    info!(
        "Device: {}, Rate: {}, Channels: {}, Format: {:?}",
        device.name().unwrap_or_default(),
        config_rate,
        config_channels,
        sample_format
    );

    let processor = create_resampler(config_rate, target_rate, None)?;

    let tx_clone = tx.clone();

    let processor_shared = Arc::new(Mutex::new(processor));

    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            let proc = processor_shared.clone();
            device.build_input_stream(
                &supported_config.into(),
                move |data: &[f32], _: &_| {
                    let ts = chrono::Local::now();
                    let mut mono = Vec::with_capacity(data.len() / config_channels);
                    for frame in data.chunks_exact(config_channels) {
                        mono.push(frame[0]);
                    }

                    let mut out = Vec::new();
                    if let Ok(mut p) = proc.lock() {
                        if let Err(e) = p.process(&mono, &mut out) {
                            error!("Resample error: {}", e);
                        }
                    }

                    if !out.is_empty() {
                        let _ = tx_clone.try_send((ts, out));
                    }
                },
                |_| {},
                None,
            )
        }
        cpal::SampleFormat::I16 => {
            let proc = processor_shared.clone();
            device.build_input_stream(
                &supported_config.into(),
                move |data: &[i16], _: &_| {
                    let ts = chrono::Local::now();
                    let mut mono = Vec::with_capacity(data.len() / config_channels);
                    for frame in data.chunks_exact(config_channels) {
                        // Normalization for I16 device input
                        mono.push(frame[0] as f32 / 32768.0);
                    }
                    let mut out = Vec::new();
                    if let Ok(mut p) = proc.lock() {
                        if let Err(e) = p.process(&mono, &mut out) {
                            error!("Resample error: {}", e);
                        }
                    }
                    if !out.is_empty() {
                        let _ = tx_clone.try_send((ts, out));
                    }
                },
                |_| {},
                None,
            )
        }
        _ => return Err(anyhow::anyhow!("Unsupported format: {:?}", sample_format)),
    }?;

    stream.play()?;
    while running.load(Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    
    // Explicit drop handle to stream to prevent ALSA invalid handle bugs at closure
    drop(stream);
    
    Ok(())
}
