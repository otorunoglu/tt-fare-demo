use anyhow::{Context, Result};
use log::{error, warn};
use std::fs::File;
use std::path::Path;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::resample::create_resampler;

/// Loads an audio file, mixes to mono, resamples to `target_rate`, and returns the waveform.
///
/// # Arguments
/// * `path` - Path to the audio file
/// * `target_rate` - Desired sample rate (e.g. 16000)
///
/// # Returns
/// * `(Vec<f32>, u32)` - Tuple containing the audio data and the sample rate (which should match target_rate)
pub fn load_audio_file<P: AsRef<Path>>(path: P, target_rate: u32) -> Result<(Vec<f32>, u32)> {
    let path = path.as_ref();
    let src = File::open(path).with_context(|| format!("Failed to open audio file: {:?}", path))?;

    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    let hint = Hint::new();

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)
        .context("Unsupported format")?;

    let mut format = probed.format;

    let track = format.default_track().context("No default track")?;
    let track_id = track.id;
    let source_rate = track.codec_params.sample_rate.unwrap_or(target_rate);
    let _channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);

    let dec_opts: DecoderOptions = Default::default();
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &dec_opts)?;

    let mut processor = create_resampler(source_rate, target_rate, None)?;
    let mut output_buffer = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(_)) => break, // EOF
            Err(e) => return Err(e.into()),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let frames = decoded.frames();
                let mut pcm = Vec::with_capacity(frames);
                let channels = decoded.spec().channels.count();

                // Mix to mono and normalize to f32 [-1.0, 1.0]
                match decoded {
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
                    _ => warn!("Unsupported sample format, skipping packet"),
                }

                // Resample
                let mut resampled_chunk = Vec::new();
                if let Err(e) = processor.process(&pcm, &mut resampled_chunk) {
                    error!("Resampling error: {}", e);
                    continue;
                }
                output_buffer.extend(resampled_chunk);
            }
            Err(e) => {
                warn!("Decode error: {}", e);
                continue;
            }
        }
    }

    Ok((output_buffer, target_rate))
}
