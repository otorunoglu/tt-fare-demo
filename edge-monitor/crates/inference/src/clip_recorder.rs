//! Rolling-window alert clip recorder.
//!
//! Keeps the most recent `pre_seconds` of captured audio in a ring buffer. When a
//! qualifying [`DomainWarning`] fires, it snapshots that pre-roll and continues
//! collecting `post_seconds` of audio, then writes the combined clip to disk as a
//! 16-bit PCM WAV file.
//!
//! All audio in the runner flows through a single async loop, so this recorder is
//! intentionally single-threaded and lock-free: the loop calls [`ClipRecorder::push_audio`]
//! for every incoming chunk and [`ClipRecorder::on_warning`] whenever a warning is decided.

use crate::config::RecordingConfig;
use log::{error, info, warn};
use shared::DomainWarning;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

/// An in-progress clip capture: the pre-roll has been snapshotted and we are
/// collecting the post-roll tail.
struct Pending {
    /// Accumulating samples: pre-roll snapshot followed by post-roll audio.
    samples: Vec<f32>,
    /// How many more post-roll samples still need to be collected before flushing.
    remaining_post: usize,
    label: String,
    warning_type: String,
    timestamp: chrono::DateTime<chrono::Local>,
}

pub struct ClipRecorder {
    sample_rate: u32,
    pre_samples: usize,
    post_samples: usize,
    severity_threshold: f32,
    output_dir: PathBuf,
    /// Maximum number of clips to keep on disk. 0 = unlimited.
    max_clips: usize,
    /// Delete clips older than this many days. 0 = never delete by age.
    max_age_days: f32,
    /// Rolling ring buffer holding at most `pre_samples` of the most recent audio.
    ring: VecDeque<f32>,
    /// Set while a clip is being captured.
    pending: Option<Pending>,
}

impl ClipRecorder {
    /// Build a recorder from config. Returns `None` when the feature is disabled,
    /// so callers can keep an `Option<ClipRecorder>` and skip work cheaply.
    pub fn from_config(cfg: &RecordingConfig, sample_rate: u32) -> Option<Self> {
        if !cfg.enabled {
            return None;
        }

        let pre_samples = (cfg.pre_seconds.max(0.0) * sample_rate as f32) as usize;
        let post_samples = (cfg.post_seconds.max(0.0) * sample_rate as f32) as usize;
        let output_dir = PathBuf::from(&cfg.output_dir);

        if let Err(e) = std::fs::create_dir_all(&output_dir) {
            error!(
                "Alert clip recording disabled: could not create output dir {:?}: {}",
                output_dir, e
            );
            return None;
        }

        info!(
            "Alert clip recording enabled: pre={:.1}s post={:.1}s severity>={:.2} dir={:?}",
            cfg.pre_seconds, cfg.post_seconds, cfg.severity_threshold, output_dir
        );

        Some(Self {
            sample_rate,
            pre_samples,
            post_samples,
            severity_threshold: cfg.severity_threshold,
            output_dir,
            max_clips: cfg.max_clips,
            max_age_days: cfg.max_age_days,
            ring: VecDeque::with_capacity(pre_samples + 1),
            pending: None,
        })
    }

    /// Feed a freshly captured audio chunk. Maintains the pre-roll ring buffer and,
    /// if a clip is being captured, appends to it and flushes once the post-roll is full.
    pub fn push_audio(&mut self, chunk: &[f32]) {
        // Maintain the rolling pre-roll window.
        if self.pre_samples > 0 {
            self.ring.extend(chunk.iter().copied());
            while self.ring.len() > self.pre_samples {
                self.ring.pop_front();
            }
        }

        // If we are capturing a clip, collect the post-roll tail.
        if let Some(pending) = self.pending.as_mut() {
            if pending.remaining_post > 0 {
                let take = pending.remaining_post.min(chunk.len());
                pending.samples.extend_from_slice(&chunk[..take]);
                pending.remaining_post -= take;
            }
            if pending.remaining_post == 0 {
                self.flush();
            }
        }
    }

    /// React to a decided warning. Starts (or extends) a clip capture when the
    /// warning's severity meets the configured threshold.
    pub fn on_warning(&mut self, warning: &DomainWarning) {
        if warning.severity < self.severity_threshold {
            return;
        }

        match self.pending.as_mut() {
            // Already capturing — refresh the post-roll so a burst of warnings merges
            // into a single clip that extends past the latest trigger.
            Some(pending) => {
                pending.remaining_post = self.post_samples;
            }
            // Start a new capture: snapshot the current pre-roll ring.
            None => {
                let mut samples: Vec<f32> =
                    Vec::with_capacity(self.ring.len() + self.post_samples);
                samples.extend(self.ring.iter().copied());
                self.pending = Some(Pending {
                    samples,
                    remaining_post: self.post_samples,
                    label: warning.label.clone(),
                    warning_type: format!("{:?}", warning.warning_type),
                    timestamp: warning.timestamp,
                });

                // With no post-roll configured, write the pre-roll immediately.
                if self.post_samples == 0 {
                    self.flush();
                }
            }
        }
    }

    /// Write any in-progress capture to disk immediately (e.g. on shutdown).
    pub fn finish(&mut self) {
        if self.pending.is_some() {
            self.flush();
        }
    }

    fn flush(&mut self) {
        let Some(pending) = self.pending.take() else {
            return;
        };

        let file_name = format!(
            "{}_{}_{}.wav",
            pending.timestamp.format("%Y%m%d_%H%M%S"),
            pending.warning_type,
            sanitize_label(&pending.label),
        );
        let path = self.output_dir.join(file_name);

        match write_wav_i16(&path, &pending.samples, self.sample_rate) {
            Ok(()) => info!(
                "Saved alert clip {:?} ({:.1}s, {} samples)",
                path,
                pending.samples.len() as f32 / self.sample_rate as f32,
                pending.samples.len()
            ),
            Err(e) => error!("Failed to write alert clip {:?}: {}", path, e),
        }

        self.prune();
    }

    /// Enforce the retention policy: delete clips older than `max_age_days`, then
    /// delete the oldest clips until at most `max_clips` remain. Failures are logged
    /// but never abort recording.
    fn prune(&self) {
        if self.max_clips == 0 && self.max_age_days <= 0.0 {
            return;
        }

        // Gather (path, modified-time) for all .wav files in the output dir.
        let mut clips: Vec<(PathBuf, std::time::SystemTime)> = Vec::new();
        let entries = match std::fs::read_dir(&self.output_dir) {
            Ok(entries) => entries,
            Err(e) => {
                warn!("Clip pruning skipped, cannot read {:?}: {}", self.output_dir, e);
                return;
            }
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("wav") {
                continue;
            }
            let modified = entry
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            clips.push((path, modified));
        }

        // Oldest first.
        clips.sort_by_key(|(_, modified)| *modified);

        // Age-based deletion.
        if self.max_age_days > 0.0 {
            let max_age = std::time::Duration::from_secs_f64(self.max_age_days as f64 * 86_400.0);
            let now = std::time::SystemTime::now();
            clips.retain(|(path, modified)| {
                let too_old = now
                    .duration_since(*modified)
                    .map(|age| age > max_age)
                    .unwrap_or(false);
                if too_old {
                    remove_clip(path);
                    false // drop from the list so it isn't counted below
                } else {
                    true
                }
            });
        }

        // Count-based deletion: remove oldest until within budget.
        if self.max_clips > 0 && clips.len() > self.max_clips {
            let excess = clips.len() - self.max_clips;
            for (path, _) in clips.iter().take(excess) {
                remove_clip(path);
            }
        }
    }
}

fn remove_clip(path: &Path) {
    match std::fs::remove_file(path) {
        Ok(()) => info!("Pruned old alert clip {:?}", path),
        Err(e) => warn!("Failed to prune alert clip {:?}: {}", path, e),
    }
}

/// Make a label safe to embed in a filename across platforms.
fn sanitize_label(label: &str) -> String {
    let cleaned: String = label
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let trimmed = cleaned.trim_matches('-');
    if trimmed.is_empty() {
        "unknown".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Write mono f32 samples (normalized to [-1.0, 1.0]) as a 16-bit PCM WAV file.
fn write_wav_i16(path: &Path, samples: &[f32], sample_rate: u32) -> anyhow::Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let value = (clamped * i16::MAX as f32).round() as i16;
        writer.write_sample(value)?;
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::WarningType;

    fn cfg(pre: f32, post: f32, threshold: f32, dir: &str) -> RecordingConfig {
        RecordingConfig {
            enabled: true,
            output_dir: dir.to_string(),
            pre_seconds: pre,
            post_seconds: post,
            severity_threshold: threshold,
            max_clips: 0,
            max_age_days: 0.0,
        }
    }

    fn warning(severity: f32) -> DomainWarning {
        DomainWarning {
            timestamp: chrono::Local::now(),
            label: "kick/bang".to_string(),
            warning_type: WarningType::ALERT,
            severity,
            confidence: 0.9,
        }
    }

    #[test]
    fn sanitize_label_replaces_unsafe_chars() {
        assert_eq!(sanitize_label("kick/bang"), "kick-bang");
        assert_eq!(sanitize_label("!!!"), "unknown");
        assert_eq!(sanitize_label("loud noise"), "loud-noise");
    }

    #[test]
    fn disabled_config_yields_none() {
        let mut c = cfg(1.0, 1.0, 1.0, "/tmp/does-not-matter");
        c.enabled = false;
        assert!(ClipRecorder::from_config(&c, 16000).is_none());
    }

    #[test]
    fn writes_clip_after_post_roll_fills() {
        let dir = std::env::temp_dir().join(format!("clip_test_{}", uuid::Uuid::new_v4()));
        // 100 samples/sec equivalent: use sample_rate 100 for tiny buffers.
        let sample_rate = 100;
        let mut rec = ClipRecorder::from_config(
            &cfg(1.0, 1.0, 1.0, dir.to_str().unwrap()),
            sample_rate,
        )
        .unwrap();

        // Fill more than the pre-roll window.
        rec.push_audio(&vec![0.5f32; 200]);
        rec.on_warning(&warning(2.0));
        // Not yet flushed: post-roll (100 samples) not collected.
        assert!(rec.pending.is_some());

        rec.push_audio(&vec![0.5f32; 50]);
        assert!(rec.pending.is_some());
        rec.push_audio(&vec![0.5f32; 60]); // now post-roll exceeded -> flush
        assert!(rec.pending.is_none());

        let files: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(files.len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn max_clips_prunes_oldest() {
        let dir = std::env::temp_dir().join(format!("clip_test_{}", uuid::Uuid::new_v4()));
        let mut c = cfg(1.0, 0.0, 1.0, dir.to_str().unwrap()); // post=0 -> flush immediately
        c.max_clips = 2;
        let mut rec = ClipRecorder::from_config(&c, 100).unwrap();

        // Produce 4 clips; only the 2 newest should survive.
        for _ in 0..4 {
            rec.push_audio(&vec![0.2f32; 100]);
            rec.on_warning(&warning(2.0));
            // Ensure distinct modified-times so pruning order is deterministic.
            std::thread::sleep(std::time::Duration::from_millis(1100));
        }

        let count = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).count();
        assert_eq!(count, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn below_threshold_does_not_trigger() {
        let dir = std::env::temp_dir().join(format!("clip_test_{}", uuid::Uuid::new_v4()));
        let mut rec =
            ClipRecorder::from_config(&cfg(1.0, 1.0, 1.5, dir.to_str().unwrap()), 100).unwrap();
        rec.push_audio(&vec![0.1f32; 100]);
        rec.on_warning(&warning(1.0)); // below 1.5
        assert!(rec.pending.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
