pub mod audio;
pub mod clip_recorder;
pub mod config;
pub mod inference;
pub mod metamodel;
pub mod runner;
pub mod verification_artifact;

pub use audio::get_audio_devices;
pub use config::{CliArgs, Commands, RuntimeConfig};
pub use inference::InferenceEngine;
pub use metamodel::MetaModelDecider;
pub use runner::run_inference;
