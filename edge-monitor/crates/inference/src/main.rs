use inference::{
    config::{Commands, RuntimeConfig},
    run_inference,
};
use log::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    env_logger::Builder::from_default_env()
        .filter_module("ort", log::LevelFilter::Warn)
        .init();

    info!("Starting Horse Stable Monitor (Rust)...");

    // Initialize ORT environment globally
    let _ = ort::init().with_name("stable-monitor").commit();

    // Initialize global configuration (merges CLI and Config file)
    let config = RuntimeConfig::initialize();

    match &config.command {
        Some(Commands::VerifyArtifact { file }) => {
            inference::verification_artifact::run_verification(file, &config).await
        }
        _ => run_inference(config, None, None).await,
    }
}
