use anyhow::Result;
use log::info;
use rubato::{FftFixedIn, Resampler};
use std::f64::consts::PI;
use std::sync::Mutex;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ResampleError {
    #[error("Rubato error: {0}")]
    RubatoError(#[from] rubato::ResampleError),
    #[error("Rubato construction error: {0}")]
    RubatoConstructionError(#[from] rubato::ResamplerConstructionError),
    #[error("Resampling failed")]
    ProcessingError,
}

/// Abstract Resampler Trait
pub trait AudioResampler: Send {
    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) -> Result<()>;
}

/// Linear (Box Filter) Resampler - Fast, Low Quality
pub struct LinearResampler {
    ratio: f64,
    phase: f64,
    sum: f32,
    count: u32,
}

impl LinearResampler {
    pub fn new(source_rate: u32, target_rate: u32) -> Self {
        Self {
            ratio: source_rate as f64 / target_rate as f64,
            phase: 0.0,
            sum: 0.0,
            count: 0,
        }
    }
}

impl AudioResampler for LinearResampler {
    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) -> Result<()> {
        if (self.ratio - 1.0).abs() < 0.001 {
            output.extend_from_slice(input);
            return Ok(());
        }

        for &sample in input {
            self.sum += sample;
            self.count += 1;
            self.phase += 1.0;

            if self.phase >= self.ratio {
                output.push(self.sum / self.count as f32);
                self.sum = 0.0;
                self.count = 0;
                self.phase -= self.ratio;
            }
        }
        Ok(())
    }
}

/// Sinc Interpolation Resampler (Rubato) - High Quality, CPU Intensive
pub struct SincResampler {
    resampler: Mutex<FftFixedIn<f32>>,
    input_buffer: Vec<f32>,
    chunk_size: usize,
}

impl SincResampler {
    pub fn new(source_rate: u32, target_rate: u32) -> Result<Self> {
        let chunk_size = 1024;

        let resampler =
            FftFixedIn::<f32>::new(source_rate as usize, target_rate as usize, chunk_size, 2, 1)
                .map_err(ResampleError::RubatoConstructionError)?;

        Ok(Self {
            resampler: Mutex::new(resampler),
            input_buffer: Vec::with_capacity(chunk_size),
            chunk_size,
        })
    }
}

impl AudioResampler for SincResampler {
    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) -> Result<()> {
        self.input_buffer.extend_from_slice(input);

        while self.input_buffer.len() >= self.chunk_size {
            let chunk: Vec<f32> = self.input_buffer.drain(0..self.chunk_size).collect();
            let wave_input = vec![chunk];

            let mut resampler = self.resampler.lock().unwrap();
            let wave_output = resampler
                .process(&wave_input, None)
                .map_err(ResampleError::RubatoError)?;

            if let Some(chan0) = wave_output.first() {
                output.extend_from_slice(chan0);
            }
        }
        Ok(())
    }
}

/// FIR Decimation Resampler — matches ffmpeg's SWResample for integer downsampling.
pub struct FirDecimator {
    factor: usize,
    coefficients: Vec<f32>,
    history: Vec<f32>,
    write_pos: usize,
    phase: usize,
}

impl FirDecimator {
    pub fn new(source_rate: u32, target_rate: u32) -> Self {
        assert!(
            source_rate > target_rate,
            "FirDecimator only supports downsampling"
        );
        assert_eq!(
            source_rate % target_rate,
            0,
            "FirDecimator requires integer ratio"
        );

        let factor = (source_rate / target_rate) as usize;

        let taps_per_phase: usize = 16;
        let num_taps = taps_per_phase * factor;
        let cutoff_hz = 0.97 * (target_rate as f64 / 2.0);
        let kaiser_beta = 6.2;

        let coefficients = design_lowpass_fir(num_taps, cutoff_hz, source_rate as f64, kaiser_beta);

        info!(
            "FIR Decimator: {}x decimation ({} -> {} Hz), {} taps, cutoff {:.0} Hz",
            factor, source_rate, target_rate, num_taps, cutoff_hz
        );

        Self {
            factor,
            coefficients,
            history: vec![0.0f32; num_taps],
            write_pos: 0,
            phase: 0,
        }
    }
}

impl AudioResampler for FirDecimator {
    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) -> Result<()> {
        let num_taps = self.coefficients.len();

        for &sample in input {
            self.history[self.write_pos] = sample;
            self.write_pos = (self.write_pos + 1) % num_taps;

            self.phase += 1;
            if self.phase >= self.factor {
                self.phase = 0;

                let mut sum = 0.0f32;
                for k in 0..num_taps {
                    let idx = (self.write_pos + k) % num_taps;
                    sum += self.history[idx] * self.coefficients[k];
                }
                output.push(sum);
            }
        }
        Ok(())
    }
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let half_x = x / 2.0;
    let mut term = 1.0;
    for k in 1..25 {
        term *= half_x / k as f64;
        sum += term * term;
    }
    sum
}

fn design_lowpass_fir(num_taps: usize, cutoff_hz: f64, sample_rate: f64, beta: f64) -> Vec<f32> {
    let fc = cutoff_hz / sample_rate;
    let center = (num_taps - 1) as f64 / 2.0;
    let m = (num_taps - 1) as f64;
    let inv_i0_beta = 1.0 / bessel_i0(beta);

    let mut coeffs: Vec<f64> = (0..num_taps)
        .map(|i| {
            let n = i as f64 - center;
            let sinc = if n.abs() < 1e-10 {
                2.0 * fc
            } else {
                (2.0 * PI * fc * n).sin() / (PI * n)
            };
            let t = 2.0 * i as f64 / m - 1.0;
            let window = bessel_i0(beta * (1.0 - t * t).max(0.0).sqrt()) * inv_i0_beta;
            sinc * window
        })
        .collect();

    let sum: f64 = coeffs.iter().sum();
    for c in &mut coeffs {
        *c /= sum;
    }

    coeffs.iter().map(|&c| c as f32).collect()
}

pub fn create_resampler(
    source_rate: u32,
    target_rate: u32,
    resampler_type: Option<&str>,
) -> Result<Box<dyn AudioResampler>> {
    info!(
        "Resampler check: Source={} Hz, Target={} Hz",
        source_rate, target_rate
    );

    if source_rate == target_rate {
        info!("Source and Target rates match. Using pass-through.");
        return Ok(Box::new(LinearResampler::new(source_rate, target_rate)));
    }

    if let Some(rt) = resampler_type {
        if rt == "sinc" || rt == "rubato" {
            return Ok(Box::new(SincResampler::new(source_rate, target_rate)?));
        }
    }

    // Default to 'sinc' (Rubato) to ensure high-fidelity resampling and feature parity
    // with the Python backend used for training, as linear creates severe aliasing issues.
    info!("Defaulting to SincResampler (Rubato) for high quality");
    Ok(Box::new(SincResampler::new(source_rate, target_rate)?))
}

/// Helper to resample a single chunk of audio
pub fn resample_chunk(input: &[f32], source_rate: u32, target_rate: u32) -> Result<Vec<f32>> {
    let mut resampler = create_resampler(source_rate, target_rate, None)?; // Use default (Linear) for consistency
    let mut output = Vec::with_capacity(
        (input.len() as f64 * target_rate as f64 / source_rate as f64) as usize + 100,
    );
    resampler.process(input, &mut output)?;
    Ok(output)
}
