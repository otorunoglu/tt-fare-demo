#[cfg(feature = "python")]
use numpy::ToPyArray;
#[cfg(feature = "python")]
use pyo3::prelude::*;

pub mod io;
pub mod resample;

/// Loads an audio file, mixes to mono, resamples to `target_rate`.
///
/// Returns a tuple (audio_data_numpy_array, sample_rate).
#[cfg(feature = "python")]
#[pyfunction]
fn load_audio(py: Python, path: String, target_rate: u32) -> PyResult<(PyObject, u32)> {
    let (audio, sr) = io::load_audio_file(&path, target_rate)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let audio_np = audio.to_pyarray(py).to_object(py);
    Ok((audio_np, sr))
}

/// Resamples a numpy array from `orig_sr` to `target_sr`.
#[cfg(feature = "python")]
#[pyfunction]
fn resample_numpy(
    py: Python,
    audio: numpy::PyReadonlyArray1<f32>,
    orig_sr: u32,
    target_sr: u32,
) -> PyResult<PyObject> {
    let audio_slice = audio.as_slice()?;
    let resampled = resample::resample_chunk(audio_slice, orig_sr, target_sr)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let audio_np = resampled.to_pyarray(py).to_object(py);
    Ok(audio_np)
}

/// A Python module implemented in Rust.
#[cfg(feature = "python")]
#[pymodule]
fn audio_preprocessing(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_audio, m)?)?;
    m.add_function(wrap_pyfunction!(resample_numpy, m)?)?;
    Ok(())
}
