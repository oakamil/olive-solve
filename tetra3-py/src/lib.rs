// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

use tetra3_core::Tetra3;
use tetra3_core::extractor::{BgSubMode, ExtractOptions, SigmaMode};
use tetra3_core::solver::SolveOptions;

/// Python wrapper for the highly optimized Tetra3 engine.
#[pyclass(name = "Tetra3", unsendable)]
pub struct PyTetra3 {
    inner: Tetra3,
}

#[pymethods]
impl PyTetra3 {
    /// Creates a new Tetra3 instance.
    /// The database is lazy-loaded, meaning it won't hit the disk until
    /// the first plate solving operation is executed.
    #[new]
    fn new(database_path: String) -> Self {
        Self {
            inner: Tetra3::new(PathBuf::from(database_path)),
        }
    }

    /// Extracts centroids from a 2D NumPy array.
    /// Uses PyO3's buffer protocol to read directly from Python's memory (zero-copy).
    #[pyo3(signature = (image, **kwargs))]
    fn get_centroids_from_image<'py>(
        &mut self,
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f32>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        // 1. Parse Python **kwargs into the native ExtractOptions struct
        let options = parse_extract_options(kwargs)?;

        // Check if Python specifically asked for moments (defaults to False in standard tetra3)
        let return_moments = kwargs
            .and_then(|d| d.get_item("return_moments").ok().flatten())
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(false);

        // 2. Extract a zero-copy ndarray view directly from the Python array memory
        let img_view = image.as_array();

        // 3. Run the native Rust extraction pipeline
        let result = self.inner.get_centroids_from_image(&img_view, options);

        // 4. Setup the output dictionary
        let out_dict = PyDict::new(py);
        let num_centroids = result.centroids.len();

        // 5. Pack the Centroids (Optionally including moments)
        if return_moments {
            let mut flat_centroids = Vec::with_capacity(num_centroids * 8);
            for c in result.centroids {
                flat_centroids.push(c.y);
                flat_centroids.push(c.x);
                flat_centroids.push(c.sum);
                flat_centroids.push(c.area as f64);
                flat_centroids.push(c.m2_xx);
                flat_centroids.push(c.m2_yy);
                flat_centroids.push(c.m2_xy);
                flat_centroids.push(c.axis_ratio);
            }

            let py_centroids = numpy::PyArray1::from_slice(py, &flat_centroids)
                .reshape([num_centroids, 8])
                .unwrap();
            out_dict.set_item("centroids", py_centroids)?;
        } else {
            let mut flat_centroids = Vec::with_capacity(num_centroids * 4);
            for c in result.centroids {
                flat_centroids.push(c.y);
                flat_centroids.push(c.x);
                flat_centroids.push(c.sum);
                flat_centroids.push(c.area as f64);
            }

            let py_centroids = numpy::PyArray1::from_slice(py, &flat_centroids)
                .reshape([num_centroids, 4])
                .unwrap();
            out_dict.set_item("centroids", py_centroids)?;
        }

        // 6. Return Debug Images (if requested and present)
        // Mapped to the new nested DebugImages struct
        if let Some(debug_images) = result.debug_images {
            // Zero-copy ownership transfer of underlying memory
            let py_removed_bg = debug_images.removed_background.into_pyarray(py);
            out_dict.set_item("image_removed_background", py_removed_bg)?;

            let py_cropped = debug_images.cropped_and_downsampled.into_pyarray(py);
            out_dict.set_item("image_cropped_and_downsampled", py_cropped)?;

            let py_mask = debug_images.binary_mask.into_pyarray(py);
            out_dict.set_item("image_binary_mask", py_mask)?;
        }

        Ok(out_dict)
    }

    /// Runs extraction and plate solving in one uninterrupted pipeline.
    /// Returns a dictionary containing the solution and execution times.
    #[pyo3(signature = (image, **kwargs))]
    fn solve_from_image<'py>(
        &mut self,
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f32>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let extract_options = parse_extract_options(kwargs)?;
        let solve_options = parse_solve_options(kwargs)?;

        let img_view = image.as_array();

        let (solution, ext_time) = self
            .inner
            .solve_from_image(&img_view, extract_options, solve_options)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Map the new Solution structure to a Python Dict
        let out_dict = PyDict::new(py);
        out_dict.set_item("ra", solution.ra)?;
        out_dict.set_item("dec", solution.dec)?;
        out_dict.set_item("roll", solution.roll)?;
        out_dict.set_item("fov", solution.fov)?;

        out_dict.set_item("distortion", solution.distortion)?;
        out_dict.set_item("rmse", solution.rmse)?;
        out_dict.set_item("prob", solution.prob)?;
        out_dict.set_item("matches", solution.matches)?;

        out_dict.set_item("t_extract_ms", ext_time)?;
        out_dict.set_item("t_solve_ms", solution.t_solve_ms)?;

        if let Some(rm) = solution.rotation_matrix {
            let flat_slice = rm.as_slice().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Matrix not contiguous")
            })?;
            let py_matrix = numpy::PyArray1::from_slice(py, flat_slice)
                .reshape([3, 3])
                .unwrap();
            out_dict.set_item("rotation_matrix", py_matrix)?;
        }

        Ok(out_dict)
    }
}

// --- Helper Functions to Map Python kwargs to Rust Structs ---

fn parse_extract_options(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<ExtractOptions> {
    let mut options = ExtractOptions::default();

    if let Some(dict) = kwargs {
        if let Some(val) = dict.get_item("sigma")? {
            options.sigma = val.extract()?;
        }
        if let Some(val) = dict.get_item("image_th")? {
            options.image_th = val.extract()?;
        }
        if let Some(val) = dict.get_item("downsample")? {
            options.downsample = val.extract()?;
        }
        if let Some(val) = dict.get_item("filtsize")? {
            options.filtsize = val.extract()?;
        }
        if let Some(val) = dict.get_item("binary_open")? {
            options.binary_open = val.extract()?;
        }
        if let Some(val) = dict.get_item("centroid_window")? {
            options.centroid_window = val.extract()?;
        }
        if let Some(val) = dict.get_item("min_area")? {
            options.min_area = val.extract()?;
        }
        if let Some(val) = dict.get_item("max_area")? {
            options.max_area = val.extract()?;
        }
        if let Some(val) = dict.get_item("min_sum")? {
            options.min_sum = val.extract()?;
        }
        if let Some(val) = dict.get_item("max_sum")? {
            options.max_sum = val.extract()?;
        }
        if let Some(val) = dict.get_item("max_axis_ratio")? {
            options.max_axis_ratio = val.extract()?;
        }
        if let Some(val) = dict.get_item("max_returned")? {
            options.max_returned = val.extract()?;
        }
        if let Some(val) = dict.get_item("return_images")? {
            options.return_images = val.extract()?;
        }

        // Background Subtraction Mode
        if let Some(val) = dict.get_item("bg_sub_mode")? {
            if val.is_none() {
                options.bg_sub_mode = None;
            } else {
                let mode_str: String = val.extract()?;
                options.bg_sub_mode = match mode_str.to_lowercase().as_str() {
                    "local_median" => Some(BgSubMode::LocalMedian),
                    "local_mean" => Some(BgSubMode::LocalMean),
                    "global_median" => Some(BgSubMode::GlobalMedian),
                    "global_mean" => Some(BgSubMode::GlobalMean),
                    "none" => None,
                    _ => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Invalid bg_sub_mode: {}",
                            mode_str
                        )));
                    }
                };
            }
        }

        // Sigma Threshold Mode
        if let Some(val) = dict.get_item("sigma_mode")? {
            let mode_str: String = val.extract()?;
            options.sigma_mode = match mode_str.to_lowercase().as_str() {
                "local_median_abs" => SigmaMode::LocalMedianAbs,
                "local_root_square" => SigmaMode::LocalRootSquare,
                "global_median_abs" => SigmaMode::GlobalMedianAbs,
                "global_root_square" => SigmaMode::GlobalRootSquare,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid sigma_mode: {}",
                        mode_str
                    )));
                }
            };
        }
    }
    Ok(options)
}

fn parse_solve_options(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<SolveOptions> {
    let mut options = SolveOptions::default();

    if let Some(dict) = kwargs {
        if let Some(val) = dict.get_item("fov_estimate")? {
            options.fov_estimate = val.extract()?;
        }
        if let Some(val) = dict.get_item("fov_max_error")? {
            options.fov_max_error = val.extract()?;
        }
        if let Some(val) = dict.get_item("match_radius")? {
            options.match_radius = val.extract()?;
        }
        if let Some(val) = dict.get_item("match_threshold")? {
            options.match_threshold = val.extract()?;
        }
        if let Some(val) = dict.get_item("solve_timeout_ms")? {
            options.solve_timeout_ms = val.extract()?;
        }
        if let Some(val) = dict.get_item("distortion")? {
            options.distortion = val.extract()?;
        }
        if let Some(val) = dict.get_item("match_max_error")? {
            options.match_max_error = val.extract()?;
        }
        if let Some(val) = dict.get_item("return_matches")? {
            options.return_matches = val.extract()?;
        }
        if let Some(val) = dict.get_item("return_catalog")? {
            options.return_catalog = val.extract()?;
        }
        if let Some(val) = dict.get_item("return_rotation_matrix")? {
            options.return_rotation_matrix = val.extract()?;
        }

        // Target configurations (Parsing 2D arrays directly into ndarrays)
        if let Some(val) = dict.get_item("target_pixel")? {
            if !val.is_none() {
                let py_arr: numpy::PyReadonlyArray2<f64> = val.extract()?;
                options.target_pixel = Some(py_arr.as_array().to_owned());
            }
        }
        if let Some(val) = dict.get_item("target_sky_coord")? {
            if !val.is_none() {
                let py_arr: numpy::PyReadonlyArray2<f64> = val.extract()?;
                options.target_sky_coord = Some(py_arr.as_array().to_owned());
            }
        }
    }
    Ok(options)
}

// --- Module Initialization ---

/// The module initialization function. This matches the name defined in Cargo.toml.
/// This exposes the PyTetra3 class to Python under the name `Tetra3`.
#[pymodule]
#[pyo3(name = "tetra3")]
fn tetra3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTetra3>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyModule;
    use std::path::Path;

    const PYTHON_EXAMPLE_CODE: &std::ffi::CStr = cr#"
import os
import time
import glob
import numpy as np
from PIL import Image

import tetra3 

def run_benchmark(db_path, img_dir):
    if not os.path.exists(db_path):
        return
    if not os.path.exists(img_dir):
        return

    print(f"\n{'Filename':<40} | {'RA':<8} | {'Dec':<8} | {'Extract (ms)':<12} | {'Solve (ms)':<12} | {'Total Inv (ms)':<14}")
    print("-" * 105)

    t3 = tetra3.Tetra3(db_path)
    search_pattern = os.path.join(img_dir, "*.jpg")
    image_files = sorted(glob.glob(search_pattern))

    for img_path in image_files:
        filename = os.path.basename(img_path)
        img = Image.open(img_path).convert('L')
        img_arr = np.asarray(img, dtype=np.float32)
        
        t0 = time.perf_counter()
        result = t3.solve_from_image(img_arr)
        inv_time_ms = (time.perf_counter() - t0) * 1000.0

        ra = result.get("ra")
        dec = result.get("dec")
        ext_ms = result.get("t_extract_ms", 0.0)
        solve_ms = result.get("t_solve_ms", 0.0)

        ra_str = f"{ra:.3f}" if ra is not None else "N/A"
        dec_str = f"{dec:.3f}" if dec is not None else "N/A"

        print(f"{filename:<40.40} | {ra_str:<8} | {dec_str:<8} | {ext_ms:<12.2f} | {solve_ms:<12.2f} | {inv_time_ms:<14.2f}")

    print("-" * 105 + "\n")
"#;

    #[test]
    #[ignore]
    fn test_python_wrapper_from_python_script() {
        let db_path = "../tetra3/tests/fixtures/default_database.npz";
        let img_dir = "../tetra3/tests/fixtures/sample_images";

        if !Path::new(db_path).exists() || !Path::new(img_dir).exists() {
            return; // Skip gracefully if data isn't present
        }

        Python::initialize();
        Python::attach(|py| {
            // 1. Manually build the `tetra3` module
            let tetra3_mod = PyModule::new(py, "tetra3").unwrap();
            tetra3_mod.add_class::<PyTetra3>().unwrap();

            // 2. Inject the module into Python's `sys.modules`
            let sys = py.import("sys").unwrap();
            sys.getattr("modules")
                .unwrap()
                .set_item("tetra3", tetra3_mod)
                .unwrap();

            // 3. Compile and load the embedded script
            let main_mod =
                PyModule::from_code(py, PYTHON_EXAMPLE_CODE, c"example.py", c"example").unwrap();

            // 4. Run it
            let run_benchmark = main_mod.getattr("run_benchmark").unwrap();
            run_benchmark.call1((db_path, img_dir)).unwrap();
        });
    }
}
