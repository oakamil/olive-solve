// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

use crate::extractor::ExtractOptions;
use crate::solver::SolveOptions;
use crate::tetra3::Tetra3;

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

        // 2. Extract a zero-copy ndarray view directly from the Python array memory
        let img_view = image.as_array();

        // 3. Run the native Rust extraction pipeline
        let result = self.inner.get_centroids_from_image(&img_view, options);

        // 4. Setup the output dictionary
        let out_dict = PyDict::new_bound(py);

        // Instead of allocating hundreds of individual Python dictionaries and float objects for each star,
        // we pack everything into a single flat vector.
        let num_centroids = result.centroids.len();
        let mut flat_centroids = Vec::with_capacity(num_centroids * 4);

        for c in result.centroids {
            flat_centroids.push(c.y);
            flat_centroids.push(c.x);
            flat_centroids.push(c.sum);
            flat_centroids.push(c.area as f64); // Cast usize area to f64 to maintain homogeneous array type
        }

        // Convert the flat Vec into a single contiguous N x 4 NumPy array in one shot.
        // Format: [[y, x, sum, area], [y, x, sum, area], ...]
        let py_centroids = numpy::PyArray1::from_slice(py, &flat_centroids)
            .reshape([num_centroids, 4])
            .unwrap();

        out_dict.set_item("centroids", py_centroids)?;
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
        // 1. Parse kwargs into respective native configuration structs
        let extract_options = parse_extract_options(kwargs)?;
        let solve_options = parse_solve_options(kwargs)?;

        // 2. Get zero-copy view of the NumPy array
        let img_view = image.as_array();

        // 3. Run the full native pipeline
        let (solution, ext_time) = self
            .inner
            .solve_from_image(&img_view, extract_options, solve_options)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // 4. Convert Solution to a Python Dict
        let out_dict = PyDict::new_bound(py);
        out_dict.set_item("ra", solution.ra)?;
        out_dict.set_item("dec", solution.dec)?;
        out_dict.set_item("roll", solution.roll)?;
        out_dict.set_item("fov", solution.fov)?;
        out_dict.set_item("t_extract_ms", ext_time)?;
        out_dict.set_item("t_solve_ms", solution.t_solve_ms)?;

        // 5. Convert the rotation matrix into a 3x3 NumPy array if the solver provided one
        if let Some(rm) = solution.rotation_matrix {
            let flat_slice = rm.as_slice().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Matrix not contiguous")
            })?;

            // Map the flat slice back to a 3x3 NumPy array natively
            let py_matrix = numpy::PyArray1::from_slice(py, flat_slice)
                .reshape([3, 3])
                .unwrap();
            out_dict.set_item("rotation_matrix", py_matrix)?;
        }

        Ok(out_dict)
    }
}

// --- Helper Functions to Map Python kwargs to Rust Structs ---

/// Helper function to parse extraction **kwargs.
fn parse_extract_options(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<ExtractOptions> {
    let mut options = ExtractOptions::default();
    if let Some(dict) = kwargs {
        if let Some(sigma) = dict.get_item("sigma")? {
            options.sigma = sigma.extract()?;
        }
        if let Some(ds) = dict.get_item("downsample")? {
            options.downsample = ds.extract()?;
        }
        // Expand this section to catch other ExtractOptions overrides passed from Python
    }
    Ok(options)
}

/// Helper function to parse solver **kwargs.
fn parse_solve_options(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<SolveOptions> {
    let mut options = SolveOptions::default();
    if let Some(dict) = kwargs {
        if let Some(fov) = dict.get_item("fov_estimate")? {
            options.fov_estimate = fov.extract()?;
        }
        if let Some(rad) = dict.get_item("match_radius")? {
            options.match_radius = rad.extract()?;
        }
        // Expand this section to catch other SolveOptions overrides passed from Python
    }
    Ok(options)
}

// --- Module Initialization ---

/// The module initialization function. This matches the name defined in Cargo.toml.
/// This exposes the PyTetra3 class to Python under the name `Tetra3`.
#[pymodule]
fn tetra3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTetra3>()?;
    Ok(())
}
