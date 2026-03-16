// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use numpy::{PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

use tetra3_core::Tetra3;
use tetra3_core::extractor::ExtractOptions;
use tetra3_core::solver::SolveOptions;

/// Python wrapper for the highly optimized Tetra3 engine.
#[pyclass(name = "Tetra3", unsendable)]
pub struct PyTetra3 {
    inner: Tetra3,
}

#[pymethods]
impl PyTetra3 {
    #[new]
    fn new(database_path: String) -> Self {
        Self {
            inner: Tetra3::new(PathBuf::from(database_path)),
        }
    }

    #[pyo3(signature = (image, **kwargs))]
    fn get_centroids_from_image<'py>(
        &mut self,
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f32>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let options = parse_extract_options(kwargs)?;
        let img_view = image.as_array();
        let result = self.inner.get_centroids_from_image(&img_view, options);

        // Modern PyO3 returns Bound pointers by default
        let out_dict = PyDict::new(py);
        let num_centroids = result.centroids.len();
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
        Ok(out_dict)
    }

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

        let out_dict = PyDict::new(py);
        out_dict.set_item("ra", solution.ra)?;
        out_dict.set_item("dec", solution.dec)?;
        out_dict.set_item("roll", solution.roll)?;
        out_dict.set_item("fov", solution.fov)?;
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

fn parse_extract_options(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<ExtractOptions> {
    let mut options = ExtractOptions::default();
    if let Some(dict) = kwargs {
        if let Some(sigma) = dict.get_item("sigma")? {
            options.sigma = sigma.extract()?;
        }
        if let Some(ds) = dict.get_item("downsample")? {
            options.downsample = ds.extract()?;
        }
    }
    Ok(options)
}

fn parse_solve_options(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<SolveOptions> {
    let mut options = SolveOptions::default();
    if let Some(dict) = kwargs {
        if let Some(fov) = dict.get_item("fov_estimate")? {
            options.fov_estimate = fov.extract()?;
        }
        if let Some(rad) = dict.get_item("match_radius")? {
            options.match_radius = rad.extract()?;
        }
    }
    Ok(options)
}

/// The module initialization function exported to Python.
#[pymodule]
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
