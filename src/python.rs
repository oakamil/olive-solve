// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use crate::FusedSolver;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

#[pyclass(name = "FusedSolver")]
/// A Python wrapper for the FusedSolver, providing plate solving and star extraction.
/// The `FusedSolver` unifies standard and fast extraction pipelines, along with IMU support.
pub struct PyFusedSolver {
    inner: FusedSolver,
}

#[pymethods]
impl PyFusedSolver {
    #[new]
    #[pyo3(signature = (database_path, imu_type=None, enable_accel=true))]
    /// Initializes a new FusedSolver instance.
    ///
    /// Args:
    ///     database_path (str): The file path to the npz star database.
    ///     imu_type (str, optional): The specific IMU to use (e.g. "bno085", "bmi160", "auto", "none"). Defaults to auto-detection.
    ///     enable_accel (bool, optional): Whether to enable accelerometer hardware/features. Defaults to true.
    pub fn new(database_path: &str, imu_type: Option<&str>, enable_accel: bool) -> PyResult<Self> {
        let rust_imu_type = match imu_type {
            Some(s) => Some(
                s.parse::<crate::ImuType>()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?,
            ),
            None => None,
        };

        let inner = FusedSolver::new(
            std::path::Path::new(database_path),
            rust_imu_type,
            None,
            enable_accel,
        )
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (image, **kwargs))]
    /// Extracts star centroids using the standard pipeline.
    ///
    /// Args:
    ///     image (numpy.ndarray): 2D float32 image array.
    ///     **kwargs: Extraction options (sigma, max_returned, downsample, etc).
    ///
    /// Returns:
    ///     numpy.ndarray: An array of shape (N, 2) containing (y, x) centroid coordinates.
    pub fn get_centroids_from_image<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f32>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let options = tetra3::extractor::ExtractOptions::from_kwargs(kwargs)?;
        let img_view = image.as_array();

        let result = self
            .inner
            .extract(&img_view, options)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        Ok(tetra3::python::centroids_to_numpy(py, &result.centroids))
    }

    #[pyo3(signature = (image, **kwargs))]
    /// Extracts star centroids using the highly-optimized fast sequential pipeline.
    ///
    /// Args:
    ///     image (numpy.ndarray): 2D uint8 or float32 image array.
    ///     **kwargs: Fast extraction options (downsample, max_returned, etc).
    ///
    /// Returns:
    ///     numpy.ndarray or tuple: Array of shape (N, 2) for centroids. If virtual crops
    ///     are used, returns a tuple containing (base_centroids, (crop_1_centroids, ...)).
    pub fn get_centroids_from_image_fast<'py>(
        &self,
        py: Python<'py>,
        image: Bound<'py, PyAny>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let options = tetra3::fast_extractor::FastExtractOptions::from_kwargs(kwargs)?;

        let result = if let Ok(img_u8) = image.extract::<numpy::PyReadonlyArray2<u8>>() {
            let img_view = img_u8.as_array();
            self.inner.extract_fast(&img_view, options)
        } else if let Ok(img_f32) = image.extract::<numpy::PyReadonlyArray2<f32>>() {
            let img_view = img_f32.as_array();
            self.inner.extract_fast(&img_view, options)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Image must be a 2D NumPy array of u8 or f32",
            ));
        }
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        let core_result = tetra3::python::fast_centroids_to_numpy(py, &result.centroids);
        if let Some(crop_results) = &result.virtual_crop_centroids {
            let mut crop_list = Vec::with_capacity(crop_results.len());
            for crop in crop_results {
                crop_list.push(tetra3::python::fast_centroids_to_numpy(py, crop));
            }
            let elements: Vec<Bound<'py, pyo3::types::PyAny>> =
                vec![core_result, PyTuple::new(py, crop_list).unwrap().into_any()];
            Ok(PyTuple::new(py, elements).unwrap().into_any())
        } else {
            Ok(core_result)
        }
    }

    #[pyo3(signature = (image, variants = None, **kwargs))]
    /// Extracts star centroids using multiple sequential configurations concurrently via the fast pipeline.
    ///
    /// Args:
    ///     image (numpy.ndarray): 2D uint8 or float32 image array.
    ///     variants (list): A list of dictionaries, where each dictionary specifies options to override
    ///                      for that specific extraction pass (e.g. `{'sigma': 5.0, 'min_area': 1}`).
    ///     **kwargs: Base fast extraction options.
    ///
    /// Returns:
    ///     list: A list of extraction results corresponding to each variant. Each result is
    ///           either a numpy.ndarray of shape (N, 2), or a tuple of (base, crops) if virtual crops are used.
    pub fn get_centroids_from_image_variants<'py>(
        &self,
        py: Python<'py>,
        image: Bound<'py, PyAny>,
        variants: Option<Vec<Bound<'py, PyDict>>>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let options = tetra3::fast_extractor::FastExtractOptions::from_kwargs(kwargs)?;

        let rust_variants = match variants {
            Some(v_list) => {
                let mut rv = Vec::with_capacity(v_list.len());
                for v in v_list {
                    rv.push(tetra3::fast_extractor::FastExtractOptionsUpdate::from_dict(
                        &v,
                    )?);
                }
                rv
            }
            None => vec![tetra3::fast_extractor::FastExtractOptionsUpdate::default()],
        };

        let results = if let Ok(img_u8) = image.extract::<numpy::PyReadonlyArray2<u8>>() {
            let img_view = img_u8.as_array();
            self.inner
                .get_centroids_from_image_variants(&img_view, options, &rust_variants)
        } else if let Ok(img_f32) = image.extract::<numpy::PyReadonlyArray2<f32>>() {
            let img_view = img_f32.as_array();
            self.inner
                .get_centroids_from_image_variants(&img_view, options, &rust_variants)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Image must be a 2D NumPy array of u8 or f32",
            ));
        }
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        let mut py_results = Vec::with_capacity(results.len());
        for result in results {
            let core_result = tetra3::python::fast_centroids_to_numpy(py, &result.centroids);
            if let Some(crop_results) = &result.virtual_crop_centroids {
                let mut crop_list = Vec::with_capacity(crop_results.len());
                for crop in crop_results {
                    crop_list.push(tetra3::python::fast_centroids_to_numpy(py, crop));
                }
                let elements: Vec<Bound<'py, pyo3::types::PyAny>> =
                    vec![core_result, PyTuple::new(py, crop_list).unwrap().into_any()];
                py_results.push(PyTuple::new(py, elements).unwrap().into_any());
            } else {
                py_results.push(core_result);
            }
        }

        Ok(py_results)
    }

    #[pyo3(signature = (image, **kwargs))]
    /// Performs a full plate solve from an image using the standard pipeline.
    ///
    /// Args:
    ///     image (numpy.ndarray): 2D float32 image array.
    ///     **kwargs: Options for both extraction and solving (fov_estimate, match_radius, etc).
    ///
    /// Returns:
    ///     dict: A dictionary containing the solve results (RA, Dec, Roll, FOV, status, T_extract, etc).
    pub fn solve_from_image<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f32>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let extract_options = tetra3::extractor::ExtractOptions::from_kwargs(kwargs)?;
        let solve_options = tetra3::solver::SolveOptions::from_kwargs(kwargs)?;
        let img_view = image.as_array();

        let (solution, ext_time) = self
            .inner
            .solve_from_image(&img_view, extract_options, solve_options, None)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        solution.to_dict(py, Some(ext_time))
    }

    #[pyo3(signature = (image, **kwargs))]
    /// Performs a full plate solve from an image using the highly-optimized fast sequential pipeline.
    ///
    /// Args:
    ///     image (numpy.ndarray): 2D uint8 or float32 image array.
    ///     **kwargs: Options for both fast extraction and solving.
    ///
    /// Returns:
    ///     dict: A dictionary containing the solve results (RA, Dec, Roll, FOV, status, T_extract, etc).
    pub fn solve_from_image_fast<'py>(
        &self,
        py: Python<'py>,
        image: Bound<'py, PyAny>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let extract_options = tetra3::fast_extractor::FastExtractOptions::from_kwargs(kwargs)?;
        let solve_options = tetra3::solver::SolveOptions::from_kwargs(kwargs)?;

        let (solution, ext_time) =
            if let Ok(img_u8) = image.extract::<numpy::PyReadonlyArray2<u8>>() {
                let img_view = img_u8.as_array();
                self.inner
                    .solve_from_image_fast(&img_view, extract_options, solve_options, None)
            } else if let Ok(img_f32) = image.extract::<numpy::PyReadonlyArray2<f32>>() {
                let img_view = img_f32.as_array();
                self.inner
                    .solve_from_image_fast(&img_view, extract_options, solve_options, None)
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Image must be a 2D NumPy array of u8 or f32",
                ));
            }
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        solution.to_dict(py, Some(ext_time))
    }

    #[pyo3(signature = (centroids, size, **kwargs))]
    /// Solves for the telescope's pointing using pre-extracted centroids.
    ///
    /// Args:
    ///     centroids (numpy.ndarray): 2D float64 array of shape (N, 2) representing (y, x) coordinates.
    ///     size (tuple): Image (height, width).
    ///     **kwargs: Solve options (fov_estimate, match_radius, solve_timeout, etc).
    ///
    /// Returns:
    ///     dict: A dictionary containing the solve results.
    pub fn solve_from_centroids<'py>(
        &self,
        py: Python<'py>,
        centroids: PyReadonlyArray2<'py, f64>,
        size: (f64, f64),
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let solve_options = tetra3::solver::SolveOptions::from_kwargs(kwargs)?;
        let cents_view = centroids.as_array().to_owned();
        let solution = self
            .inner
            .solve_from_centroids(&cents_view, size, solve_options, None)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        solution.to_dict(py, None)
    }

    #[pyo3(signature = (centroids_list, size, **kwargs))]
    /// Solves from multiple centroid sets sequentially, returning the first successful match.
    ///
    /// Args:
    ///     centroids_list (list): A list of numpy arrays, each of shape (N, 2).
    ///     size (tuple): Image (height, width).
    ///     **kwargs: Solve options.
    ///
    /// Returns:
    ///     dict: A dictionary containing the solve results from the successful set,
    ///           including the `crop_index` representing which set succeeded,
    ///           and `solve_time_ms` for the batch execution duration.
    pub fn solve_from_centroids_batch<'py>(
        &self,
        py: Python<'py>,
        centroids_list: Vec<PyReadonlyArray2<'py, f64>>,
        size: (f64, f64),
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let solve_options = tetra3::solver::SolveOptions::from_kwargs(kwargs)?;

        let cents_views: Vec<ndarray::ArrayView2<f64>> =
            centroids_list.iter().map(|c| c.as_array()).collect();
        let batch: Vec<(ndarray::Array2<f64>, Option<tetra3::extractor::Crop>)> =
            cents_views.iter().map(|v| (v.to_owned(), None)).collect();
        let batch_solution = self
            .inner
            .solve_from_centroids_batch(&batch, size, solve_options, None)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        let dict = batch_solution.solution.to_dict(py, None)?;
        dict.set_item("crop_index", batch_solution.crop_index)?;
        dict.set_item(
            "solve_time_ms",
            batch_solution.solve_time.as_millis() as u64,
        )?;
        Ok(dict)
    }

    /// Sets the observer's location, allowing the solver to compute azimuth/elevation
    /// or incorporate magnetic declination.
    ///
    /// Args:
    ///     lat (float): Latitude in degrees.
    ///     lon (float): Longitude in degrees.
    pub fn set_observer_location(&self, lat: f64, lon: f64) {
        self.inner.set_observer_location(lat, lon);
    }

    /// Attempts to initialize and start the configured IMU hardware (e.g. BNO080 or BMI160)
    /// to continually track the camera's orientation.
    ///
    /// Returns:
    ///     bool: True if IMU successfully started.
    pub fn start_imu(&self) -> PyResult<bool> {
        self.inner
            .start_imu()
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }

    /// Stops the IMU and drops the background polling thread. Safe to call if not started.
    pub fn stop_imu(&self) -> PyResult<()> {
        self.inner
            .stop_imu()
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }

    /// Resets the internal IMU zero-bias and deletes the SVD calibration matrix from persistent storage.
    pub fn reset_calibration(&self) {
        self.inner.reset_calibration();
    }

    /// Fetches the latest known orientation of the device.
    ///
    /// Returns:
    ///     dict: Dictionary containing ra, dec, roll, source, timestamp
    pub fn get_latest_position<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if let Some(pos) = self.inner.get_latest_position() {
            let dict = PyDict::new(py);
            dict.set_item("ra", pos.ra)?;
            dict.set_item("dec", pos.dec)?;
            dict.set_item("roll", pos.roll)?;
            dict.set_item("source", format!("{:?}", pos.source))?;

            let dt = pos
                .timestamp
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();
            dict.set_item("timestamp", dt)?;

            Ok(dict)
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "No position estimate available",
            ))
        }
    }

    #[pyo3(signature = (ra_or_dict, dec=None, roll=None, timestamp=None, pointing_only=false))]
    /// Feeds an external high-confidence plate solve into the fused solver.
    ///
    /// Updates the latest solved celestial position, clears the solve failure flag,
    /// and anchors the IMU if running and observer coordinates are configured.
    ///
    /// Can be invoked with explicit coordinates:
    ///     update_position(ra, dec, roll=0.0, timestamp=None, pointing_only=False)
    /// or with a solution dictionary (matching the format returned by `solve_from_*`):
    ///     update_position(solution_dict, timestamp=None, pointing_only=False)
    ///
    /// Args:
    ///     ra_or_dict (float or dict): Celestial Right Ascension in degrees (J2000), or a dictionary
    ///         containing 'ra', 'dec', and optional 'roll' keys.
    ///     dec (float, optional): Celestial Declination in degrees (J2000). Required if ra_or_dict is a float.
    ///     roll (float, optional): Camera position angle / roll in degrees. Defaults to 0.0.
    ///     timestamp (float, optional): Observation time as Unix timestamp in seconds. Defaults to current system time.
    ///     pointing_only (bool, optional): If False (default), the slew delta contributes to IMU SVD calibration.
    ///         If True, updates only the pointing anchor (useful for synthetic testing or drift reset). Defaults to False.
    ///
    /// Returns:
    ///     bool: True if the IMU tracking anchor was successfully updated, or False if only the internal position was updated.
    ///
    /// Raises:
    ///     TypeError: If arguments are of incorrect types or required coordinate keys are missing.
    ///     ValueError: If coordinate values cannot be parsed or are invalid.
    pub fn update_position<'py>(
        &self,
        _py: Python<'py>,
        ra_or_dict: Bound<'py, PyAny>,
        dec: Option<f64>,
        roll: Option<f64>,
        timestamp: Option<f64>,
        pointing_only: bool,
    ) -> PyResult<bool> {
        let (final_ra, final_dec, final_roll, dict_ts) =
            if let Ok(dict) = ra_or_dict.cast::<PyDict>() {
                let ra: f64 = dict
                    .get_item("ra")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err("Dictionary must contain 'ra' key")
                    })?
                    .extract()?;
                let dec: f64 = dict
                    .get_item("dec")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err("Dictionary must contain 'dec' key")
                    })?
                    .extract()?;
                let r: f64 = if let Some(r_val) = dict.get_item("roll")? {
                    r_val.extract().unwrap_or(0.0)
                } else {
                    roll.unwrap_or(0.0)
                };
                let ts: Option<f64> = if let Some(ts_val) = dict.get_item("timestamp")? {
                    ts_val.extract().ok()
                } else {
                    None
                };
                (ra, dec, r, ts)
            } else if let Ok(ra_num) = ra_or_dict.extract::<f64>() {
                let dec_num = dec.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "'dec' must be provided when passing 'ra' as a number",
                    )
                })?;
                (ra_num, dec_num, roll.unwrap_or(0.0), None)
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "First argument must be a dictionary or a numeric 'ra' coordinate",
                ));
            };

        let effective_ts = timestamp.or(dict_ts);
        let rust_timestamp = effective_ts.map(|ts| {
            if ts >= 0.0 {
                std::time::UNIX_EPOCH + std::time::Duration::from_secs_f64(ts)
            } else {
                std::time::SystemTime::now()
            }
        });

        Ok(self.inner.update_position(
            final_ra,
            final_dec,
            Some(final_roll),
            rust_timestamp,
            pointing_only,
        ))
    }

    /// Fetches the latest real-time hardware telemetry from the IMU.
    ///
    /// Returns:
    ///     dict: Dictionary containing timestamp, gyro, gravity_vector, relative_gyro_quaternion, hardware_quaternion, motion_state.
    pub fn get_sensor_data<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        if let Some(data) = self.inner.get_sensor_data() {
            let dict = PyDict::new(py);
            let dt = data
                .timestamp
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();
            dict.set_item("timestamp", dt)?;

            let py_gyro = pyo3::types::PyList::new(py, data.gyro)?;
            dict.set_item("gyro", py_gyro)?;

            let py_rel = pyo3::types::PyList::new(py, data.relative_gyro_quaternion)?;
            dict.set_item("relative_gyro_quaternion", py_rel)?;

            if let Some(gv) = data.gravity_vector {
                let py_gv = pyo3::types::PyList::new(py, gv)?;
                dict.set_item("gravity_vector", py_gv)?;
            } else {
                dict.set_item("gravity_vector", py.None())?;
            }

            if let Some(hq) = data.hardware_quaternion {
                let py_hq = pyo3::types::PyList::new(py, hq)?;
                dict.set_item("hardware_quaternion", py_hq)?;
            } else {
                dict.set_item("hardware_quaternion", py.None())?;
            }

            dict.set_item("motion_state", format!("{:?}", data.motion_state))?;

            Ok(Some(dict))
        } else {
            Ok(None)
        }
    }

    /// Retrieves the real-time motion stability state from the IMU hardware, if running.
    pub fn get_motion_state(&self) -> Option<String> {
        self.inner.get_motion_state().map(|s| format!("{:?}", s))
    }

    /// Retrieves the real-time calibration metrics from the IMU hardware, if running.
    pub fn get_calibration_status<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        if let Some(status) = self.inner.get_calibration_status() {
            let dict = PyDict::new(py);
            dict.set_item("transform_error_fraction", status.transform_error_fraction)?;
            dict.set_item("camera_view_gyro_axis", status.camera_view_gyro_axis)?;
            dict.set_item("camera_view_misalignment", status.camera_view_misalignment)?;
            dict.set_item("camera_up_gyro_axis", status.camera_up_gyro_axis)?;
            dict.set_item("camera_up_misalignment", status.camera_up_misalignment)?;
            Ok(Some(dict))
        } else {
            Ok(None)
        }
    }
}

#[pymodule]
fn olive_solve(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFusedSolver>()?;
    Ok(())
}
