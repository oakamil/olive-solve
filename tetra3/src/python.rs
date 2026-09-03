// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

//! Python/PyO3 interoperability helpers for the `tetra3` crate.
//!
//! Provides conversion functions between native Rust structs and Python dictionaries / NumPy arrays.

use crate::extractor::{BgSubMode, Crop, ExtractOptions, SigmaMode};
use crate::fast_extractor::{FastBgSubMode, FastDownsample, FastExtractOptions, FastSigmaMode};
use crate::solver::{Solution, SolveOptions};
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Converts a slice of standard extractor `CentroidResult`s into an `N x 2` NumPy array `(y, x)`.
pub fn centroids_to_numpy<'py>(
    py: Python<'py>,
    centroids: &[crate::extractor::CentroidResult],
) -> Bound<'py, pyo3::types::PyAny> {
    let num_centroids = centroids.len();
    let mut cents = Vec::with_capacity(num_centroids * 2);
    for c in centroids {
        cents.push(c.y);
        cents.push(c.x);
    }
    numpy::PyArray1::from_slice(py, &cents)
        .reshape([num_centroids, 2])
        .unwrap()
        .into_any()
}

/// Converts a slice of fast extractor `FastCentroidResult`s into an `N x 2` NumPy array `(y, x)`.
pub fn fast_centroids_to_numpy<'py>(
    py: Python<'py>,
    centroids: &[crate::fast_extractor::FastCentroidResult],
) -> Bound<'py, pyo3::types::PyAny> {
    let num_centroids = centroids.len();
    let mut cents = Vec::with_capacity(num_centroids * 2);
    for c in centroids {
        cents.push(c.y);
        cents.push(c.x);
    }
    numpy::PyArray1::from_slice(py, &cents)
        .reshape([num_centroids, 2])
        .unwrap()
        .into_any()
}

impl ExtractOptions {
    /// Parses `ExtractOptions` from a Python dictionary of keyword arguments.
    pub fn from_kwargs(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
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
}

impl FastExtractOptions {
    /// Parses `FastExtractOptions` from a Python dictionary of keyword arguments.
    pub fn from_kwargs(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut options = FastExtractOptions::default();
        options.approximate_background = true; // Default to true for fast path
        let mut block_size = 32;

        if let Some(dict) = kwargs {
            if let Some(val) = dict.get_item("filtsize")? {
                block_size = val.extract()?;
            }
            if let Some(val) = dict.get_item("sigma")? {
                options.sigma = val.extract()?;
            }
            if let Some(val) = dict.get_item("downsample")? {
                let ds: Option<usize> = val.extract()?;
                options.downsample = match ds {
                    None | Some(1) => FastDownsample::None,
                    Some(2) => FastDownsample::X2,
                    Some(4) => FastDownsample::X4,
                    _ => {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Invalid downsample for fast path",
                        ));
                    }
                };
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
            if let Some(val) = dict.get_item("approximate_background")? {
                options.approximate_background = val.extract()?;
            }

            // Background Subtraction Mode
            if let Some(val) = dict.get_item("bg_sub_mode")? {
                if val.is_none() {
                    options.bg_sub_mode = None;
                } else {
                    let mode_str: String = val.extract()?;
                    options.bg_sub_mode = match mode_str.to_lowercase().as_str() {
                        "local_median" | "block_median" => {
                            Some(FastBgSubMode::BlockMedian { block_size })
                        }
                        "line_median" => Some(FastBgSubMode::LineMedian),
                        "global_median" => Some(FastBgSubMode::GlobalMedian),
                        "global_mean" => Some(FastBgSubMode::GlobalMean),
                        "none" => None,
                        _ => {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "Invalid bg_sub_mode for fast path: {}",
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
                    "global_median_abs" => FastSigmaMode::GlobalMedianAbs,
                    "global_root_square" => FastSigmaMode::GlobalRootSquare,
                    _ => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Invalid sigma_mode for fast path: {}",
                            mode_str
                        )));
                    }
                };
            }

            // Virtual crops
            if let Some(val) = dict.get_item("virtual_crops")? {
                if !val.is_none() {
                    let py_list: Vec<Bound<'_, pyo3::types::PyTuple>> = val.extract()?;
                    let mut crops = Vec::new();
                    for py_crop in py_list {
                        let len = py_crop.len();
                        if len == 1 {
                            let fraction: usize = py_crop.get_item(0)?.extract()?;
                            crops.push(Crop::Fraction(fraction));
                        } else if len == 2 {
                            let height: usize = py_crop.get_item(0)?.extract()?;
                            let width: usize = py_crop.get_item(1)?.extract()?;
                            crops.push(Crop::Center { height, width });
                        } else if len == 4 {
                            let height: usize = py_crop.get_item(0)?.extract()?;
                            let width: usize = py_crop.get_item(1)?.extract()?;
                            let offset_y: isize = py_crop.get_item(2)?.extract()?;
                            let offset_x: isize = py_crop.get_item(3)?.extract()?;
                            crops.push(Crop::Region {
                                height,
                                width,
                                offset_y,
                                offset_x,
                            });
                        } else {
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                "Invalid virtual crop format",
                            ));
                        }
                    }
                    options.virtual_crops = Some(crops);
                }
            }
        }
        Ok(options)
    }
}

impl SolveOptions {
    /// Parses `SolveOptions` from a Python dictionary of keyword arguments.
    pub fn from_kwargs(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
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
            if let Some(val) = dict.get_item("solve_timeout")? {
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

            // Target configurations
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
            if let Some(val) = dict.get_item("observer_latitude")? {
                if !val.is_none() {
                    options.observer_latitude = Some(val.extract()?);
                }
            }
            if let Some(val) = dict.get_item("observer_lst")? {
                if !val.is_none() {
                    options.observer_lst = Some(val.extract()?);
                }
            }
            if let Some(val) = dict.get_item("min_boresight_altitude")? {
                if !val.is_none() {
                    options.min_boresight_altitude = Some(val.extract()?);
                }
            }
            if let Some(val) = dict.get_item("return_best_failed_match")? {
                if !val.is_none() {
                    options.return_best_failed_match = val.extract()?;
                }
            }
        }
        Ok(options)
    }
}

impl Solution {
    /// Serializes the solver `Solution` into a standard Python dictionary.
    /// Includes coordinate properties, matches, and timing metrics.
    pub fn to_dict<'py>(
        &self,
        py: Python<'py>,
        ext_time: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let out_dict = PyDict::new(py);

        out_dict.set_item("RA", self.ra)?;
        out_dict.set_item("Dec", self.dec)?;
        out_dict.set_item("Roll", self.roll)?;
        out_dict.set_item("FOV", self.fov)?;
        out_dict.set_item("distortion", self.distortion)?;

        out_dict.set_item("RMSE", self.rmse)?;
        out_dict.set_item("P90E", self.p90e)?;
        out_dict.set_item("MAXE", self.maxe)?;
        out_dict.set_item("Matches", self.matches)?;
        out_dict.set_item("Prob", self.prob)?;
        out_dict.set_item("is_mirrored", self.is_mirrored)?;

        out_dict.set_item("epoch_equinox", self.epoch_equinox)?;
        out_dict.set_item("epoch_proper_motion", self.epoch_proper_motion)?;
        out_dict.set_item("status", format!("{:?}", self.status))?;

        if let Some(et) = ext_time {
            out_dict.set_item("T_extract", et)?;
        }
        out_dict.set_item("T_solve", self.t_solve_ms)?;

        if let Some(ref target_ra) = self.target_ra {
            out_dict.set_item("RA_target", target_ra)?;
        }
        if let Some(ref target_dec) = self.target_dec {
            out_dict.set_item("Dec_target", target_dec)?;
        }
        if let Some(ref target_y) = self.target_y {
            out_dict.set_item("y_target", target_y)?;
        }
        if let Some(ref target_x) = self.target_x {
            out_dict.set_item("x_target", target_x)?;
        }

        if let Some(ref matched_centroids) = self.matched_centroids {
            out_dict.set_item("matched_centroids", matched_centroids)?;
        }
        if let Some(ref matched_stars) = self.matched_stars {
            out_dict.set_item("matched_stars", matched_stars)?;
        }
        if let Some(ref matched_cat_id) = self.matched_cat_id {
            out_dict.set_item("matched_catID", matched_cat_id)?;
        }
        if let Some(ref catalog_stars) = self.catalog_stars {
            out_dict.set_item("catalog_stars", catalog_stars)?;
        }

        if let Some(ref rm) = self.rotation_matrix {
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

impl crate::fast_extractor::FastExtractOptionsUpdate {
    /// Parses `FastExtractOptionsUpdate` from a Python dictionary.
    pub fn from_dict(dict: &pyo3::Bound<'_, pyo3::types::PyDict>) -> pyo3::PyResult<Self> {
        let mut update = Self::default();
        if let Some(val) = dict.get_item("sigma")? {
            update.sigma = val.extract()?;
        }
        if let Some(val) = dict.get_item("noise_filter")? {
            update.noise_filter = val.extract()?;
        } else if let Some(val) = dict.get_item("binary_open")? {
            update.noise_filter = val.extract()?;
        }
        if let Some(val) = dict.get_item("min_area")? {
            update.min_area = val.extract()?;
        }
        if let Some(val) = dict.get_item("max_area")? {
            update.max_area = val.extract()?;
        }
        if let Some(val) = dict.get_item("virtual_crops")? {
            if val.is_none() {
                update.virtual_crops = Some(None);
            } else {
                let py_list: Vec<pyo3::Bound<'_, pyo3::types::PyTuple>> = val.extract()?;
                let mut crops = Vec::new();
                for py_crop in py_list {
                    let len = py_crop.len();
                    if len == 1 {
                        let fraction: usize = py_crop.get_item(0)?.extract()?;
                        crops.push(crate::extractor::Crop::Fraction(fraction));
                    } else if len == 2 {
                        let height: usize = py_crop.get_item(0)?.extract()?;
                        let width: usize = py_crop.get_item(1)?.extract()?;
                        crops.push(crate::extractor::Crop::Center { height, width });
                    } else if len == 4 {
                        let height: usize = py_crop.get_item(0)?.extract()?;
                        let width: usize = py_crop.get_item(1)?.extract()?;
                        let offset_y: isize = py_crop.get_item(2)?.extract()?;
                        let offset_x: isize = py_crop.get_item(3)?.extract()?;
                        crops.push(crate::extractor::Crop::Region {
                            height,
                            width,
                            offset_y,
                            offset_x,
                        });
                    } else {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Invalid crop specification",
                        ));
                    }
                }
                update.virtual_crops = Some(Some(crops));
            }
        }
        Ok(update)
    }
}
