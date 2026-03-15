// Copyright (c) 2026 Omair Kamil oakamil@gmail.com
// See LICENSE file in root directory for license terms.

use image::GrayImage;
use ndarray::Array2;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use walkdir::WalkDir;

use cedar_detect::algorithm::{estimate_noise_from_image, get_stars_from_image};
use tetra3::extractor::{CentroidConfig, TetraExtractor};

const PY_HELPER_CODE: &str = r#"
from PIL import Image
import numpy as np
from tetra3.tetra3 import get_centroids_from_image

def load_image_as_array(path, crop_to_multiple_of=1):
    image = Image.open(path)
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = image[:, :, 0]*.299 + image[:, :, 1]*.587 + image[:, :, 2]*.114
        else:
            image = image.squeeze(axis=2)
            
    # CRITICAL FIX: Truncate remainder pixels for safe downsampling.
    # NumPy reshaping in tetra3.py and binning in cedar-detect will crash if dimensions
    # are not perfectly divisible by the downsample factor. 
    if crop_to_multiple_of > 1:
        h, w = image.shape
        new_h = h - (h % crop_to_multiple_of)
        new_w = w - (w % crop_to_multiple_of)
        image = image[:new_h, :new_w]
        
    return np.ascontiguousarray(image)

def run_py_extraction(image_array, bg_sub_mode, sigma_mode, downsample):
    # Handle the "None" string gracefully for Python logic
    if bg_sub_mode == "None":
        bg_sub_mode = None
    if downsample == 0:
        downsample = None

    # return_moments=True returns: (centroids, [sum, area, moments, axis_ratio])
    result = get_centroids_from_image(
        image_array, 
        bg_sub_mode=bg_sub_mode, 
        sigma_mode=sigma_mode,
        downsample=downsample,
        return_moments=True
    )
    
    # tetra3.py returns a 5-tuple if 0 stars are found, but a 2-tuple containing a list if >0 stars are found.
    # We must explicitly handle this empty case to prevent unpacking errors.
    if len(result) == 5:
        centroids = result[0]
        moments = result[1:] 
    else:
        centroids, moments = result
        
    # Force float64 arrays AND strictly enforce dimensionality to prevent PyO3 TypeError panics.
    # tetra3.py inconsistently returns shape (0, 1) [2D] for 0-star runs instead of (0,) [1D].
    return (
        np.asarray(centroids, dtype=np.float64).reshape(-1, 2),
        [
            np.asarray(moments[0], dtype=np.float64).flatten(),    # sum -> 1D
            np.asarray(moments[1], dtype=np.float64).flatten(),    # area -> 1D
            np.asarray(moments[2], dtype=np.float64).reshape(-1, 3), # moments -> 2D
            np.asarray(moments[3], dtype=np.float64).flatten()     # axis_ratio -> 1D
        ]
    )

def run_py_extraction_perf(image_array, downsample):
    if downsample == 0:
        downsample = None
    # Execute the extraction but do not allocate wrapper arrays or return data over the PyO3 boundary.
    # This ensures the timer strictly captures the algorithmic execution time.
    get_centroids_from_image(image_array, downsample=downsample, return_moments=True)
"#;

/// Helper function to locate and return the test images, maximizing code reuse across tests.
fn get_test_images() -> Vec<PathBuf> {
    // Assuming this test runs from the `tetra3` crate root, and `cedar-solve`
    // is cloned adjacent to the parent directory.
    let test_dir = "../../cedar-solve/examples/data/medium_fov";
    let mut image_paths = Vec::new();

    if Path::new(test_dir).exists() {
        for entry in WalkDir::new(test_dir).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if ext_str == "jpg" || ext_str == "jpeg" || ext_str == "tiff" {
                    image_paths.push(path.to_path_buf());
                }
            }
        }
    }

    assert!(
        !image_paths.is_empty(),
        "No test images found in {}! Ensure cedar-solve is cloned in the correct relative location.",
        test_dir
    );

    image_paths
}

/// Core validation logic that runs the Rust extractor against the Python extractor
/// for a specific set of background, sigma modes, and downsampling values.
fn run_validation_suite(
    bg_modes: &[(Option<tetra3::extractor::BgSubMode>, &str)],
    sigma_modes: &[(tetra3::extractor::SigmaMode, &str)],
    downsamples: &[Option<usize>],
) {
    let mut total_rust_time = Duration::ZERO;
    let mut total_py_time = Duration::ZERO;
    let mut image_count = 0;

    let image_paths = get_test_images();

    Python::with_gil(|py| {
        let module =
            PyModule::from_code(py, PY_HELPER_CODE, "test_helper.py", "test_helper").unwrap();
        let load_image_as_array = module.getattr("load_image_as_array").unwrap();
        let run_py_extraction = module.getattr("run_py_extraction").unwrap();

        for image_path in &image_paths {
            let img_name = image_path.file_name().unwrap();
            println!("\nTesting image: {:?}", img_name);

            for &ds_opt in downsamples {
                let ds_val = ds_opt.unwrap_or(1);

                // 1. Load the image into Python and get the raw f32 numpy array.
                // We pass the downsample factor so Python can crop remainder pixels
                let py_image_array: &PyAny = load_image_as_array
                    .call1((image_path.to_str().unwrap(), ds_val))
                    .unwrap();

                // Extract the Array2<f32> for Rust.
                // We do a manual copy into a Vec to bridge the gap between `numpy`'s ndarray v0.15
                // and the main crate's ndarray v0.17, avoiding version-conflict compile errors.
                let py_readonly_img: PyReadonlyArray2<f32> = py_image_array.extract().unwrap();
                let py_view = py_readonly_img.as_array();
                let nrows = py_view.shape()[0];
                let ncols = py_view.shape()[1];

                let mut vec_data = Vec::with_capacity(nrows * ncols);
                for r in 0..nrows {
                    for c in 0..ncols {
                        vec_data.push(*py_view.get([r, c]).unwrap());
                    }
                }
                let rust_input_img: Array2<f32> =
                    Array2::from_shape_vec((nrows, ncols), vec_data).unwrap();

                let py_ds = ds_opt.unwrap_or(0);

                // Iterate through every combination of modes
                for (bg_rs, bg_py) in bg_modes {
                    for (sig_rs, sig_py) in sigma_modes {
                        let mut extractor =
                            tetra3::extractor::TetraExtractor::new(CentroidConfig {
                                bg_sub_mode: *bg_rs,
                                sigma_mode: *sig_rs,
                                downsample: ds_opt,
                                return_images: false,
                                ..Default::default()
                            });

                        // 2. Run Python Algorithm
                        println!(
                            "  -> Running Python extraction (bg: {}, sig: {}, ds: {:?})...",
                            bg_py, sig_py, ds_opt
                        );
                        let start_py = Instant::now();
                        let py_result = run_py_extraction
                            .call1((py_image_array, *bg_py, *sig_py, py_ds))
                            .unwrap();
                        total_py_time += start_py.elapsed();

                        // 3. Run Rust Algorithm
                        println!(
                            "  -> Running Rust extraction (bg: {}, sig: {}, ds: {:?})...",
                            bg_py, sig_py, ds_opt
                        );
                        let start_rust = Instant::now();
                        let rust_result = extractor.extract(&rust_input_img);
                        total_rust_time += start_rust.elapsed();

                        image_count += 1;

                        // 4. Parse Python Results using safe indexing methods
                        let py_tuple: &PyTuple = py_result.downcast().unwrap();
                        let py_centroids_arr: PyReadonlyArray2<f64> =
                            py_tuple.get_item(0).unwrap().extract().unwrap();

                        let py_moments_list: &PyList =
                            py_tuple.get_item(1).unwrap().downcast().unwrap();
                        let py_sum_arr: PyReadonlyArray1<f64> =
                            py_moments_list.get_item(0).unwrap().extract().unwrap();
                        let py_area_arr: PyReadonlyArray1<f64> =
                            py_moments_list.get_item(1).unwrap().extract().unwrap();
                        let py_m2_arr: PyReadonlyArray2<f64> =
                            py_moments_list.get_item(2).unwrap().extract().unwrap();
                        let py_ratio_arr: PyReadonlyArray1<f64> =
                            py_moments_list.get_item(3).unwrap().extract().unwrap();

                        let py_centroids_view = py_centroids_arr.as_array();
                        let py_sum = py_sum_arr.as_array();
                        let py_area = py_area_arr.as_array();
                        let py_m2_view = py_m2_arr.as_array();
                        let py_ratio = py_ratio_arr.as_array();

                        // --- STABILIZE SORTING FOR TIES ---
                        // Floating point math accumulation differs slightly between Rust and Python.
                        // We must quantize the values so identical sums that differ by 0.0000001
                        // are treated as perfect ties, allowing the logic to fall through to the Y and X coordinates.
                        let q_sum = |v: f64| (v * 1000.0).round() as i64;
                        let q_coord = |v: f64| (v * 10.0).round() as i64;

                        let mut rust_centroids_sorted = rust_result.centroids.clone();
                        rust_centroids_sorted.sort_by(|a, b| {
                            q_sum(b.sum)
                                .cmp(&q_sum(a.sum))
                                .then_with(|| q_coord(a.y).cmp(&q_coord(b.y)))
                                .then_with(|| q_coord(a.x).cmp(&q_coord(b.x)))
                        });

                        let mut py_indices: Vec<usize> =
                            (0..py_centroids_view.shape()[0]).collect();
                        py_indices.sort_by(|&i, &j| {
                            let sum_i = py_sum.get(i).unwrap();
                            let sum_j = py_sum.get(j).unwrap();
                            let y_i = py_centroids_view.get([i, 0]).unwrap();
                            let y_j = py_centroids_view.get([j, 0]).unwrap();
                            let x_i = py_centroids_view.get([i, 1]).unwrap();
                            let x_j = py_centroids_view.get([j, 1]).unwrap();

                            q_sum(*sum_j)
                                .cmp(&q_sum(*sum_i))
                                .then_with(|| q_coord(*y_i).cmp(&q_coord(*y_j)))
                                .then_with(|| q_coord(*x_i).cmp(&q_coord(*x_j)))
                        });

                        let eps = 1e-3;
                        let mut mismatch_detected = false;

                        // 5. Compare Results (Pre-check for logging block)
                        if rust_centroids_sorted.len() != py_indices.len() {
                            mismatch_detected = true;
                        } else {
                            // Check for data mismatch
                            for i in 0..rust_centroids_sorted.len() {
                                let r_c = &rust_centroids_sorted[i];
                                let py_idx = py_indices[i];
                                let p_y = *py_centroids_view.get([py_idx, 0]).unwrap();
                                let p_x = *py_centroids_view.get([py_idx, 1]).unwrap();
                                let p_sum = *py_sum.get(py_idx).unwrap();

                                if (r_c.y - p_y).abs() >= eps
                                    || (r_c.x - p_x).abs() >= eps
                                    || (r_c.sum - p_sum).abs() / p_sum.max(1.0) >= 1e-4
                                {
                                    mismatch_detected = true;
                                    break;
                                }
                            }
                        }

                        // --- LOGGING BLOCK ---
                        if mismatch_detected {
                            let total_len = rust_centroids_sorted.len().max(py_indices.len());

                            // Find the exact index where the divergence starts
                            let mut first_mismatch_idx = 0;
                            for i in 0..total_len {
                                let r_c_match = rust_centroids_sorted.get(i);
                                let py_idx_match = py_indices.get(i);

                                let mut row_mismatch = false;
                                if let (Some(r), Some(&p_idx)) = (r_c_match, py_idx_match) {
                                    let p_y = *py_centroids_view.get([p_idx, 0]).unwrap();
                                    let p_x = *py_centroids_view.get([p_idx, 1]).unwrap();
                                    let p_sum = *py_sum.get(p_idx).unwrap();

                                    if (r.y - p_y).abs() >= eps
                                        || (r.x - p_x).abs() >= eps
                                        || (r.sum - p_sum).abs() / p_sum.max(1.0) >= 1e-4
                                    {
                                        row_mismatch = true;
                                    }
                                } else {
                                    row_mismatch = true; // One list is shorter than the other
                                }

                                if row_mismatch {
                                    first_mismatch_idx = i;
                                    break;
                                }
                            }

                            println!("\n=======================================================");
                            println!(
                                "MISMATCH DETECTED: [{:?} | bg: {}, sig: {}, ds: {:?}]",
                                img_name, bg_py, sig_py, ds_opt
                            );
                            println!(
                                "Rust total: {}, Py total: {}",
                                rust_centroids_sorted.len(),
                                py_indices.len()
                            );
                            println!(
                                "Showing window around first failure at index {}",
                                first_mismatch_idx
                            );
                            println!(
                                "{:<5} | {:<30} | {:<30}",
                                "Index", "Rust (Y, X, Sum, Area)", "Python (Y, X, Sum, Area)"
                            );
                            println!("-------------------------------------------------------");

                            // Print a window: 5 rows before the mismatch, 15 rows after
                            let start_idx = first_mismatch_idx.saturating_sub(5);
                            let end_idx = (first_mismatch_idx + 15).min(total_len);

                            if start_idx > 0 {
                                println!("...   | ...                            | ...");
                            }

                            for i in start_idx..end_idx {
                                let r_str = if i < rust_centroids_sorted.len() {
                                    let c = &rust_centroids_sorted[i];
                                    format!("({:.1}, {:.1}, {:.1}, {})", c.y, c.x, c.sum, c.area)
                                } else {
                                    "NONE".to_string()
                                };

                                let p_str = if i < py_indices.len() {
                                    let py_idx = py_indices[i];
                                    let p_y = *py_centroids_view.get([py_idx, 0]).unwrap();
                                    let p_x = *py_centroids_view.get([py_idx, 1]).unwrap();
                                    let p_sum = *py_sum.get(py_idx).unwrap();
                                    let p_area = *py_area.get(py_idx).unwrap() as usize;
                                    format!("({:.1}, {:.1}, {:.1}, {})", p_y, p_x, p_sum, p_area)
                                } else {
                                    "NONE".to_string()
                                };

                                let marker = if r_str != p_str { " * " } else { "   " };
                                println!("{:<5}{}| {:<30} | {:<30}", i, marker, r_str, p_str);
                            }

                            if end_idx < total_len {
                                println!("...   | ...                            | ...");
                            }
                            println!("=======================================================\n");
                        }

                        // Proceed with standard assertions so the test still explicitly fails
                        assert_eq!(
                            rust_centroids_sorted.len(),
                            py_indices.len(),
                            "[{:?} | bg: {}, sig: {}, ds: {:?}] Length mismatch",
                            img_name,
                            bg_py,
                            sig_py,
                            ds_opt
                        );

                        for i in 0..rust_centroids_sorted.len() {
                            let r_c = &rust_centroids_sorted[i];
                            let py_idx = py_indices[i];

                            let p_y = *py_centroids_view.get([py_idx, 0]).unwrap();
                            let p_x = *py_centroids_view.get([py_idx, 1]).unwrap();
                            let p_sum = *py_sum.get(py_idx).unwrap();
                            let p_area = *py_area.get(py_idx).unwrap() as usize;

                            // m2_xx is at index 0, m2_yy is at index 1
                            let p_m2_xx = *py_m2_view.get([py_idx, 0]).unwrap();
                            let p_m2_yy = *py_m2_view.get([py_idx, 1]).unwrap();
                            let p_m2_xy = *py_m2_view.get([py_idx, 2]).unwrap();

                            let p_ratio = *py_ratio.get(py_idx).unwrap();

                            assert!(
                                (r_c.y - p_y).abs() < eps,
                                "[{:?} | bg: {}, sig: {}, ds: {:?}] Y mismatch at sorted index {}: Rust {} vs Py {}",
                                img_name,
                                bg_py,
                                sig_py,
                                ds_opt,
                                i,
                                r_c.y,
                                p_y
                            );
                            assert!(
                                (r_c.x - p_x).abs() < eps,
                                "[{:?} | bg: {}, sig: {}, ds: {:?}] X mismatch at sorted index {}: Rust {} vs Py {}",
                                img_name,
                                bg_py,
                                sig_py,
                                ds_opt,
                                i,
                                r_c.x,
                                p_x
                            );
                            assert!(
                                (r_c.sum - p_sum).abs() / p_sum.max(1.0) < 1e-4,
                                "[{:?} | bg: {}, sig: {}, ds: {:?}] Sum mismatch at sorted index {} (y:{:.1}, x:{:.1}): Rust {} vs Py {}",
                                img_name,
                                bg_py,
                                sig_py,
                                ds_opt,
                                i,
                                r_c.y,
                                r_c.x,
                                r_c.sum,
                                p_sum
                            );
                            assert_eq!(
                                r_c.area, p_area,
                                "[{:?} | bg: {}, sig: {}, ds: {:?}] Area mismatch at sorted index {} (y:{:.1}, x:{:.1}): Rust {} vs Py {}",
                                img_name, bg_py, sig_py, ds_opt, i, r_c.y, r_c.x, r_c.area, p_area
                            );

                            // Higher order moments can be NaN in Python if area constraints fail
                            if !p_m2_xx.is_nan() && !r_c.m2_xx.is_nan() {
                                assert!(
                                    (r_c.m2_xx - p_m2_xx).abs() < eps,
                                    "[{:?} | bg: {}, sig: {}, ds: {:?}] m2_xx mismatch at sorted index {} (y:{:.1}, x:{:.1}): Rust {} vs Py {}",
                                    img_name,
                                    bg_py,
                                    sig_py,
                                    ds_opt,
                                    i,
                                    r_c.y,
                                    r_c.x,
                                    r_c.m2_xx,
                                    p_m2_xx
                                );
                                assert!(
                                    (r_c.m2_yy - p_m2_yy).abs() < eps,
                                    "[{:?} | bg: {}, sig: {}, ds: {:?}] m2_yy mismatch at sorted index {} (y:{:.1}, x:{:.1}): Rust {} vs Py {}",
                                    img_name,
                                    bg_py,
                                    sig_py,
                                    ds_opt,
                                    i,
                                    r_c.y,
                                    r_c.x,
                                    r_c.m2_yy,
                                    p_m2_yy
                                );
                                assert!(
                                    (r_c.m2_xy - p_m2_xy).abs() < eps,
                                    "[{:?} | bg: {}, sig: {}, ds: {:?}] m2_xy mismatch at sorted index {} (y:{:.1}, x:{:.1}): Rust {} vs Py {}",
                                    img_name,
                                    bg_py,
                                    sig_py,
                                    ds_opt,
                                    i,
                                    r_c.y,
                                    r_c.x,
                                    r_c.m2_xy,
                                    p_m2_xy
                                );
                                assert!(
                                    (r_c.axis_ratio - p_ratio).abs() < eps,
                                    "[{:?} | bg: {}, sig: {}, ds: {:?}] Axis ratio mismatch at sorted index {} (y:{:.1}, x:{:.1}): Rust {} vs Py {}",
                                    img_name,
                                    bg_py,
                                    sig_py,
                                    ds_opt,
                                    i,
                                    r_c.y,
                                    r_c.x,
                                    r_c.axis_ratio,
                                    p_ratio
                                );
                            }
                        }
                    }
                }
            }
        }
    });
}

#[test]
fn test_extraction_against_python_sanity() {
    // Tests only the default extraction modes to ensure the base path works.
    let bg_modes = [(Some(tetra3::extractor::BgSubMode::LocalMean), "local_mean")];
    let sigma_modes = [(
        tetra3::extractor::SigmaMode::GlobalRootSquare,
        "global_root_square",
    )];
    let downsamples = [None];

    run_validation_suite(&bg_modes, &sigma_modes, &downsamples);
}

#[test]
fn test_extraction_against_python_binning() {
    // Tests the 2x2 and 4x4 downsampling binning logic against Python extraction
    let bg_modes = [(Some(tetra3::extractor::BgSubMode::LocalMean), "local_mean")];
    let sigma_modes = [(
        tetra3::extractor::SigmaMode::GlobalRootSquare,
        "global_root_square",
    )];
    // No more skipping! The Python loader automatically aligns dimensions.
    let downsamples = [Some(2), Some(4)];

    run_validation_suite(&bg_modes, &sigma_modes, &downsamples);
}

#[test]
#[ignore]
fn test_extraction_against_python_full() {
    // Tests all permutations of background subtraction and sigma thresholding.
    // This test takes > 15 minutes on a Raspberry Pi 5
    let bg_modes = [
        (
            Some(tetra3::extractor::BgSubMode::LocalMedian),
            "local_median",
        ),
        (Some(tetra3::extractor::BgSubMode::LocalMean), "local_mean"),
        (
            Some(tetra3::extractor::BgSubMode::GlobalMedian),
            "global_median",
        ),
        (
            Some(tetra3::extractor::BgSubMode::GlobalMean),
            "global_mean",
        ),
        (None, "None"),
    ];

    let sigma_modes = [
        (
            tetra3::extractor::SigmaMode::LocalMedianAbs,
            "local_median_abs",
        ),
        (
            tetra3::extractor::SigmaMode::LocalRootSquare,
            "local_root_square",
        ),
        (
            tetra3::extractor::SigmaMode::GlobalMedianAbs,
            "global_median_abs",
        ),
        (
            tetra3::extractor::SigmaMode::GlobalRootSquare,
            "global_root_square",
        ),
    ];
    let downsamples = [None];

    run_validation_suite(&bg_modes, &sigma_modes, &downsamples);
}

#[test]
fn test_performance_vs_python() {
    let mut total_rust_time = Duration::ZERO;
    let mut total_py_time = Duration::ZERO;
    let iterations = 10;
    let image_paths = get_test_images();

    // Centralized downsample control for performance tests
    let ds_opt = None;
    let ds_val = ds_opt.unwrap_or(1);
    let py_ds_arg = ds_opt.unwrap_or(0);

    // Initialize global buffer to prevent OS allocation overhead during benchmarking
    let mut tetra_extractor = TetraExtractor::new(CentroidConfig {
        downsample: ds_opt,
        return_images: false,
        ..Default::default()
    });

    Python::with_gil(|py| {
        let module =
            PyModule::from_code(py, PY_HELPER_CODE, "test_helper.py", "test_helper").unwrap();
        let load_image_as_array = module.getattr("load_image_as_array").unwrap();
        let run_py_extraction_perf = module.getattr("run_py_extraction_perf").unwrap();

        // Preload images to ensure disk I/O and numpy construction isn't counted in benchmarking loops
        let mut preloaded_images = Vec::new();
        for path in &image_paths {
            let py_image_array: &PyAny = load_image_as_array
                .call1((path.to_str().unwrap(), ds_val))
                .unwrap();

            let py_readonly_img: PyReadonlyArray2<f32> = py_image_array.extract().unwrap();
            let py_view = py_readonly_img.as_array();
            let rust_input_img: Array2<f32> =
                Array2::from_shape_fn((py_view.nrows(), py_view.ncols()), |(y, x)| py_view[[y, x]]);

            preloaded_images.push((py_image_array, rust_input_img));
        }

        println!(
            "Running {} iterations vs Python for accurate benchmarking...",
            iterations
        );

        for _ in 0..iterations {
            for (py_image_array, rust_input_img) in &preloaded_images {
                // Run Python Algorithm
                let start_py = Instant::now();
                let _py_result = run_py_extraction_perf
                    .call1((*py_image_array, py_ds_arg))
                    .unwrap();
                total_py_time += start_py.elapsed();

                // Run Rust Algorithm (Tetra3 Port)
                let start_rust = Instant::now();
                let _rust_result = tetra_extractor.extract(rust_input_img);
                total_rust_time += start_rust.elapsed();
            }
        }
    });

    let image_count = iterations * image_paths.len();

    println!("\n==============================================");
    println!("PERFORMANCE REPORT: RUST VS PYTHON");
    println!(
        "Images tested: {} ({} unique x {} iterations)",
        image_count,
        image_paths.len(),
        iterations
    );
    println!("----------------------------------------------");
    println!("Python (tetra3) Total Time : {:.2?}", total_py_time);
    if image_count > 0 {
        println!(
            "Python (tetra3) Avg/Image  : {:.2?}",
            total_py_time / image_count as u32
        );
    }
    println!("----------------------------------------------");
    println!("Rust (Port) Total Time     : {:.2?}", total_rust_time);
    if image_count > 0 {
        println!(
            "Rust (Port) Avg/Image      : {:.2?}",
            total_rust_time / image_count as u32
        );
    }
    println!("----------------------------------------------");

    let speedup = total_py_time.as_secs_f64() / total_rust_time.as_secs_f64();
    println!("Rust Speedup               : {:.2}x", speedup);
    println!("==============================================\n");
}

#[test]
fn test_performance_vs_cedar() {
    let iterations = 50;
    let image_paths = get_test_images();
    let downsamples = [None, Some(2), Some(4)];

    for &ds_opt in &downsamples {
        let mut total_rust_time = Duration::ZERO;
        let mut total_cedar_time = Duration::ZERO;

        let ds_val = ds_opt.unwrap_or(1);
        let cedar_ds = ds_opt.unwrap_or(1) as u32;

        let mut tetra_extractor = TetraExtractor::new(CentroidConfig {
            downsample: ds_opt,
            return_images: false,
            ..Default::default()
        });

        // Use Python loader to ensure perfectly identical f32-to-Luma u8 pixel math
        // across both algorithms prior to benchmarking.
        let mut preloaded_images = Vec::new();
        Python::with_gil(|py| {
            let module =
                PyModule::from_code(py, PY_HELPER_CODE, "test_helper.py", "test_helper").unwrap();
            let load_image_as_array = module.getattr("load_image_as_array").unwrap();

            for path in &image_paths {
                let py_image_array: &PyAny = load_image_as_array
                    .call1((path.to_str().unwrap(), ds_val))
                    .unwrap();

                let py_readonly_img: PyReadonlyArray2<f32> = py_image_array.extract().unwrap();
                let py_view = py_readonly_img.as_array();
                let rust_input_img: Array2<f32> =
                    Array2::from_shape_fn((py_view.nrows(), py_view.ncols()), |(y, x)| {
                        py_view[[y, x]]
                    });

                let (height, width) = (rust_input_img.nrows(), rust_input_img.ncols());
                let mut cedar_img = GrayImage::new(width as u32, height as u32);
                for y in 0..height {
                    for x in 0..width {
                        let p_val = rust_input_img[[y, x]].clamp(0.0, 255.0) as u8;
                        cedar_img.put_pixel(x as u32, y as u32, image::Luma([p_val]));
                    }
                }

                preloaded_images.push((rust_input_img, cedar_img));
            }
        });

        println!(
            "Running {} iterations vs Cedar-Detect (Downsample: {:?})...",
            iterations, ds_opt
        );

        for _ in 0..iterations {
            for (rust_input_img, cedar_img) in &preloaded_images {
                // Run Rust Algorithm (Tetra3 Port)
                let start_rust = Instant::now();
                let _rust_result = tetra_extractor.extract(rust_input_img);
                total_rust_time += start_rust.elapsed();

                // Run Cedar Detect Algorithm
                let start_cedar = Instant::now();
                let noise_estimate = estimate_noise_from_image(cedar_img);
                let _cedar_result = get_stars_from_image(
                    cedar_img,
                    noise_estimate,
                    8.0,
                    false,
                    cedar_ds,
                    true,
                    false,
                );
                total_cedar_time += start_cedar.elapsed();
            }
        }

        let image_count = iterations * image_paths.len();

        println!("\n==============================================");
        println!(
            "PERFORMANCE REPORT: RUST PORT VS CEDAR-DETECT (DS: {:?})",
            ds_opt
        );
        println!(
            "Images tested: {} ({} unique x {} iterations)",
            image_count,
            image_paths.len(),
            iterations
        );
        println!("----------------------------------------------");
        println!("Rust (Port) Total Time     : {:.2?}", total_rust_time);
        if image_count > 0 {
            println!(
                "Rust (Port) Avg/Image      : {:.2?}",
                total_rust_time / image_count as u32
            );
        }
        println!("----------------------------------------------");
        println!("Rust (CedarDetect) Total   : {:.2?}", total_cedar_time);
        if image_count > 0 {
            println!(
                "Rust (CedarDetect) Avg     : {:.2?}",
                total_cedar_time / image_count as u32
            );
        }
        println!("----------------------------------------------");
        let speedup = total_cedar_time.as_secs_f64() / total_rust_time.as_secs_f64();
        println!("Port Speedup vs Cedar      : {:.2}x", speedup);
        println!("==============================================\n");
    }
}
