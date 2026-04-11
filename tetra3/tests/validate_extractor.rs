// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use image::GenericImageView;
use ndarray::Array2;
use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList, PyListMethods, PyModule, PyTuple, PyTupleMethods};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use walkdir::WalkDir;
use zip::write::FileOptions;
use zip::{CompressionMethod, ZipWriter};

use cedar_detect::algorithm::{estimate_noise_from_image, get_stars_from_image};
use tetra3::fast_extractor::{
    FastBgSubMode, FastDownsample, FastExtractOptions, FastExtractor, FastSigmaMode,
};
use tetra3::{ExtractOptions, Extractor, SolveOptions, SolveStatus, Solver};

// Use Rust 1.77+ c"..." literals for PyO3 0.21+ CStr requirements
const PY_HELPER_CODE: &std::ffi::CStr = cr#"
import numpy as np
from tetra3.tetra3 import get_centroids_from_image

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

#[derive(Debug, Serialize, Deserialize)]
pub struct PyCentroidResult {
    pub y: f64,
    pub x: f64,
    pub sum: f64,
    pub area: usize,
    pub m2_xx: f64,
    pub m2_yy: f64,
    pub m2_xy: f64,
    pub axis_ratio: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PyImageResult {
    pub image_name: String,
    pub centroids: Vec<PyCentroidResult>,
}

/// Helper function to locate and return the test images, maximizing code reuse across tests.
fn get_test_images() -> Vec<PathBuf> {
    let test_dir = "tests/fixtures/sample_images";
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
        "No test images found in {}!",
        test_dir
    );

    image_paths
}

/// Core validation logic that runs the Rust extractor against the pre-generated Python
/// ground-truth zip fixtures for a specific set of background, sigma modes, and downsampling.
fn run_validation_suite_from_fixtures(
    bg_modes: &[(Option<tetra3::extractor::BgSubMode>, &str)],
    sigma_modes: &[(tetra3::extractor::SigmaMode, &str)],
    downsamples: &[Option<usize>],
) {
    let image_paths = get_test_images();

    // Pre-load base images into memory to ensure disk I/O and pixel casting
    // are not counted in the total extraction benchmarking time.
    let mut base_images = Vec::new();
    for path in &image_paths {
        let img = image::open(path).unwrap();
        let img_name = path.file_name().unwrap().to_string_lossy().to_string();
        base_images.push((img_name, img));
    }

    for &ds_opt in downsamples {
        let ds_val = ds_opt.unwrap_or(1);
        let ds_str = ds_opt
            .map(|d| d.to_string())
            .unwrap_or_else(|| "none".to_string());

        for (bg_rs, bg_py) in bg_modes {
            for (sig_rs, sig_py) in sigma_modes {
                let zip_name = format!("py-{}-{}-{}.zip", bg_py.to_lowercase(), sig_py, ds_str);
                let zip_path = PathBuf::from("tests/fixtures").join(&zip_name);

                println!("\nValidating configuration against fixture: {}", zip_name);

                if !zip_path.exists() {
                    panic!(
                        "Fixture {} not found! Run `cargo test generate_python_test_fixtures --release -- --ignored` first.",
                        zip_name
                    );
                }

                // 1. Read JSON ground truth from Zip
                let file = File::open(&zip_path).unwrap();
                let mut archive = zip::ZipArchive::new(file).unwrap();
                let mut results_file = archive.by_name("results.json").unwrap();
                let py_results: Vec<PyImageResult> =
                    serde_json::from_reader(&mut results_file).unwrap();

                let mut total_rust_time = Duration::ZERO;

                for (img_name, base_img) in &base_images {
                    let py_img_res = py_results
                        .iter()
                        .find(|r| r.image_name == *img_name)
                        .unwrap();

                    // CRITICAL FIX: Truncate remainder pixels for safe downsampling.
                    let (w, h) = base_img.dimensions();
                    let new_w = w - (w % ds_val as u32);
                    let new_h = h - (h % ds_val as u32);

                    let mut rust_input_img = Array2::<f32>::zeros((new_h as usize, new_w as usize));
                    let luma_img = base_img.to_luma8();

                    for y in 0..new_h {
                        for x in 0..new_w {
                            rust_input_img[[y as usize, x as usize]] =
                                luma_img.get_pixel(x, y)[0] as f32;
                        }
                    }

                    let mut extractor = Extractor::new();
                    let options = ExtractOptions {
                        bg_sub_mode: *bg_rs,
                        sigma_mode: *sig_rs,
                        downsample: ds_opt,
                        return_images: false,
                        ..Default::default()
                    };

                    // 2. Run Rust Algorithm
                    let start_rust = Instant::now();
                    let rust_result = extractor.extract(&rust_input_img, options);
                    total_rust_time += start_rust.elapsed();

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

                    let mut py_centroids_sorted = py_img_res.centroids.iter().collect::<Vec<_>>();
                    py_centroids_sorted.sort_by(|a, b| {
                        q_sum(b.sum)
                            .cmp(&q_sum(a.sum))
                            .then_with(|| q_coord(a.y).cmp(&q_coord(b.y)))
                            .then_with(|| q_coord(a.x).cmp(&q_coord(b.x)))
                    });

                    let eps = 1e-3;
                    let mut mismatch_detected = false;

                    // 3. Compare Results (Pre-check for logging block)
                    if rust_centroids_sorted.len() != py_centroids_sorted.len() {
                        mismatch_detected = true;
                    } else {
                        for i in 0..rust_centroids_sorted.len() {
                            let r_c = &rust_centroids_sorted[i];
                            let p_c = py_centroids_sorted[i];

                            if (r_c.y - p_c.y).abs() >= eps
                                || (r_c.x - p_c.x).abs() >= eps
                                || (r_c.sum - p_c.sum).abs() / p_c.sum.max(1.0) >= 1e-4
                            {
                                mismatch_detected = true;
                                break;
                            }
                        }
                    }

                    // --- LOGGING BLOCK ---
                    if mismatch_detected {
                        let total_len = rust_centroids_sorted.len().max(py_centroids_sorted.len());

                        // Find the exact index where the divergence starts
                        let mut first_mismatch_idx = 0;
                        for i in 0..total_len {
                            let r_c_match = rust_centroids_sorted.get(i);
                            let p_c_match = py_centroids_sorted.get(i);

                            let mut row_mismatch = false;
                            if let (Some(r), Some(&p)) = (r_c_match, p_c_match) {
                                if (r.y - p.y).abs() >= eps
                                    || (r.x - p.x).abs() >= eps
                                    || (r.sum - p.sum).abs() / p.sum.max(1.0) >= 1e-4
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
                            py_centroids_sorted.len()
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

                            let p_str = if i < py_centroids_sorted.len() {
                                let p = py_centroids_sorted[i];
                                format!("({:.1}, {:.1}, {:.1}, {})", p.y, p.x, p.sum, p.area)
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
                        py_centroids_sorted.len(),
                        "[{:?} | bg: {}, sig: {}, ds: {:?}] Length mismatch",
                        img_name,
                        bg_py,
                        sig_py,
                        ds_opt
                    );

                    for i in 0..rust_centroids_sorted.len() {
                        let r_c = &rust_centroids_sorted[i];
                        let p_c = py_centroids_sorted[i];

                        assert!(
                            (r_c.y - p_c.y).abs() < eps,
                            "[{:?} | bg: {}, sig: {}, ds: {:?}] Y mismatch at sorted index {}: Rust {} vs Py {}",
                            img_name,
                            bg_py,
                            sig_py,
                            ds_opt,
                            i,
                            r_c.y,
                            p_c.y
                        );
                        assert!(
                            (r_c.x - p_c.x).abs() < eps,
                            "[{:?} | bg: {}, sig: {}, ds: {:?}] X mismatch at sorted index {}: Rust {} vs Py {}",
                            img_name,
                            bg_py,
                            sig_py,
                            ds_opt,
                            i,
                            r_c.x,
                            p_c.x
                        );
                        assert!(
                            (r_c.sum - p_c.sum).abs() / p_c.sum.max(1.0) < 1e-4,
                            "[{:?} | bg: {}, sig: {}, ds: {:?}] Sum mismatch at sorted index {} (y:{:.1}, x:{:.1}): Rust {} vs Py {}",
                            img_name,
                            bg_py,
                            sig_py,
                            ds_opt,
                            i,
                            r_c.y,
                            r_c.x,
                            r_c.sum,
                            p_c.sum
                        );
                        assert_eq!(
                            r_c.area, p_c.area,
                            "[{:?} | bg: {}, sig: {}, ds: {:?}] Area mismatch at sorted index {} (y:{:.1}, x:{:.1}): Rust {} vs Py {}",
                            img_name, bg_py, sig_py, ds_opt, i, r_c.y, r_c.x, r_c.area, p_c.area
                        );

                        // Higher order moments can be NaN in Python if area constraints fail
                        if !p_c.m2_xx.is_nan() && !r_c.m2_xx.is_nan() {
                            assert!(
                                (r_c.m2_xx - p_c.m2_xx).abs() < eps,
                                "[{:?} | bg: {}, sig: {}, ds: {:?}] m2_xx mismatch at sorted index {}: Rust {} vs Py {}",
                                img_name,
                                bg_py,
                                sig_py,
                                ds_opt,
                                i,
                                r_c.m2_xx,
                                p_c.m2_xx
                            );
                            assert!(
                                (r_c.m2_yy - p_c.m2_yy).abs() < eps,
                                "[{:?} | bg: {}, sig: {}, ds: {:?}] m2_yy mismatch at sorted index {}: Rust {} vs Py {}",
                                img_name,
                                bg_py,
                                sig_py,
                                ds_opt,
                                i,
                                r_c.m2_yy,
                                p_c.m2_yy
                            );
                            assert!(
                                (r_c.m2_xy - p_c.m2_xy).abs() < eps,
                                "[{:?} | bg: {}, sig: {}, ds: {:?}] m2_xy mismatch at sorted index {}: Rust {} vs Py {}",
                                img_name,
                                bg_py,
                                sig_py,
                                ds_opt,
                                i,
                                r_c.m2_xy,
                                p_c.m2_xy
                            );
                            assert!(
                                (r_c.axis_ratio - p_c.axis_ratio).abs() < eps,
                                "[{:?} | bg: {}, sig: {}, ds: {:?}] Axis ratio mismatch at sorted index {}: Rust {} vs Py {}",
                                img_name,
                                bg_py,
                                sig_py,
                                ds_opt,
                                i,
                                r_c.axis_ratio,
                                p_c.axis_ratio
                            );
                        }
                    }
                }

                println!(
                    "  -> Total Rust Extraction Time ({} images): {:.2?}",
                    base_images.len(),
                    total_rust_time
                );
            }
        }
    }
}

#[test]
fn test_extraction_against_python_sanity() {
    // Tests only the default extraction modes to ensure the base path works against fixtures.
    let bg_modes = [(Some(tetra3::extractor::BgSubMode::LocalMean), "local_mean")];
    let sigma_modes = [(
        tetra3::extractor::SigmaMode::GlobalRootSquare,
        "global_root_square",
    )];
    let downsamples = [None];

    run_validation_suite_from_fixtures(&bg_modes, &sigma_modes, &downsamples);
}

#[test]
fn test_extraction_against_python_full() {
    // Tests all permutations of background subtraction, sigma thresholding, and downsampling.
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

    let downsamples = [None, Some(2), Some(4)];

    run_validation_suite_from_fixtures(&bg_modes, &sigma_modes, &downsamples);
}

/// Fully instrumented u8 validation suite. Loads Python JSON ground-truths
/// and validates the integer pipeline identically to the f32 pipeline.
fn run_validation_suite_u8(
    bg_modes: &[(Option<tetra3::extractor::BgSubMode>, &str)],
    sigma_modes: &[(tetra3::extractor::SigmaMode, &str)],
    downsamples: &[Option<usize>],
) {
    let image_paths = get_test_images();

    let mut base_images = Vec::new();
    for path in &image_paths {
        let img = image::open(path).unwrap();
        let img_name = path.file_name().unwrap().to_string_lossy().to_string();
        base_images.push((img_name, img));
    }

    for &ds_opt in downsamples {
        let ds_val = ds_opt.unwrap_or(1);
        let ds_str = ds_opt
            .map(|d| d.to_string())
            .unwrap_or_else(|| "none".to_string());

        for (bg_rs, bg_py) in bg_modes {
            for (sig_rs, sig_py) in sigma_modes {
                let zip_name = format!("py-{}-{}-{}.zip", bg_py.to_lowercase(), sig_py, ds_str);
                let zip_path = PathBuf::from("tests/fixtures").join(&zip_name);

                println!(
                    "\nValidating u8 configuration against fixture: {}",
                    zip_name
                );

                if !zip_path.exists() {
                    panic!(
                        "Fixture {} not found! Run `cargo test generate_python_test_fixtures --release -- --ignored` first.",
                        zip_name
                    );
                }

                // 1. Read JSON ground truth from Zip
                let file = File::open(&zip_path).unwrap();
                let mut archive = zip::ZipArchive::new(file).unwrap();
                let mut results_file = archive.by_name("results.json").unwrap();
                let py_results: Vec<PyImageResult> =
                    serde_json::from_reader(&mut results_file).unwrap();

                let mut total_rust_time = Duration::ZERO;
                let mut centroids_found = 0;

                for (img_name, base_img) in &base_images {
                    let py_img_res = py_results
                        .iter()
                        .find(|r| r.image_name == *img_name)
                        .unwrap();

                    let (w, h) = base_img.dimensions();
                    let new_w = w - (w % ds_val as u32);
                    let new_h = h - (h % ds_val as u32);

                    let mut rust_input_img = Array2::<u8>::zeros((new_h as usize, new_w as usize));
                    let luma_img = base_img.to_luma8();

                    for y in 0..new_h {
                        for x in 0..new_w {
                            rust_input_img[[y as usize, x as usize]] = luma_img.get_pixel(x, y)[0];
                        }
                    }

                    let mut extractor = Extractor::new();
                    let options = ExtractOptions {
                        bg_sub_mode: *bg_rs,
                        sigma_mode: *sig_rs,
                        downsample: ds_opt,
                        return_images: false,
                        ..Default::default()
                    };

                    let start_rust = Instant::now();
                    let rust_result = extractor.extract_u8(&rust_input_img, options);
                    total_rust_time += start_rust.elapsed();
                    centroids_found += rust_result.centroids.len();

                    let q_sum = |v: f64| (v * 1000.0).round() as i64;
                    let q_coord = |v: f64| (v * 10.0).round() as i64;

                    let mut rust_centroids_sorted = rust_result.centroids.clone();
                    rust_centroids_sorted.sort_by(|a, b| {
                        q_sum(b.sum)
                            .cmp(&q_sum(a.sum))
                            .then_with(|| q_coord(a.y).cmp(&q_coord(b.y)))
                            .then_with(|| q_coord(a.x).cmp(&q_coord(b.x)))
                    });

                    let mut py_centroids_sorted = py_img_res.centroids.iter().collect::<Vec<_>>();
                    py_centroids_sorted.sort_by(|a, b| {
                        q_sum(b.sum)
                            .cmp(&q_sum(a.sum))
                            .then_with(|| q_coord(a.y).cmp(&q_coord(b.y)))
                            .then_with(|| q_coord(a.x).cmp(&q_coord(b.x)))
                    });

                    let eps = 1e-3;
                    let mut mismatch_detected = false;

                    if rust_centroids_sorted.len() != py_centroids_sorted.len() {
                        mismatch_detected = true;
                    } else {
                        for i in 0..rust_centroids_sorted.len() {
                            let r_c = &rust_centroids_sorted[i];
                            let p_c = py_centroids_sorted[i];

                            if (r_c.y - p_c.y).abs() >= eps
                                || (r_c.x - p_c.x).abs() >= eps
                                || (r_c.sum - p_c.sum).abs() / p_c.sum.max(1.0) >= 1e-4
                            {
                                mismatch_detected = true;
                                break;
                            }
                        }
                    }

                    if mismatch_detected {
                        let total_len = rust_centroids_sorted.len().max(py_centroids_sorted.len());
                        let mut first_mismatch_idx = 0;
                        for i in 0..total_len {
                            let r_c_match = rust_centroids_sorted.get(i);
                            let p_c_match = py_centroids_sorted.get(i);

                            let mut row_mismatch = false;
                            if let (Some(r), Some(&p)) = (r_c_match, p_c_match) {
                                if (r.y - p.y).abs() >= eps
                                    || (r.x - p.x).abs() >= eps
                                    || (r.sum - p.sum).abs() / p.sum.max(1.0) >= 1e-4
                                {
                                    row_mismatch = true;
                                }
                            } else {
                                row_mismatch = true;
                            }

                            if row_mismatch {
                                first_mismatch_idx = i;
                                break;
                            }
                        }

                        println!("\n=======================================================");
                        println!(
                            "U8 MISMATCH DETECTED: [{:?} | bg: {}, sig: {}, ds: {:?}]",
                            img_name, bg_py, sig_py, ds_opt
                        );
                        println!(
                            "Rust u8 total: {}, Py total: {}",
                            rust_centroids_sorted.len(),
                            py_centroids_sorted.len()
                        );
                        println!(
                            "Showing window around first failure at index {}",
                            first_mismatch_idx
                        );
                        println!(
                            "{:<5} | {:<30} | {:<30}",
                            "Index", "Rust u8 (Y, X, Sum, Area)", "Python (Y, X, Sum, Area)"
                        );
                        println!("-------------------------------------------------------");

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

                            let p_str = if i < py_centroids_sorted.len() {
                                let p = py_centroids_sorted[i];
                                format!("({:.1}, {:.1}, {:.1}, {})", p.y, p.x, p.sum, p.area)
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

                    assert_eq!(
                        rust_centroids_sorted.len(),
                        py_centroids_sorted.len(),
                        "[{:?} | bg: {}, sig: {}, ds: {:?}] Length mismatch",
                        img_name,
                        bg_py,
                        sig_py,
                        ds_opt
                    );

                    for i in 0..rust_centroids_sorted.len() {
                        let r_c = &rust_centroids_sorted[i];
                        let p_c = py_centroids_sorted[i];

                        assert!(
                            (r_c.y - p_c.y).abs() < eps,
                            "[{:?} | bg: {}, sig: {}, ds: {:?}] Y mismatch at sorted index {}: Rust {} vs Py {}",
                            img_name,
                            bg_py,
                            sig_py,
                            ds_opt,
                            i,
                            r_c.y,
                            p_c.y
                        );
                        assert!(
                            (r_c.x - p_c.x).abs() < eps,
                            "[{:?} | bg: {}, sig: {}, ds: {:?}] X mismatch at sorted index {}: Rust {} vs Py {}",
                            img_name,
                            bg_py,
                            sig_py,
                            ds_opt,
                            i,
                            r_c.x,
                            p_c.x
                        );
                        assert!(
                            (r_c.sum - p_c.sum).abs() / p_c.sum.max(1.0) < 1e-4,
                            "[{:?} | bg: {}, sig: {}, ds: {:?}] Sum mismatch at sorted index {} (y:{:.1}, x:{:.1}): Rust {} vs Py {}",
                            img_name,
                            bg_py,
                            sig_py,
                            ds_opt,
                            i,
                            r_c.y,
                            r_c.x,
                            r_c.sum,
                            p_c.sum
                        );
                        assert_eq!(
                            r_c.area, p_c.area,
                            "[{:?} | bg: {}, sig: {}, ds: {:?}] Area mismatch at sorted index {} (y:{:.1}, x:{:.1}): Rust {} vs Py {}",
                            img_name, bg_py, sig_py, ds_opt, i, r_c.y, r_c.x, r_c.area, p_c.area
                        );

                        if !p_c.m2_xx.is_nan() && !r_c.m2_xx.is_nan() {
                            assert!(
                                (r_c.m2_xx - p_c.m2_xx).abs() < eps,
                                "[{:?} | bg: {}, sig: {}, ds: {:?}] m2_xx mismatch at sorted index {}: Rust {} vs Py {}",
                                img_name,
                                bg_py,
                                sig_py,
                                ds_opt,
                                i,
                                r_c.m2_xx,
                                p_c.m2_xx
                            );
                            assert!(
                                (r_c.m2_yy - p_c.m2_yy).abs() < eps,
                                "[{:?} | bg: {}, sig: {}, ds: {:?}] m2_yy mismatch at sorted index {}: Rust {} vs Py {}",
                                img_name,
                                bg_py,
                                sig_py,
                                ds_opt,
                                i,
                                r_c.m2_yy,
                                p_c.m2_yy
                            );
                            assert!(
                                (r_c.m2_xy - p_c.m2_xy).abs() < eps,
                                "[{:?} | bg: {}, sig: {}, ds: {:?}] m2_xy mismatch at sorted index {}: Rust {} vs Py {}",
                                img_name,
                                bg_py,
                                sig_py,
                                ds_opt,
                                i,
                                r_c.m2_xy,
                                p_c.m2_xy
                            );
                            assert!(
                                (r_c.axis_ratio - p_c.axis_ratio).abs() < eps,
                                "[{:?} | bg: {}, sig: {}, ds: {:?}] Axis ratio mismatch at sorted index {}: Rust {} vs Py {}",
                                img_name,
                                bg_py,
                                sig_py,
                                ds_opt,
                                i,
                                r_c.axis_ratio,
                                p_c.axis_ratio
                            );
                        }
                    }
                }

                println!(
                    "  -> Total Rust u8 Extraction Time [bg: {}, sig: {}, ds: {:?}] ({} images, {} centroids): {:.2?}",
                    bg_py,
                    sig_py,
                    ds_opt,
                    base_images.len(),
                    centroids_found,
                    total_rust_time
                );
            }
        }
    }
}

#[test]
fn test_extraction_u8_sanity() {
    let bg_modes = [(Some(tetra3::extractor::BgSubMode::LocalMean), "local_mean")];
    let sigma_modes = [(
        tetra3::extractor::SigmaMode::GlobalRootSquare,
        "global_root_square",
    )];
    let downsamples = [None];

    run_validation_suite_u8(&bg_modes, &sigma_modes, &downsamples);
}

#[test]
fn test_extraction_u8_full() {
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

    let downsamples = [None, Some(2), Some(4)];

    run_validation_suite_u8(&bg_modes, &sigma_modes, &downsamples);
}

#[test]
#[ignore]
// Run intentionally via: cargo test generate_python_test_fixtures --release -- --ignored --nocapture
// This test requires the tetra3 module to be loaded in the Python environment
fn generate_python_test_fixtures() {
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

    let downsamples = [None, Some(2), Some(4)];
    let image_paths = get_test_images();

    // Create the directory where the zip files will be stored
    let fixtures_dir = Path::new("tests/fixtures");
    std::fs::create_dir_all(fixtures_dir).unwrap();

    // Ensure the freethreaded runtime is properly initialized before attaching
    Python::initialize();
    Python::attach(|py| {
        let module =
            PyModule::from_code(py, PY_HELPER_CODE, c"test_helper.py", c"test_helper").unwrap();
        let run_py_extraction = module.getattr("run_py_extraction").unwrap();

        for &ds_opt in &downsamples {
            let ds_val = ds_opt.unwrap_or(1);
            let py_ds = ds_opt.unwrap_or(0);

            // Load via Rust's image crate instead of Python's PIL
            // so we guarantee both algorithms process the exact same bytes!
            let mut loaded_rust_images = Vec::new();
            for path in &image_paths {
                let img_name = path.file_name().unwrap().to_string_lossy().to_string();
                let base_img = image::open(path).unwrap();
                let (w, h) = base_img.dimensions();
                let new_w = w - (w % ds_val as u32);
                let new_h = h - (h % ds_val as u32);

                let mut rust_input_img = Array2::<f32>::zeros((new_h as usize, new_w as usize));
                let luma_img = base_img.to_luma8();

                for y in 0..new_h {
                    for x in 0..new_w {
                        rust_input_img[[y as usize, x as usize]] =
                            luma_img.get_pixel(x, y)[0] as f32;
                    }
                }
                loaded_rust_images.push((img_name, rust_input_img));
            }

            for (_bg_rs, bg_py) in &bg_modes {
                for (_sig_rs, sig_py) in &sigma_modes {
                    let ds_str = ds_opt
                        .map(|d| d.to_string())
                        .unwrap_or_else(|| "none".to_string());

                    let zip_name = format!("py-{}-{}-{}.zip", bg_py.to_lowercase(), sig_py, ds_str);
                    let zip_path = fixtures_dir.join(&zip_name);

                    println!("Generating fixture: {}", zip_name);

                    let mut all_results = Vec::new();

                    for (img_name, rust_input_img) in &loaded_rust_images {
                        // Pass the exact Rust-generated f32 array over the PyO3 boundary.
                        // We use standard slices to completely bypass ndarray version mismatches!
                        let shape = rust_input_img.shape();
                        let py_image_array =
                            numpy::PyArray1::from_slice(py, rust_input_img.as_slice().unwrap())
                                .reshape([shape[0], shape[1]])
                                .unwrap();

                        let py_result = run_py_extraction
                            .call1((py_image_array, *bg_py, *sig_py, py_ds))
                            .unwrap();

                        let py_tuple = py_result.cast_into::<PyTuple>().unwrap();
                        let py_centroids_arr: PyReadonlyArray2<f64> =
                            py_tuple.get_item(0).unwrap().extract().unwrap();
                        let py_moments_list =
                            py_tuple.get_item(1).unwrap().cast_into::<PyList>().unwrap();

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

                        let mut image_result = PyImageResult {
                            image_name: img_name.clone(),
                            centroids: Vec::with_capacity(py_centroids_view.shape()[0]),
                        };

                        for i in 0..py_centroids_view.shape()[0] {
                            image_result.centroids.push(PyCentroidResult {
                                y: *py_centroids_view.get([i, 0]).unwrap(),
                                x: *py_centroids_view.get([i, 1]).unwrap(),
                                sum: *py_sum.get(i).unwrap(),
                                area: *py_area.get(i).unwrap() as usize,
                                m2_xx: *py_m2_view.get([i, 0]).unwrap(),
                                m2_yy: *py_m2_view.get([i, 1]).unwrap(),
                                m2_xy: *py_m2_view.get([i, 2]).unwrap(),
                                axis_ratio: *py_ratio.get(i).unwrap(),
                            });
                        }

                        all_results.push(image_result);
                    }

                    let file = File::create(&zip_path).unwrap();
                    let mut zip = ZipWriter::new(file);
                    let options =
                        FileOptions::default().compression_method(CompressionMethod::Deflated);

                    zip.start_file("results.json", options).unwrap();
                    let json_bytes = serde_json::to_vec_pretty(&all_results).unwrap();
                    zip.write_all(&json_bytes).unwrap();
                    zip.finish().unwrap();
                }
            }
        }
    });
}

#[test]
#[ignore]
// Run intentionally via: cargo test test_performance_vs_python --release -- --ignored
// This test requires the tetra3 module to be loaded in the Python environment
fn test_performance_vs_python() {
    let mut total_rust_time = Duration::ZERO;
    let mut total_rust_u8_time = Duration::ZERO;
    let mut total_py_time = Duration::ZERO;
    let iterations = 10;
    let image_paths = get_test_images();

    // Centralized downsample control for performance tests
    let ds_opt = None;
    let ds_val = ds_opt.unwrap_or(1);
    let py_ds_arg = ds_opt.unwrap_or(0);

    // Initialize global buffer to prevent OS allocation overhead during benchmarking
    let mut tetra_extractor = Extractor::new();
    let options = ExtractOptions {
        downsample: ds_opt,
        return_images: false,
        ..Default::default()
    };

    // Ensure the freethreaded runtime is properly initialized before attaching
    Python::initialize();
    Python::attach(|py| {
        let module =
            PyModule::from_code(py, PY_HELPER_CODE, c"test_helper.py", c"test_helper").unwrap();
        let run_py_extraction_perf = module.getattr("run_py_extraction_perf").unwrap();

        // Preload images to ensure disk I/O and numpy construction isn't counted in benchmarking loops
        let mut preloaded_images = Vec::new();
        for path in &image_paths {
            let base_img = image::open(path).unwrap();
            let (w, h) = base_img.dimensions();
            let new_w = w - (w % ds_val as u32);
            let new_h = h - (h % ds_val as u32);

            let mut rust_input_img = Array2::<f32>::zeros((new_h as usize, new_w as usize));
            let mut rust_input_img_u8 = Array2::<u8>::zeros((new_h as usize, new_w as usize));
            let luma_img = base_img.to_luma8();

            for y in 0..new_h {
                for x in 0..new_w {
                    let p_val = luma_img.get_pixel(x, y)[0];
                    rust_input_img[[y as usize, x as usize]] = p_val as f32;
                    rust_input_img_u8[[y as usize, x as usize]] = p_val;
                }
            }

            // Create a Python copy of the exactly equal array for the benchmark loop.
            // We use standard slices to completely bypass ndarray version mismatches!
            let py_image_array =
                numpy::PyArray1::from_slice(py, rust_input_img.as_slice().unwrap())
                    .reshape([new_h as usize, new_w as usize])
                    .unwrap();

            preloaded_images.push((py_image_array, rust_input_img, rust_input_img_u8));
        }

        println!(
            "Running {} iterations vs Python for accurate benchmarking...",
            iterations
        );

        for _ in 0..iterations {
            for (py_image_array, rust_input_img, rust_input_img_u8) in &preloaded_images {
                // Run Python Algorithm
                let start_py = Instant::now();
                let _py_result = run_py_extraction_perf
                    .call1((py_image_array.clone(), py_ds_arg))
                    .unwrap();
                total_py_time += start_py.elapsed();

                // Run Rust Algorithm (Tetra3 Port)
                let start_rust = Instant::now();
                let _rust_result = tetra_extractor.extract(rust_input_img, options.clone());
                total_rust_time += start_rust.elapsed();

                let start_rust_u8 = Instant::now();
                let _rust_result_u8 =
                    tetra_extractor.extract_u8(rust_input_img_u8, options.clone());
                total_rust_u8_time += start_rust_u8.elapsed();
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
    println!("Rust (Port f32) Total Time : {:.2?}", total_rust_time);
    if image_count > 0 {
        println!(
            "Rust (Port f32) Avg/Image  : {:.2?}",
            total_rust_time / image_count as u32
        );
    }
    println!("----------------------------------------------");
    println!("Rust (Port u8) Total Time  : {:.2?}", total_rust_u8_time);
    if image_count > 0 {
        println!(
            "Rust (Port u8) Avg/Image   : {:.2?}",
            total_rust_u8_time / image_count as u32
        );
    }
    println!("----------------------------------------------");

    let speedup = total_py_time.as_secs_f64() / total_rust_time.as_secs_f64();
    println!("Rust Speedup               : {:.2}x", speedup);
    println!("==============================================\n");
}

#[test]
#[ignore]
fn test_performance_vs_cedar() {
    let iterations = 50;
    let image_paths = get_test_images();
    let downsamples = [None, Some(2), Some(4)];

    for &ds_opt in &downsamples {
        let mut total_rust_time = Duration::ZERO;
        let mut total_rust_u8_time = Duration::ZERO;
        let mut total_rust_fast_u8_time = Duration::ZERO;
        let mut total_cedar_time = Duration::ZERO;

        let ds_val = ds_opt.unwrap_or(1);
        let cedar_ds = ds_opt.unwrap_or(1) as u32;

        let mut tetra_extractor = Extractor::new();

        let options = ExtractOptions {
            downsample: ds_opt,
            return_images: false,
            ..Default::default()
        };

        // Use the exact same loaded array for both algorithms to benchmark effectively
        let mut preloaded_images = Vec::new();
        for path in &image_paths {
            let base_img = image::open(path).unwrap();
            let (w, h) = base_img.dimensions();
            let new_w = w - (w % ds_val as u32);
            let new_h = h - (h % ds_val as u32);

            let mut rust_input_img = Array2::<f32>::zeros((new_h as usize, new_w as usize));
            let mut rust_input_img_u8 = Array2::<u8>::zeros((new_h as usize, new_w as usize));
            let mut cedar_img = image::GrayImage::new(new_w, new_h);
            let luma_img = base_img.to_luma8();

            for y in 0..new_h {
                for x in 0..new_w {
                    let p_val = luma_img.get_pixel(x, y)[0];
                    rust_input_img[[y as usize, x as usize]] = p_val as f32;
                    rust_input_img_u8[[y as usize, x as usize]] = p_val;
                    cedar_img.put_pixel(x, y, image::Luma([p_val]));
                }
            }

            let fast_ds = match ds_opt {
                None | Some(1) => FastDownsample::None,
                Some(2) => FastDownsample::X2,
                Some(4) => FastDownsample::X4,
                _ => panic!("Unsupported downsample factor"),
            };
            let fast_options = FastExtractOptions {
                downsample: fast_ds,
                ..Default::default()
            };
            let fast_extractor = FastExtractor::new(new_w as usize, new_h as usize, fast_options);

            preloaded_images.push((rust_input_img, rust_input_img_u8, cedar_img, fast_extractor));
        }

        println!(
            "Running {} iterations vs Cedar-Detect (Downsample: {:?})...",
            iterations, ds_opt
        );

        for _ in 0..iterations {
            for (rust_input_img, rust_input_img_u8, cedar_img, fast_extractor) in
                &mut preloaded_images
            {
                // Run Rust Algorithm (Tetra3 Port f32)
                let start_rust = Instant::now();
                let _rust_result = tetra_extractor.extract(rust_input_img, options.clone());
                total_rust_time += start_rust.elapsed();

                // Run Rust Algorithm (Tetra3 Port u8)
                let start_rust_u8 = Instant::now();
                let _rust_result_u8 =
                    tetra_extractor.extract_u8(rust_input_img_u8, options.clone());
                total_rust_u8_time += start_rust_u8.elapsed();

                // Run Rust Algorithm for Fast u8
                let start_rust_fast_u8 = Instant::now();
                let _rust_result_fast_u8 = fast_extractor.extract(rust_input_img_u8);
                total_rust_fast_u8_time += start_rust_fast_u8.elapsed();

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
        println!("Rust (Port f32) Total Time : {:.2?}", total_rust_time);
        if image_count > 0 {
            println!(
                "Rust (Port f32) Avg/Image  : {:.2?}",
                total_rust_time / image_count as u32
            );
        }
        println!("----------------------------------------------");
        println!("Rust (Port u8) Total Time  : {:.2?}", total_rust_u8_time);
        if image_count > 0 {
            println!(
                "Rust (Port u8) Avg/Image   : {:.2?}",
                total_rust_u8_time / image_count as u32
            );
        }

        println!("----------------------------------------------");
        println!(
            "Rust (Fast Extractor) Total Time  : {:.2?}",
            total_rust_fast_u8_time
        );
        if image_count > 0 {
            println!(
                "Rust (Fast Extractor) Avg/Image   : {:.2?}",
                total_rust_fast_u8_time / image_count as u32
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

// Struct to hold processed results to easily print two distinct tables
struct TableRow {
    img_name: String,
    port_u8_count: usize,
    port_f32_count: usize,
    fast_u8_count: usize,
    cedar_count: usize,
    match_str_u8: String,
    match_str_fast_u8: String,
    u8_solve_str: String,
    fast_u8_solve_str: String,
    cedar_solve_str: String,
    orig_f32_solve_str: String,
}

#[test]
#[ignore]
fn test_grayscale_vs_cedar() {
    let db_path = Path::new("tests/fixtures/default_database.npz");
    if !db_path.exists() {
        eprintln!("Skipping test: default_database.npz not found.");
        return;
    }
    let mut solver = Solver::load_database(db_path).expect("Failed to load Tetra3 database");

    let image_paths = get_test_images();
    let mut tetra_extractor = Extractor::new();

    let options = ExtractOptions {
        downsample: None,
        return_images: false,
        ..Default::default()
    };

    let mut table_rows = Vec::new();

    // Macros bypass the need for explicit type signatures, fixing compilation errors
    // for types that are nested or not exported in the root namespace.
    macro_rules! format_solve {
        ($res:expr) => {
            if $res.status == SolveStatus::MatchFound {
                format!(
                    "{:.3}, {:.3}, {:.3}",
                    $res.ra.unwrap(),
                    $res.dec.unwrap(),
                    $res.roll.unwrap()
                )
            } else {
                "None".to_string()
            }
        };
    }

    macro_rules! to_array {
        ($cents:expr, $y:ident, $x:ident) => {{
            let mut flat = Vec::with_capacity($cents.len() * 2);
            for c in $cents {
                flat.push(c.$y as f64);
                flat.push(c.$x as f64);
            }
            Array2::from_shape_vec(($cents.len(), 2), flat).unwrap()
        }};
    }

    macro_rules! compare_top4 {
        ($olive:expr, $cedar:expr) => {{
            let mut match_top4 = true;
            let eps = 2.0; // Distance tolerance
            let compare_len = 4.min($olive.len()).min($cedar.len());
            if ($olive.len() < 4 || $cedar.len() < 4) && $olive.len() != $cedar.len() {
                match_top4 = false;
            }
            for i in 0..compare_len {
                let p = &$olive[i];
                let c = &$cedar[i];
                let dist = ((p.x - c.centroid_x as f64).powi(2)
                    + (p.y - c.centroid_y as f64).powi(2))
                .sqrt();
                if dist > eps {
                    match_top4 = false;
                    break;
                }
            }
            if match_top4 && compare_len > 0 {
                "YES".to_string()
            } else if compare_len == 0 && $olive.len() == $cedar.len() {
                "YES (0)".to_string()
            } else {
                "NO".to_string()
            }
        }};
    }

    for path in &image_paths {
        let img_name = path.file_name().unwrap().to_string_lossy().to_string();
        let base_img = image::open(path).unwrap();
        let luma_img = base_img.to_luma8();
        let (w, h) = luma_img.dimensions();

        let mut rust_input_img_u8 = Array2::<u8>::zeros((h as usize, w as usize));
        let mut rust_input_img_f32 = Array2::<f32>::zeros((h as usize, w as usize));
        for y in 0..h {
            for x in 0..w {
                let p = luma_img.get_pixel(x, y)[0];
                rust_input_img_u8[[y as usize, x as usize]] = p;
                rust_input_img_f32[[y as usize, x as usize]] = p as f32;
            }
        }

        // Run Port u8
        let port_u8_result = tetra_extractor.extract_u8(&rust_input_img_u8, options.clone());
        let port_u8_count = port_u8_result.centroids.len();

        // Run Fast Extractor
        let fe_options = FastExtractOptions {
            downsample: FastDownsample::None,
            ..Default::default()
        };
        let mut fast_extractor = FastExtractor::new(w as usize, h as usize, fe_options);
        let fast_u8_result = fast_extractor.extract(&rust_input_img_u8);
        let fast_u8_count = fast_u8_result.len();

        // Run Port f32 (Original f32)
        let port_f32_result = tetra_extractor.extract(&rust_input_img_f32, options.clone());
        let port_f32_count = port_f32_result.centroids.len();

        // Run Cedar Detect
        let noise_estimate = estimate_noise_from_image(&luma_img);
        let cedar_result = get_stars_from_image(
            &luma_img,
            noise_estimate,
            8.0,
            false,
            1, // ds
            true,
            false,
        );
        let cedar_count = cedar_result.0.len();

        // Solve Port u8
        let u8_cents_arr = to_array!(&port_u8_result.centroids, y, x);
        let u8_solve_res =
            solver.solve(&u8_cents_arr, (h as f64, w as f64), SolveOptions::default());
        let u8_solve_str = format_solve!(u8_solve_res);

        // Solve Fast u8
        let fast_u8_cents_arr = to_array!(&fast_u8_result, y, x);
        let fast_u8_solve_res = solver.solve(
            &fast_u8_cents_arr,
            (h as f64, w as f64),
            SolveOptions::default(),
        );
        let fast_u8_solve_str = format_solve!(fast_u8_solve_res);

        // Solve Cedar
        let cedar_cents_arr = to_array!(&cedar_result.0, centroid_y, centroid_x);
        let cedar_solve_res = solver.solve(
            &cedar_cents_arr,
            (h as f64, w as f64),
            SolveOptions::default(),
        );
        let cedar_solve_str = format_solve!(cedar_solve_res);

        // Solve Original f32
        let orig_f32_cents_arr = to_array!(&port_f32_result.centroids, y, x);
        let orig_f32_solve_res = solver.solve(
            &orig_f32_cents_arr,
            (h as f64, w as f64),
            SolveOptions::default(),
        );
        let orig_f32_solve_str = format_solve!(orig_f32_solve_res);

        // Calculate top 4 Matches against Cedar
        let match_str_u8 = compare_top4!(&port_u8_result.centroids, &cedar_result.0);
        let match_str_fast_u8 = compare_top4!(&fast_u8_result, &cedar_result.0);

        table_rows.push(TableRow {
            img_name,
            port_u8_count,
            port_f32_count,
            fast_u8_count,
            cedar_count,
            match_str_u8,
            match_str_fast_u8,
            u8_solve_str,
            fast_u8_solve_str,
            cedar_solve_str,
            orig_f32_solve_str,
        });
    }

    println!(
        "\n{:<40} | {:<10} | {:<10} | {:<10} | {:<10} | {:<18} | {:<22}",
        "Image",
        "Port f32",
        "Port u8",
        "Fast Ext",
        "Cedar",
        "Top 4 Match (u8)",
        "Top 4 Match (Fast)"
    );
    println!("{}", "-".repeat(140));

    for row in &table_rows {
        println!(
            "{:<40} | {:<10} | {:<10} | {:<10} | {:<10} | {:<18} | {:<22}",
            row.img_name,
            row.port_f32_count,
            row.port_u8_count,
            row.fast_u8_count,
            row.cedar_count,
            row.match_str_u8,
            row.match_str_fast_u8
        );
    }

    println!(
        "\n\n{:<40} | {:<25} | {:<25} | {:<25} | {:<25}",
        "Image", "Color f32 Solved", "Port u8 Solved", "Fast Ext Solved", "Cedar Solved"
    );
    println!("{}", "-".repeat(154));

    for row in &table_rows {
        println!(
            "{:<40} | {:<25} | {:<25} | {:<25} | {:<25}",
            row.img_name,
            row.orig_f32_solve_str,
            row.u8_solve_str,
            row.fast_u8_solve_str,
            row.cedar_solve_str
        );
    }
    println!();
}

#[derive(Debug)]
struct ExtractorComparisonRow {
    img_name: String,
    global_median_count: usize,
    global_median_time: std::time::Duration,
    local_median_count: usize,
    local_median_time: std::time::Duration,
    fast_extractor_count: usize,
    fast_extractor_time: std::time::Duration,
    block_median_count: usize,
    block_median_time: std::time::Duration,
    global_solve_str: String,
    local_solve_str: String,
    fast_solve_str: String,
    block_solve_str: String,
}

#[test]
fn test_fast_extractor_vs_others() {
    let db_path = std::path::Path::new("tests/fixtures/default_database.npz");
    if !db_path.exists() {
        eprintln!("Skipping test: default_database.npz not found.");
        return;
    }
    let mut solver =
        tetra3::solver::Solver::load_database(db_path).expect("Failed to load Tetra3 database");

    let image_paths = get_test_images();
    let mut tetra_extractor = tetra3::extractor::Extractor::new();

    let mut table_rows = Vec::new();

    macro_rules! format_solve {
        ($res:expr) => {
            if $res.status == tetra3::solver::SolveStatus::MatchFound {
                format!(
                    "{:.3}, {:.3}, {:.3}",
                    $res.ra.unwrap(),
                    $res.dec.unwrap(),
                    $res.roll.unwrap()
                )
            } else {
                "None".to_string()
            }
        };
    }

    macro_rules! to_array {
        ($cents:expr, $y:ident, $x:ident) => {{
            let mut flat = Vec::with_capacity($cents.len() * 2);
            for c in $cents {
                flat.push(c.$y as f64);
                flat.push(c.$x as f64);
            }
            ndarray::Array2::from_shape_vec(($cents.len(), 2), flat).unwrap()
        }};
    }

    for path in &image_paths {
        let img_name = path.file_name().unwrap().to_string_lossy().to_string();
        let base_img = image::open(path).unwrap();
        let luma_img = base_img.to_luma8();
        let (w, h) = luma_img.dimensions();

        let mut rust_input_img_u8 = ndarray::Array2::<u8>::zeros((h as usize, w as usize));
        for y in 0..h {
            for x in 0..w {
                rust_input_img_u8[[y as usize, x as usize]] = luma_img.get_pixel(x, y)[0];
            }
        }

        // Global Median (Original Extractor)
        let opt_global = tetra3::extractor::ExtractOptions {
            downsample: None,
            return_images: false,
            bg_sub_mode: Some(tetra3::extractor::BgSubMode::GlobalMedian),
            ..Default::default()
        };
        let t0 = std::time::Instant::now();
        let res_global = tetra_extractor.extract_u8(&rust_input_img_u8, opt_global);
        let global_median_time = t0.elapsed();
        let global_median_count = res_global.centroids.len();

        // Local Median (Original Extractor)
        let opt_local = tetra3::extractor::ExtractOptions {
            downsample: None,
            return_images: false,
            bg_sub_mode: Some(tetra3::extractor::BgSubMode::LocalMedian),
            ..Default::default()
        };
        let t0 = std::time::Instant::now();
        let res_local = tetra_extractor.extract_u8(&rust_input_img_u8, opt_local);
        let local_median_time = t0.elapsed();
        let local_median_count = res_local.centroids.len();

        // Fast Extractor
        let opt_fast = FastExtractOptions {
            downsample: FastDownsample::None,
            bg_sub_mode: Some(FastBgSubMode::GlobalMedian),
            ..Default::default()
        };
        let mut fast_extractor = FastExtractor::new(w as usize, h as usize, opt_fast);
        let t0 = std::time::Instant::now();
        let res_fast = fast_extractor.extract(&rust_input_img_u8);
        let fast_extractor_time = t0.elapsed();
        let fast_extractor_count = res_fast.len();

        // Fast Extractor with Block Median
        let opt_block = FastExtractOptions {
            downsample: FastDownsample::None,
            bg_sub_mode: Some(FastBgSubMode::BlockMedian { block_size: 64 }),
            ..Default::default()
        };
        let mut block_extractor = FastExtractor::new(w as usize, h as usize, opt_block);
        let t0 = std::time::Instant::now();
        let res_block = block_extractor.extract(&rust_input_img_u8);
        let block_median_time = t0.elapsed();
        let block_median_count = res_block.len();

        let global_cents_arr = to_array!(&res_global.centroids, y, x);
        let global_solve_res = solver.solve(
            &global_cents_arr,
            (h as f64, w as f64),
            tetra3::solver::SolveOptions::default(),
        );
        let global_solve_str = format_solve!(global_solve_res);

        let local_cents_arr = to_array!(&res_local.centroids, y, x);
        let local_solve_res = solver.solve(
            &local_cents_arr,
            (h as f64, w as f64),
            tetra3::solver::SolveOptions::default(),
        );
        let local_solve_str = format_solve!(local_solve_res);

        let fast_cents_arr = to_array!(&res_fast, y, x);
        let fast_solve_res = solver.solve(
            &fast_cents_arr,
            (h as f64, w as f64),
            tetra3::solver::SolveOptions::default(),
        );
        let fast_solve_str = format_solve!(fast_solve_res);

        let block_cents_arr = to_array!(&res_block, y, x);
        let block_solve_res = solver.solve(
            &block_cents_arr,
            (h as f64, w as f64),
            tetra3::solver::SolveOptions::default(),
        );
        let block_solve_str = format_solve!(block_solve_res);

        table_rows.push(ExtractorComparisonRow {
            img_name,
            global_median_count,
            global_median_time,
            local_median_count,
            local_median_time,
            fast_extractor_count,
            fast_extractor_time,
            block_median_count,
            block_median_time,
            global_solve_str,
            local_solve_str,
            fast_solve_str,
            block_solve_str,
        });
    }

    println!(
        "\n{:<40} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15}",
        "Image",
        "Global Count",
        "Global Time",
        "Local Count",
        "Local Time",
        "Fast Count",
        "Fast Time",
        "Block Count",
        "Block Time",
    );
    println!("{}", "-".repeat(180));

    for row in &table_rows {
        println!(
            "{:<40} | {:<15} | {:<15?} | {:<15} | {:<15?} | {:<15} | {:<15?} | {:<15} | {:<15?}",
            row.img_name,
            row.global_median_count,
            row.global_median_time,
            row.local_median_count,
            row.local_median_time,
            row.fast_extractor_count,
            row.fast_extractor_time,
            row.block_median_count,
            row.block_median_time,
        );
    }

    println!(
        "\n{:<40} | {:<30} | {:<30} | {:<30} | {:<30}",
        "Image", "Global Solved", "Local Solved", "Fast Solved", "Block Solved"
    );
    println!("{}", "-".repeat(173));

    for row in &table_rows {
        println!(
            "{:<40} | {:<30} | {:<30} | {:<30} | {:<30}",
            row.img_name,
            row.global_solve_str,
            row.local_solve_str,
            row.fast_solve_str,
            row.block_solve_str
        );
    }
    println!();
}

#[test]
#[ignore]
fn test_benchmark_block_median() {
    let iterations = 200;
    let image_paths = get_test_images();
    let downsamples = [FastDownsample::None, FastDownsample::X2, FastDownsample::X4];

    for &ds in &downsamples {
        let mut total_time = Duration::ZERO;

        // Use the first image for benchmarking to keep it consistent
        let path = &image_paths[0];
        let base_img = image::open(path).unwrap();
        let (w, h) = base_img.dimensions();

        let ds_factor = ds.factor() as u32;
        let new_w = w - (w % ds_factor);
        let new_h = h - (h % ds_factor);

        let mut input_img = ndarray::Array2::<u8>::zeros((new_h as usize, new_w as usize));
        let luma_img = base_img.to_luma8();
        for y in 0..new_h {
            for x in 0..new_w {
                input_img[[y as usize, x as usize]] = luma_img.get_pixel(x, y)[0];
            }
        }

        let options = FastExtractOptions {
            downsample: ds,
            bg_sub_mode: Some(FastBgSubMode::BlockMedian { block_size: 64 }),
            sigma_mode: FastSigmaMode::GlobalRootSquare,
            ..Default::default()
        };

        let mut extractor = FastExtractor::new(new_w as usize, new_h as usize, options);

        println!(
            "Benchmarking BlockMedian + GlobalRootSquare (Downsample: {:?}, Iterations: {})...",
            ds, iterations
        );

        for _ in 0..iterations {
            let start = Instant::now();
            let _res = extractor.extract(&input_img);
            total_time += start.elapsed();
        }

        let avg_time = total_time / iterations as u32;
        println!("  -> Average Time: {:.2?}", avg_time);
        println!("  -> Total Time:   {:.2?}", total_time);
        println!("----------------------------------------------");
    }
}
