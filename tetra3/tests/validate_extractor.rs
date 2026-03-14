// Copyright (c) 2026 Omair Kamil oakamil@gmail.com
// See LICENSE file in root directory for license terms.

use ndarray::Array2;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::path::Path;
use std::time::{Duration, Instant};
use walkdir::WalkDir;
use image::GrayImage;

use tetra3::extractor::{CentroidConfig, TetraExtractor};
use cedar_detect::algorithm::{estimate_noise_from_image, get_stars_from_image};

#[test]
fn test_extraction_against_python() {
    let mut total_rust_time = Duration::ZERO;
    let mut total_py_time = Duration::ZERO;
    let mut image_count = 0;

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

    Python::with_gil(|py| {
        // Python helper to load the image into a float32 numpy array exactly as tetra3 does.
        // tetra3 is assumed to be available in the current python environment.
        let py_code = r#"
from PIL import Image
import numpy as np
from tetra3.tetra3 import get_centroids_from_image

def load_image_as_array(path):
    image = Image.open(path)
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = image[:, :, 0]*.299 + image[:, :, 1]*.587 + image[:, :, 2]*.114
        else:
            image = image.squeeze(axis=2)
    return image

def run_py_extraction(image_array):
    # return_moments=True returns: (centroids, [sum, area, moments, axis_ratio])
    centroids, moments = get_centroids_from_image(image_array, return_moments=True)
    # Force float64 arrays to prevent PyO3 TypeError panics in Rust
    return (
        np.asarray(centroids, dtype=np.float64),
        [np.asarray(m, dtype=np.float64) for m in moments]
    )
"#;

        let module = PyModule::from_code(py, py_code, "test_helper.py", "test_helper").unwrap();
        let load_image_as_array = module.getattr("load_image_as_array").unwrap();
        let run_py_extraction = module.getattr("run_py_extraction").unwrap();

        let mut extractor = tetra3::extractor::TetraExtractor::new(CentroidConfig {
                return_images: false,
                ..Default::default()
            });

        for image_path in &image_paths {
            let img_name = image_path.file_name().unwrap();
            println!("Testing image: {:?}", img_name);

            // 1. Load the image into Python and get the raw f32 numpy array.
            let py_image_array: &PyAny = load_image_as_array
                .call1((image_path.to_str().unwrap(),))
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
            let rust_input_img: Array2<f32> = Array2::from_shape_vec((nrows, ncols), vec_data).unwrap();

            // 2. Run Python Algorithm
            let start_py = Instant::now();
            let py_result = run_py_extraction.call1((py_image_array,)).unwrap();
            total_py_time += start_py.elapsed();

            // 3. Run Rust Algorithm
            let start_rust = Instant::now();
            let rust_result = extractor.extract(&rust_input_img);
            total_rust_time += start_rust.elapsed();

            image_count += 1;

            // 4. Parse Python Results using safe indexing methods
            let py_tuple: &PyTuple = py_result.downcast().unwrap();
            let py_centroids_arr: PyReadonlyArray2<f64> = py_tuple.get_item(0).unwrap().extract().unwrap();

            let py_moments_list: &PyList = py_tuple.get_item(1).unwrap().downcast().unwrap();
            let py_sum_arr: PyReadonlyArray1<f64> = py_moments_list.get_item(0).unwrap().extract().unwrap();
            let py_area_arr: PyReadonlyArray1<f64> = py_moments_list.get_item(1).unwrap().extract().unwrap();
            let py_m2_arr: PyReadonlyArray2<f64> = py_moments_list.get_item(2).unwrap().extract().unwrap();
            let py_ratio_arr: PyReadonlyArray1<f64> = py_moments_list.get_item(3).unwrap().extract().unwrap();

            let py_centroids_view = py_centroids_arr.as_array();
            let py_sum = py_sum_arr.as_array();
            let py_area = py_area_arr.as_array();
            let py_m2_view = py_m2_arr.as_array();
            let py_ratio = py_ratio_arr.as_array();

            let rust_centroids = &rust_result.centroids;

            // 5. Compare Results
            assert_eq!(
                rust_centroids.len(),
                py_centroids_view.shape()[0],
                "Mismatch in number of extracted centroids for image: {:?}",
                img_name
            );

            let eps = 1e-3;

            for i in 0..rust_centroids.len() {
                let r_c = &rust_centroids[i];

                let p_y = *py_centroids_view.get([i, 0]).unwrap();
                let p_x = *py_centroids_view.get([i, 1]).unwrap();
                let p_sum = *py_sum.get(i).unwrap();
                let p_area = *py_area.get(i).unwrap() as usize;
                
                // m2_xx is at index 0, m2_yy is at index 1
                let p_m2_xx = *py_m2_view.get([i, 0]).unwrap();
                let p_m2_yy = *py_m2_view.get([i, 1]).unwrap();
                let p_m2_xy = *py_m2_view.get([i, 2]).unwrap();
                
                let p_ratio = *py_ratio.get(i).unwrap();

                assert!(
                    (r_c.y - p_y).abs() < eps, 
                    "[{:?}] Y mismatch at star {}: Rust {} vs Py {}", img_name, i, r_c.y, p_y
                );
                assert!(
                    (r_c.x - p_x).abs() < eps, 
                    "[{:?}] X mismatch at star {}: Rust {} vs Py {}", img_name, i, r_c.x, p_x
                );
                assert!(
                    (r_c.sum - p_sum).abs() / p_sum.max(1.0) < 1e-4, 
                    "[{:?}] Sum mismatch at star {} (y:{:.1}, x:{:.1}): Rust {} vs Py {}", 
                    img_name, i, r_c.y, r_c.x, r_c.sum, p_sum
                );
                assert_eq!(
                    r_c.area, p_area, 
                    "[{:?}] Area mismatch at star {} (y:{:.1}, x:{:.1}): Rust {} vs Py {}", 
                    img_name, i, r_c.y, r_c.x, r_c.area, p_area
                );

                // Higher order moments can be NaN in Python if area constraints fail
                if !p_m2_xx.is_nan() && !r_c.m2_xx.is_nan() {
                    assert!(
                        (r_c.m2_xx - p_m2_xx).abs() < eps, 
                        "[{:?}] m2_xx mismatch at star {} (y:{:.1}, x:{:.1}): Rust {} vs Py {}", 
                        img_name, i, r_c.y, r_c.x, r_c.m2_xx, p_m2_xx
                    );
                    assert!(
                        (r_c.m2_yy - p_m2_yy).abs() < eps, 
                        "[{:?}] m2_yy mismatch at star {} (y:{:.1}, x:{:.1}): Rust {} vs Py {}", 
                        img_name, i, r_c.y, r_c.x, r_c.m2_yy, p_m2_yy
                    );
                    assert!(
                        (r_c.m2_xy - p_m2_xy).abs() < eps, 
                        "[{:?}] m2_xy mismatch at star {} (y:{:.1}, x:{:.1}): Rust {} vs Py {}", 
                        img_name, i, r_c.y, r_c.x, r_c.m2_xy, p_m2_xy
                    );
                    assert!(
                        (r_c.axis_ratio - p_ratio).abs() < eps, 
                        "[{:?}] Axis ratio mismatch at star {} (y:{:.1}, x:{:.1}): Rust {} vs Py {}", 
                        img_name, i, r_c.y, r_c.x, r_c.axis_ratio, p_ratio
                    );
                }
            }
        }
    });
}

#[test]
fn test_performance() {
    let mut total_rust_time = Duration::ZERO;
    let mut total_py_time = Duration::ZERO;
    let mut total_cedar_time = Duration::ZERO;
    let mut image_count = 0;

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

    // Initialize global buffer to prevent OS allocation overhead during benchmarking
    let mut tetra_extractor = TetraExtractor::new(CentroidConfig {
        return_images: false,
        ..Default::default()
    });

    Python::with_gil(|py| {
        // Python helper to load the image into a float32 numpy array exactly as tetra3 does.
        // tetra3 is assumed to be available in the current python environment.
        let py_code = r#"
from PIL import Image
import numpy as np
from tetra3.tetra3 import get_centroids_from_image

def load_image_as_array(path):
    image = Image.open(path)
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = image[:, :, 0]*.299 + image[:, :, 1]*.587 + image[:, :, 2]*.114
        else:
            image = image.squeeze(axis=2)
    return image

def run_py_extraction(image_array):
    # return_moments=True returns: (centroids, [sum, area, moments, axis_ratio])
    centroids, moments = get_centroids_from_image(image_array, return_moments=True)
    # Force float64 arrays to prevent PyO3 TypeError panics in Rust
    return (
        np.asarray(centroids, dtype=np.float64),
        [np.asarray(m, dtype=np.float64) for m in moments]
    )
"#;

        let module = PyModule::from_code(py, py_code, "test_helper.py", "test_helper").unwrap();
        let load_image_as_array = module.getattr("load_image_as_array").unwrap();
        let run_py_extraction = module.getattr("run_py_extraction").unwrap();

        let iterations = 10;
        println!("Running {} iterations for accurate benchmarking...", iterations);

        for _i in 0..iterations {
            for image_path in &image_paths {
                // 1. Load the image into Python and get the raw f32 numpy array.
                let py_image_array: &PyAny = load_image_as_array
                    .call1((image_path.to_str().unwrap(),))
                    .unwrap();

                // Extract the Array2<f32> for Rust. 
                // FIX: Manually copy elements to avoid ndarray version mismatch between numpy crate and tetra3 crate
                let py_readonly_img: PyReadonlyArray2<f32> = py_image_array.extract().unwrap();
                let py_view = py_readonly_img.as_array();
                let rust_input_img: Array2<f32> = Array2::from_shape_fn(
                    (py_view.nrows(), py_view.ncols()), 
                    |(y, x)| py_view[[y, x]]
                );

                // Build image::GrayImage for cedar-detect
                let (height, width) = (rust_input_img.nrows(), rust_input_img.ncols());
                let mut cedar_img = GrayImage::new(width as u32, height as u32);
                for y in 0..height {
                    for x in 0..width {
                        // Clamp and cast the f32 pixel to u8 safely
                        let p_val = rust_input_img[[y, x]].clamp(0.0, 255.0) as u8;
                        cedar_img.put_pixel(x as u32, y as u32, image::Luma([p_val]));
                    }
                }

                // 2. Run Python Algorithm
                let start_py = Instant::now();
                let _py_result = run_py_extraction.call1((py_image_array,)).unwrap();
                total_py_time += start_py.elapsed();

                // 3. Run Rust Algorithm (Tetra3 Port)
                let start_rust = Instant::now();
                let _rust_result = tetra_extractor.extract(&rust_input_img);
                total_rust_time += start_rust.elapsed();

                // 4. Run Cedar Detect Algorithm
                let start_cedar = Instant::now();
                let noise_estimate = estimate_noise_from_image(&cedar_img);
                // Execute CedarDetect without binning (binning = 1)
                let _cedar_result = get_stars_from_image(
                    &cedar_img, noise_estimate, 8.0, false, 1, true, false
                );
                total_cedar_time += start_cedar.elapsed();

                image_count += 1;
            }
        }
    });

    println!("\n==============================================");
    println!("CENTROID EXTRACTION PERFORMANCE REPORT");
    println!("Images tested: {} ({} unique x 10 iterations)", image_count, image_paths.len());
    println!("----------------------------------------------");
    println!("Python (tetra3) Total Time : {:.2?}", total_py_time);
    if image_count > 0 {
        println!("Python (tetra3) Avg/Image  : {:.2?}", total_py_time / image_count as u32);
    }
    println!("----------------------------------------------");
    println!("Rust (Port) Total Time     : {:.2?}", total_rust_time);
    if image_count > 0 {
        println!("Rust (Port) Avg/Image      : {:.2?}", total_rust_time / image_count as u32);
    }
    println!("----------------------------------------------");
    println!("Rust (CedarDetect) Total   : {:.2?}", total_cedar_time);
    if image_count > 0 {
        println!("Rust (CedarDetect) Avg     : {:.2?}", total_cedar_time / image_count as u32);
    }
    println!("----------------------------------------------");

    let speedup = total_py_time.as_secs_f64() / total_rust_time.as_secs_f64();
    println!("Rust Speedup               : {:.2}x", speedup);
    println!("==============================================\n");
}
