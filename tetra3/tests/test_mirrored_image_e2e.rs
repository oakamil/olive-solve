// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use ndarray::Array2;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use tetra3::{ExtractOptions, Extractor, SolveOptions, SolveStatus, Solver};

/// Helper function to locate and return the test images.
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

#[test]
fn test_mirrored_image_e2e_all_images() {
    let db_path = Path::new("tests/fixtures/default_database.npz");
    if !db_path.exists() {
        eprintln!("Skipping test: default_database.npz not found.");
        return;
    }
    let mut solver = Solver::load_database(db_path).expect("Failed to load Tetra3 database");
    let mut extractor = Extractor::new();

    let image_paths = get_test_images();
    let mut failures = Vec::new();
    let mut solved_count = 0;

    for image_path in &image_paths {
        let img_name = image_path.file_name().unwrap().to_string_lossy();
        println!("Testing image: {}", img_name);

        let img_load = match image::open(image_path) {
            Ok(img) => img,
            Err(e) => {
                failures.push(format!("{}: Failed to open image: {}", img_name, e));
                continue;
            }
        };
        let img = img_load.to_luma8();
        let (width, height) = img.dimensions();

        // 1. Process original image
        let mut data = Array2::<u8>::zeros((height as usize, width as usize));
        for y in 0..height {
            for x in 0..width {
                data[[y as usize, x as usize]] = img.get_pixel(x, y)[0];
            }
        }

        let extract_res = extractor.extract_u8(&data, ExtractOptions::default());
        if extract_res.centroids.len() < 4 {
            println!(
                "  -> Skipping: Too few stars found ({})",
                extract_res.centroids.len()
            );
            continue;
        }

        let mut centroids_flat = Vec::with_capacity(extract_res.centroids.len() * 2);
        for c in &extract_res.centroids {
            centroids_flat.push(c.y);
            centroids_flat.push(c.x);
        }
        let centroids =
            Array2::from_shape_vec((extract_res.centroids.len(), 2), centroids_flat).unwrap();

        let solve_res = solver.solve(
            &centroids,
            (height as f64, width as f64),
            SolveOptions::default(),
        );
        if solve_res.status != SolveStatus::MatchFound {
            println!(
                "  -> Skipping: Original image failed to solve ({:?})",
                solve_res.status
            );
            continue;
        }
        assert!(
            !solve_res.is_mirrored,
            "Original image {} should not be detected as mirrored",
            img_name
        );

        // 2. Process mirrored image
        let mirrored_img = image::imageops::flip_horizontal(&img);
        let mut mirrored_data = Array2::<u8>::zeros((height as usize, width as usize));
        for y in 0..height {
            for x in 0..width {
                mirrored_data[[y as usize, x as usize]] = mirrored_img.get_pixel(x, y)[0];
            }
        }

        let extract_res_mirrored = extractor.extract_u8(&mirrored_data, ExtractOptions::default());
        let mut centroids_flat_mirrored =
            Vec::with_capacity(extract_res_mirrored.centroids.len() * 2);
        for c in &extract_res_mirrored.centroids {
            centroids_flat_mirrored.push(c.y);
            centroids_flat_mirrored.push(c.x);
        }
        let centroids_mirrored = Array2::from_shape_vec(
            (extract_res_mirrored.centroids.len(), 2),
            centroids_flat_mirrored,
        )
        .unwrap();

        let solve_res_mirrored = solver.solve(
            &centroids_mirrored,
            (height as f64, width as f64),
            SolveOptions::default(),
        );

        if solve_res_mirrored.status != SolveStatus::MatchFound {
            failures.push(format!(
                "{}: Mirrored image failed to solve although original solved",
                img_name
            ));
            continue;
        }
        if !solve_res_mirrored.is_mirrored {
            failures.push(format!(
                "{}: Mirrored image was NOT detected as mirrored",
                img_name
            ));
            continue;
        }

        #[cfg(not(feature = "force-32bit-solver"))]
        let epsilon = 1e-4; // 0.0001 deg = 0.36 arcseconds (64-bit pipeline)
        #[cfg(feature = "force-32bit-solver")]
        let epsilon = 1e-2; // 0.01 deg = 36 arcseconds (32-bit pipeline)

        let ra1 = solve_res.ra.unwrap();
        let dec1 = solve_res.dec.unwrap();
        let roll1 = solve_res.roll.unwrap();
        let fov1 = solve_res.fov.unwrap();

        let ra2 = solve_res_mirrored.ra.unwrap();
        let dec2 = solve_res_mirrored.dec.unwrap();
        let roll2 = solve_res_mirrored.roll.unwrap();
        let fov2 = solve_res_mirrored.fov.unwrap();

        let mut image_errors = Vec::new();
        if (ra1 - ra2).abs() >= epsilon {
            image_errors.push(format!("RA mismatch: original {}, mirrored {}", ra1, ra2));
        }
        if (dec1 - dec2).abs() >= epsilon {
            image_errors.push(format!(
                "Dec mismatch: original {}, mirrored {}",
                dec1, dec2
            ));
        }

        let mut roll_diff = (roll1 - roll2).abs();
        if roll_diff > 180.0 {
            roll_diff = 360.0 - roll_diff;
        }
        if roll_diff >= epsilon {
            image_errors.push(format!(
                "Roll mismatch: original {}, mirrored {}",
                roll1, roll2
            ));
        }

        if (fov1 - fov2).abs() >= epsilon {
            image_errors.push(format!(
                "FOV mismatch: original {}, mirrored {}",
                fov1, fov2
            ));
        }

        if !image_errors.is_empty() {
            failures.push(format!(
                "{}: Mismatches detected:\n  {}",
                img_name,
                image_errors.join("\n  ")
            ));
        } else {
            solved_count += 1;
            println!(
                "  -> OK: RA={:.6}, Dec={:.6}, Roll={:.6}, FOV={:.6}",
                ra1, dec1, roll1, fov1
            );
        }
    }

    println!(
        "\nSummary: {} images tested, {} images solved and verified correctly.",
        image_paths.len(),
        solved_count
    );

    if !failures.is_empty() {
        panic!(
            "The following images failed the mirrored consistency test:\n\n{}",
            failures.join("\n\n")
        );
    }
}
