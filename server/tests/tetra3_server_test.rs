// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use shared_memory::ShmemConf;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tonic::transport::Server;

use tetra3::{extractor::CentroidConfig as T3CentroidConfig, extractor::Extractor, solver::Solver};
use tetra3_server::{
    Tetra3ServerImpl,
    proto::{
        CentroidConfig, ExtractRequest, ImageInput, Pixel, SolveFromImageRequest, SolveOptions,
        SolveRequest, tetra3_service_client::Tetra3ServiceClient,
        tetra3_service_server::Tetra3ServiceServer,
    },
};

#[tokio::test]
async fn test_solve_from_image_batch() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define paths
    let db_path = PathBuf::from("../../cedar-solve/tetra3/data/default_database.npz");
    let img_dir = PathBuf::from("../../cedar-solve/examples/data/medium_fov");

    assert!(db_path.exists(), "Database not found at {:?}", db_path);
    assert!(
        img_dir.exists(),
        "Image directory not found at {:?}",
        img_dir
    );

    // 2. Start the gRPC server in the background
    let solver = Solver::load_database(&db_path).expect("Failed to load solver database");
    let extractor = Extractor::new(T3CentroidConfig::default());

    let service = Tetra3ServerImpl {
        solver: Arc::new(Mutex::new(solver)),
        extractor: Arc::new(Mutex::new(extractor)),
    };

    let addr = "[::1]:50051".parse().unwrap();

    tokio::spawn(async move {
        Server::builder()
            .add_service(Tetra3ServiceServer::new(service))
            .serve(addr)
            .await
            .unwrap();
    });

    // Give the server a moment to bind
    tokio::time::sleep(Duration::from_millis(100)).await;

    // 3. Connect the client
    let channel_url = "http://[::1]:50051".to_string();
    let mut client = Tetra3ServiceClient::connect(channel_url).await?;

    // 4. Gather and sort image files
    let mut entries: Vec<PathBuf> = fs::read_dir(&img_dir)?
        .filter_map(Result::ok)
        .map(|res| res.path())
        .filter(|path| path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("jpg"))
        .collect();

    entries.sort(); // Process in alphabetical order

    println!(
        "\n{:<40} | {:<8} | {:<8} | {:<8} | {:<8} | {:<12} | {:<12} | {:<12}",
        "Image Name",
        "RA (°)",
        "Dec (°)",
        "Roll (°)",
        "FOV (°)",
        "Extract (ms)",
        "Solve (ms)",
        "Total (ms)"
    );
    println!("{:-<130}", "");

    // 5. Process each image
    for img_path in entries {
        let file_name = img_path.file_name().unwrap().to_string_lossy().to_string();

        // Load image as 8-bit grayscale to preserve the 0-255 range
        let img = image::open(&img_path)?.to_luma8();
        let (width, height) = img.dimensions();

        // Manually cast the u8 pixels to f32 to avoid the image crate's 0.0-1.0 normalization
        let pixel_data: Vec<f32> = img.as_raw().iter().map(|&p| p as f32).collect();
        let byte_size = pixel_data.len() * std::mem::size_of::<f32>();

        // Create a unique shared memory segment for this image
        let shmem_name = format!("/tetra3_test_shmem_{}", uuid::Uuid::new_v4().simple());
        let shmem = ShmemConf::new()
            .os_id(&shmem_name)
            .size(byte_size)
            .create()?;

        // Zero-copy transfer: copy pixel data directly into shared memory pointer
        unsafe {
            std::ptr::copy_nonoverlapping(
                pixel_data.as_ptr(),
                shmem.as_ptr() as *mut f32,
                pixel_data.len(),
            );
        }

        // Build the request using default configs
        let request = tonic::Request::new(SolveFromImageRequest {
            image: Some(ImageInput {
                shmem_name: shmem_name.clone(),
                width,
                height,
            }),
            extract_config: Some(CentroidConfig::default()),
            solve_options: Some(SolveOptions::default()),
        });

        // Fire the RPC and time the full round-trip
        let t0 = Instant::now();
        let response = client.solve_from_image(request).await?.into_inner();
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Safely extract optional fields for printing
        let ra = response
            .ra
            .map(|v| format!("{:.3}", v))
            .unwrap_or_else(|| "N/A".into());
        let dec = response
            .dec
            .map(|v| format!("{:.3}", v))
            .unwrap_or_else(|| "N/A".into());
        let roll = response
            .roll
            .map(|v| format!("{:.3}", v))
            .unwrap_or_else(|| "N/A".into());
        let fov = response
            .fov
            .map(|v| format!("{:.3}", v))
            .unwrap_or_else(|| "N/A".into());
        let extract_ms = response
            .extraction_time_ms
            .map(|v| format!("{:.2}", v))
            .unwrap_or_else(|| "N/A".into());
        let solve_ms = format!("{:.2}", response.t_solve_ms);
        let total_ms_str = format!("{:.2}", total_ms);

        println!(
            "{:<40.40} | {:<8} | {:<8} | {:<8} | {:<8} | {:<12} | {:<12} | {:<12}",
            file_name, ra, dec, roll, fov, extract_ms, solve_ms, total_ms_str
        );

        // Shmem goes out of scope and is automatically unlinked/freed here for this iteration
    }

    Ok(())
}

#[tokio::test]
async fn test_extract_then_solve() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define paths
    let db_path = PathBuf::from("../../cedar-solve/tetra3/data/default_database.npz");
    let img_dir = PathBuf::from("../../cedar-solve/examples/data/medium_fov");

    assert!(db_path.exists(), "Database not found at {:?}", db_path);
    assert!(
        img_dir.exists(),
        "Image directory not found at {:?}",
        img_dir
    );

    // 2. Start the gRPC server in the background
    let solver = Solver::load_database(&db_path).expect("Failed to load solver database");
    let extractor = Extractor::new(T3CentroidConfig::default());

    let service = Tetra3ServerImpl {
        solver: Arc::new(Mutex::new(solver)),
        extractor: Arc::new(Mutex::new(extractor)),
    };

    // Use a different port to avoid conflicts with the other test running in parallel
    let addr = "[::1]:50052".parse().unwrap();

    tokio::spawn(async move {
        Server::builder()
            .add_service(Tetra3ServiceServer::new(service))
            .serve(addr)
            .await
            .unwrap();
    });

    // Give the server a moment to bind
    tokio::time::sleep(Duration::from_millis(100)).await;

    // 3. Connect the client
    let channel_url = "http://[::1]:50052".to_string();
    let mut client = Tetra3ServiceClient::connect(channel_url).await?;

    // 4. Gather and sort image files
    let mut entries: Vec<PathBuf> = fs::read_dir(&img_dir)?
        .filter_map(Result::ok)
        .map(|res| res.path())
        .filter(|path| path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("jpg"))
        .collect();

    entries.sort(); // Process in alphabetical order

    println!(
        "\n{:<40} | {:<10} | {:<8} | {:<8} | {:<12} | {:<12} | {:<12}",
        "Image Name", "Centroids", "RA (°)", "Dec (°)", "Extract (ms)", "Solve (ms)", "Total (ms)"
    );
    println!("{:-<115}", "");

    // 5. Process each image
    for img_path in entries {
        let file_name = img_path.file_name().unwrap().to_string_lossy().to_string();

        // Load image as 8-bit grayscale to preserve the 0-255 range
        let img = image::open(&img_path)?.to_luma8();
        let (width, height) = img.dimensions();

        // Manually cast the u8 pixels to f32 to avoid the image crate's 0.0-1.0 normalization
        let pixel_data: Vec<f32> = img.as_raw().iter().map(|&p| p as f32).collect();
        let byte_size = pixel_data.len() * std::mem::size_of::<f32>();

        // Create a unique shared memory segment for this image
        let shmem_name = format!("/tetra3_test_shmem_{}", uuid::Uuid::new_v4().simple());
        let shmem = ShmemConf::new()
            .os_id(&shmem_name)
            .size(byte_size)
            .create()?;

        // Zero-copy transfer: copy pixel data directly into shared memory pointer
        unsafe {
            std::ptr::copy_nonoverlapping(
                pixel_data.as_ptr(),
                shmem.as_ptr() as *mut f32,
                pixel_data.len(),
            );
        }

        // Start total execution timer
        let t0_total = Instant::now();

        // --- Step A: Call the Extract RPC ---
        let extract_request = tonic::Request::new(ExtractRequest {
            image: Some(ImageInput {
                shmem_name: shmem_name.clone(),
                width,
                height,
            }),
            config: Some(CentroidConfig::default()),
        });

        let extract_response = client.extract(extract_request).await?.into_inner();
        let extract_ms = format!("{:.2}", extract_response.extraction_time_ms);
        let num_centroids = extract_response.centroids.len();

        // Map the extracted centroids into the generic Pixel type expected by SolveRequest
        let pixels: Vec<Pixel> = extract_response
            .centroids
            .into_iter()
            .map(|c| Pixel { y: c.y, x: c.x })
            .collect();

        // --- Step B: Call the Solve RPC ---
        let solve_request = tonic::Request::new(SolveRequest {
            centroids: pixels,
            image_width: width,
            image_height: height,
            options: Some(SolveOptions::default()),
        });

        let solve_response = client.solve(solve_request).await?.into_inner();

        let total_ms = format!("{:.2}", t0_total.elapsed().as_secs_f64() * 1000.0);
        let solve_ms = format!("{:.2}", solve_response.t_solve_ms);

        // Safely extract optional fields for printing
        let ra = solve_response
            .ra
            .map(|v| format!("{:.3}", v))
            .unwrap_or_else(|| "N/A".into());
        let dec = solve_response
            .dec
            .map(|v| format!("{:.3}", v))
            .unwrap_or_else(|| "N/A".into());

        println!(
            "{:<40.40} | {:<10} | {:<8} | {:<8} | {:12} | {:<12} | {:<12}",
            file_name, num_centroids, ra, dec, extract_ms, solve_ms, total_ms
        );

        // Shmem goes out of scope and is automatically unlinked/freed here for this iteration
    }

    Ok(())
}
