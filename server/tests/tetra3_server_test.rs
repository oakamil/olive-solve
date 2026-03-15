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
        CentroidConfig, ImageInput, SolveFromImageRequest, SolveOptions,
        tetra3_service_client::Tetra3ServiceClient, tetra3_service_server::Tetra3ServiceServer,
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
        "{:<35} | {:<8} | {:<8} | {:<8} | {:<8} | {:<12} | {:<12} | {:<12}",
        "Image Name",
        "RA (°)",
        "Dec (°)",
        "Roll (°)",
        "FOV (°)",
        "Extract (ms)",
        "Solve (ms)",
        "Total (ms)"
    );
    println!("{:-<125}", "");

    // 5. Process each image
    for img_path in entries {
        let file_name = img_path.file_name().unwrap().to_string_lossy().to_string();

        // Truncate long file names
        let max_name_len = 35;
        let display_name = if file_name.len() > max_name_len {
            format!("{}...", &file_name[0..(max_name_len - 3)])
        } else {
            file_name
        };

        // Load image and convert to 32-bit float grayscale
        let img = image::open(&img_path)?.to_luma32f();
        let (width, height) = img.dimensions();

        // Safely extract the raw f32 slice directly
        let pixel_data: &[f32] = img.as_raw().as_slice();
        let byte_size = pixel_data.len() * std::mem::size_of::<f32>();

        // Create a unique shared memory segment for this image
        let shmem_name = format!("/tetra3_test_shmem_{}", uuid::Uuid::new_v4().simple());
        let mut shmem = ShmemConf::new()
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
            "{:<35} | {:<8} | {:<8} | {:<8} | {:<8} | {:<12} | {:<12} | {:<12}",
            display_name, ra, dec, roll, fov, extract_ms, solve_ms, total_ms_str
        );

        // Shmem goes out of scope and is automatically unlinked/freed here for this iteration
    }

    Ok(())
}
