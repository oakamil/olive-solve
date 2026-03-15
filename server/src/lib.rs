// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use ndarray::{Array2, ArrayView2};
use shared_memory::ShmemConf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::{Request, Response, Status};

use tetra3::{
    extractor::{BgSubMode, CentroidConfig, Crop, Extractor, SigmaMode},
    solver::{Solution as T3Solution, SolveOptions, SolveStatus, Solver},
};

// Generate and publicly expose the protobuf code
pub mod proto {
    tonic::include_proto!("tetra3");
}

use proto::tetra3_service_server::Tetra3Service;

/// The core service holding persistent, thread-safe states for the solver and extractor.
pub struct Tetra3ServerImpl {
    pub solver: Arc<Mutex<Solver>>,
    pub extractor: Arc<Mutex<Extractor>>,
}

// --- Mapping Helpers ---

fn map_solve_options(opt: proto::SolveOptions) -> SolveOptions {
    let def = SolveOptions::default();

    let target_pixel = if opt.target_pixel.is_empty() {
        None
    } else {
        let mut arr = Array2::zeros((opt.target_pixel.len(), 2));
        for (i, p) in opt.target_pixel.iter().enumerate() {
            arr[[i, 0]] = p.y;
            arr[[i, 1]] = p.x;
        }
        Some(arr)
    };

    let target_sky_coord = if opt.target_sky_coord.is_empty() {
        None
    } else {
        let mut arr = Array2::zeros((opt.target_sky_coord.len(), 2));
        for (i, s) in opt.target_sky_coord.iter().enumerate() {
            arr[[i, 0]] = s.ra;
            arr[[i, 1]] = s.dec;
        }
        Some(arr)
    };

    SolveOptions {
        fov_estimate: opt.fov_estimate.or(def.fov_estimate),
        fov_max_error: opt.fov_max_error.or(def.fov_max_error),
        match_radius: opt.match_radius.unwrap_or(def.match_radius),
        match_threshold: opt.match_threshold.unwrap_or(def.match_threshold),
        solve_timeout_ms: opt.solve_timeout_ms.or(def.solve_timeout_ms),
        distortion: opt.distortion.or(def.distortion),
        match_max_error: opt.match_max_error.unwrap_or(def.match_max_error),
        return_matches: opt.return_matches.unwrap_or(def.return_matches),
        return_catalog: opt.return_catalog.unwrap_or(def.return_catalog),
        return_rotation_matrix: opt
            .return_rotation_matrix
            .unwrap_or(def.return_rotation_matrix),
        target_pixel,
        target_sky_coord,
    }
}

fn map_extract_config(cfg: proto::CentroidConfig) -> CentroidConfig {
    let def = CentroidConfig::default();

    let crop = cfg.crop.and_then(|c| c.crop_type).map(|ct| match ct {
        proto::crop::CropType::Fraction(f) => Crop::Fraction(f as usize),
        proto::crop::CropType::Center(c) => Crop::Center {
            height: c.height as usize,
            width: c.width as usize,
        },
        proto::crop::CropType::Region(r) => Crop::Region {
            height: r.height as usize,
            width: r.width as usize,
            offset_y: r.offset_y as isize,
            offset_x: r.offset_x as isize,
        },
    });

    let bg_sub_mode = match cfg.bg_sub_mode {
        Some(0) => Some(BgSubMode::LocalMedian),
        Some(1) => Some(BgSubMode::LocalMean),
        Some(2) => Some(BgSubMode::GlobalMedian),
        Some(3) => Some(BgSubMode::GlobalMean),
        _ => def.bg_sub_mode, // Fallback to None or default if not provided
    };

    let sigma_mode = match cfg.sigma_mode {
        Some(0) => SigmaMode::LocalMedianAbs,
        Some(1) => SigmaMode::LocalRootSquare,
        Some(2) => SigmaMode::GlobalMedianAbs,
        Some(3) => SigmaMode::GlobalRootSquare,
        _ => def.sigma_mode, // Fallback to default
    };

    CentroidConfig {
        sigma: cfg.sigma.unwrap_or(def.sigma),
        image_th: cfg.image_th.or(def.image_th),
        crop,
        downsample: cfg.downsample.map(|v| v as usize).or(def.downsample),
        filtsize: cfg.filtsize.map(|v| v as usize).unwrap_or(def.filtsize),
        bg_sub_mode,
        sigma_mode,
        binary_open: cfg.binary_open.unwrap_or(def.binary_open),
        centroid_window: cfg
            .centroid_window
            .map(|v| v as usize)
            .or(def.centroid_window),
        max_area: cfg.max_area.map(|v| v as usize).or(def.max_area),
        min_area: cfg.min_area.map(|v| v as usize).or(def.min_area),
        max_sum: cfg.max_sum.or(def.max_sum),
        min_sum: cfg.min_sum.or(def.min_sum),
        max_axis_ratio: cfg.max_axis_ratio.or(def.max_axis_ratio),
        max_returned: cfg.max_returned.map(|v| v as usize).or(def.max_returned),
        return_images: cfg.return_images.unwrap_or(def.return_images),
    }
}

fn map_solution(sol: T3Solution, extraction_time_ms: Option<f64>) -> proto::Solution {
    let status = match sol.status {
        SolveStatus::MatchFound => proto::SolveStatus::MatchFound,
        SolveStatus::NoMatch => proto::SolveStatus::NoMatch,
        SolveStatus::Timeout => proto::SolveStatus::Timeout,
        SolveStatus::Cancelled => proto::SolveStatus::Cancelled,
        SolveStatus::TooFew => proto::SolveStatus::TooFew,
    };

    let rotation_matrix = sol
        .rotation_matrix
        .map(|mat| mat.iter().cloned().collect::<Vec<f64>>())
        .unwrap_or_default();

    let matched_centroids = sol
        .matched_centroids
        .unwrap_or_default()
        .into_iter()
        .map(|c| proto::Pixel { y: c[0], x: c[1] })
        .collect();

    let matched_stars = sol
        .matched_stars
        .unwrap_or_default()
        .into_iter()
        .map(|s| proto::MatchedStar {
            ra: s[0],
            dec: s[1],
            mag: s[2],
        })
        .collect();

    let matched_cat_id = sol
        .matched_cat_id
        .unwrap_or_default()
        .into_iter()
        .map(|ids| proto::CatalogIds { ids })
        .collect();

    let catalog_stars = sol
        .catalog_stars
        .unwrap_or_default()
        .into_iter()
        .map(|(ra, dec, mag, y, x)| proto::CatalogStarInfo { ra, dec, mag, y, x })
        .collect();

    let target_y = sol
        .target_y
        .unwrap_or_default()
        .into_iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect();

    let target_x = sol
        .target_x
        .unwrap_or_default()
        .into_iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect();

    proto::Solution {
        ra: sol.ra,
        dec: sol.dec,
        roll: sol.roll,
        fov: sol.fov,
        distortion: sol.distortion,
        rmse: sol.rmse,
        p90e: sol.p90e,
        maxe: sol.maxe,
        matches: sol.matches.map(|v| v as u64),
        prob: sol.prob,
        epoch_equinox: sol.epoch_equinox,
        epoch_proper_motion: sol.epoch_proper_motion,
        status: status.into(),
        t_solve_ms: sol.t_solve_ms,
        rotation_matrix,
        target_ra: sol.target_ra.unwrap_or_default(),
        target_dec: sol.target_dec.unwrap_or_default(),
        target_y,
        target_x,
        matched_centroids,
        matched_stars,
        matched_cat_id,
        catalog_stars,
        extraction_time_ms,
    }
}

// --- Shared Memory Helper ---

fn process_shmem_extract(
    extractor: &mut Extractor,
    shmem_name: &str,
    width: usize,
    height: usize,
) -> Result<(tetra3::extractor::ExtractionResult, f64), Status> {
    let shmem = ShmemConf::new()
        .os_id(shmem_name)
        .open()
        .map_err(|e| Status::invalid_argument(format!("Failed to open shmem: {}", e)))?;

    let expected_size = width * height * std::mem::size_of::<f32>();
    if shmem.len() < expected_size {
        return Err(Status::invalid_argument(
            "Shared memory size is smaller than expected image size",
        ));
    }

    let slice = unsafe { std::slice::from_raw_parts(shmem.as_ptr() as *const f32, width * height) };
    let view = ArrayView2::from_shape((height, width), slice)
        .map_err(|e| Status::internal(format!("Failed to create ndarray view: {}", e)))?;

    let t0 = std::time::Instant::now();
    let result = extractor.extract(&view);
    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

    Ok((result, elapsed))
}

// --- Service Implementation ---

#[tonic::async_trait]
impl Tetra3Service for Tetra3ServerImpl {
    async fn solve(
        &self,
        request: Request<proto::SolveRequest>,
    ) -> Result<Response<proto::Solution>, Status> {
        let req = request.into_inner();
        let options = req.options.unwrap_or_default();
        let t3_opts = map_solve_options(options);

        let mut centroids_arr = Array2::zeros((req.centroids.len(), 2));
        for (i, p) in req.centroids.iter().enumerate() {
            centroids_arr[[i, 0]] = p.y;
            centroids_arr[[i, 1]] = p.x;
        }

        let mut solver = self.solver.lock().await;
        let solution = solver.solve_from_centroids(
            &centroids_arr,
            (req.image_height as f64, req.image_width as f64),
            t3_opts,
        );

        Ok(Response::new(map_solution(solution, None)))
    }

    async fn extract(
        &self,
        request: Request<proto::ExtractRequest>,
    ) -> Result<Response<proto::ExtractionResult>, Status> {
        let req = request.into_inner();
        let image = req
            .image
            .ok_or_else(|| Status::invalid_argument("Missing image payload"))?;
        let config = req.config.unwrap_or_default();

        let mut extractor = self.extractor.lock().await;
        extractor.config = map_extract_config(config);

        let (result, elapsed) = process_shmem_extract(
            &mut extractor,
            &image.shmem_name,
            image.width as usize,
            image.height as usize,
        )?;

        let proto_centroids = result
            .centroids
            .into_iter()
            .map(|c| proto::CentroidResult {
                y: c.y,
                x: c.x,
                sum: c.sum,
                area: c.area as u64,
                m2_xx: c.m2_xx,
                m2_yy: c.m2_yy,
                m2_xy: c.m2_xy,
                axis_ratio: c.axis_ratio,
            })
            .collect();

        Ok(Response::new(proto::ExtractionResult {
            centroids: proto_centroids,
            extraction_time_ms: elapsed,
        }))
    }

    async fn solve_from_image(
        &self,
        request: Request<proto::SolveFromImageRequest>,
    ) -> Result<Response<proto::Solution>, Status> {
        let req = request.into_inner();
        let image = req
            .image
            .ok_or_else(|| Status::invalid_argument("Missing image payload"))?;

        // 1. Extract
        let (extracted_centroids, ext_elapsed) = {
            let mut extractor = self.extractor.lock().await;
            if let Some(cfg) = req.extract_config {
                extractor.config = map_extract_config(cfg);
            }
            let (result, elapsed) = process_shmem_extract(
                &mut extractor,
                &image.shmem_name,
                image.width as usize,
                image.height as usize,
            )?;
            (result.centroids, elapsed)
        };

        // 2. Solve
        let mut centroids_arr = Array2::zeros((extracted_centroids.len(), 2));
        for (i, c) in extracted_centroids.iter().enumerate() {
            centroids_arr[[i, 0]] = c.y;
            centroids_arr[[i, 1]] = c.x;
        }

        let t3_opts = req.solve_options.map(map_solve_options).unwrap_or_default();

        let mut solver = self.solver.lock().await;
        let solution = solver.solve_from_centroids(
            &centroids_arr,
            (image.height as f64, image.width as f64),
            t3_opts,
        );

        Ok(Response::new(map_solution(solution, Some(ext_elapsed))))
    }
}
