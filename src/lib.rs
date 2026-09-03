// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

//! The `olive-solve` crate provides a robust celestial plate solver for stellar navigation.
//! It integrates the `tetra3` solver with hardware sensors (via `olive-imu`) to estimate
//! sky attitudes quickly and accurately.

use chrono::{Datelike, Timelike};
use ndarray::{Array2, ArrayBase, Data, Ix2};
use olive_imu::storage::PersistentStorage;
use olive_imu::{
    Imu, ImuDevice, MOVING_CAMERA_DEVIATION_TOLERANCE_DEG, MountCoordinates,
    STATIONARY_CAMERA_DEVIATION_TOLERANCE_DEG, STATIONARY_MOTION_THRESHOLD_DEG,
};
use std::sync::Arc;
use std::sync::RwLock;

#[cfg(feature = "python")]
pub mod python;
use std::time::SystemTime;
use tetra3::FastPixel;
use tetra3::extractor::{ExtractOptions, ExtractionResult, Extractor};
use tetra3::fast_extractor::{FastExtractOptions, FastExtractionResult, FastExtractor};
use tetra3::solver::{Solution, SolveOptions, SolveStatus, Solver};

/// Wrapper to inject custom IMU devices (e.g. for testing)
pub struct CustomImuWrapper(pub Box<dyn ImuDevice + Send>);

impl ImuDevice for CustomImuWrapper {
    fn init(&mut self) -> Result<(), String> {
        self.0.init()
    }
    fn poll(&mut self) -> Result<Vec<olive_imu::SensorEvent>, String> {
        self.0.poll()
    }
    fn revive(&mut self) -> Result<(), String> {
        self.0.revive()
    }
}

/// Defines the type of hardware IMU to start or fallback to.
pub enum ImuType {
    /// No IMU device used
    None,
    /// Automatically probe and use the first available supported IMU
    Auto,
    /// The BNO085 IMU over I2C
    Bno085,
    /// The BNO055 IMU over I2C
    Bno055,
    /// The BMI160 IMU over I2C
    Bmi160,
    /// MPU9250/MPU9150/MPU6500/MPU6050/MPU6000 IMU over I2C
    MpuXxxx,
    /// A custom mock or external IMU device
    Custom(Box<dyn ImuDevice + Send + Sync>),
}

impl std::str::FromStr for ImuType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" => Ok(ImuType::None),
            "auto" => Ok(ImuType::Auto),
            "bno085" => Ok(ImuType::Bno085),
            "bno055" => Ok(ImuType::Bno055),
            "bmi160" => Ok(ImuType::Bmi160),
            "mpuxxxx" => Ok(ImuType::MpuXxxx),
            _ => Err(format!("Unknown IMU type: {}", s)),
        }
    }
}

/// The source of the celestial position estimate.
#[derive(Debug, Clone, PartialEq)]
pub enum PositionSource {
    /// Position derived directly from a successful plate solve.
    Solver,
    /// Position estimated by the IMU using the last known solver anchor.
    Imu,
    /// The solver recently failed (e.g. due to movement or obstruction),
    /// and the IMU is unavailable to provide a real-time estimate.
    /// This represents the last known good plate-solved anchor, but it is physically stale.
    SolverStale,
}

/// A unified representation of the device's celestial orientation.
#[derive(Debug, Clone)]
pub struct Position {
    /// Right Ascension in degrees.
    pub ra: f64,
    /// Declination in degrees.
    pub dec: f64,
    /// Roll in degrees.
    pub roll: f64,
    /// The source that provided this position estimate.
    pub source: PositionSource,
    /// The time at which this position was valid.
    pub timestamp: SystemTime,
}

/// A structure capturing the raw hardware telemetry and tracking state from the IMU.
#[derive(Debug, Clone)]
pub struct SensorData {
    /// Time when the telemetry was recorded.
    pub timestamp: SystemTime,
    /// Raw gyroscope angular velocity [x, y, z] in rad/s.
    pub gyro: [f64; 3],
    /// Smoothed accelerometer gravity vector [x, y, z] in m/s^2.
    pub gravity_vector: Option<[f64; 3]>,
    /// The software-integrated relative tracking quaternion [i, j, k, w] purely from gyro integration.
    pub relative_gyro_quaternion: [f64; 4],
    /// Absolute hardware-fused orientation quaternion [i, j, k, w] (if supported).
    pub hardware_quaternion: Option<[f64; 4]>,
    /// The current classified state of motion (e.g. Stable, Moving).
    pub motion_state: olive_imu::MotionState,
}

/// The unified solver coordinating tetra3 plate solving and olive-imu hardware tracking.
/// Handles coordinate transformations between the equatorial and local horizontal frames.
pub struct FusedSolver {
    solver: Arc<RwLock<Option<Solver>>>,
    extractor: Arc<RwLock<Option<Extractor>>>,
    fast_extractor: Arc<RwLock<Option<FastExtractor>>>,
    imu: Arc<RwLock<Option<Arc<Imu>>>>,
    imu_type: Arc<RwLock<ImuType>>,
    storage: Option<Arc<dyn PersistentStorage>>,
    latest_solve_position: Arc<RwLock<Option<Position>>>,
    last_solve_failed: Arc<RwLock<bool>>,

    // Observer location required for Alt/Az IMU coordinate mapping
    latitude: Arc<RwLock<Option<f64>>>,
    longitude: Arc<RwLock<Option<f64>>>,
}

/// The result of a batch solve operation, containing the solution and metadata about the successful batch item.
pub struct BatchSolution {
    /// The actual solution produced by the solver.
    pub solution: Solution,
    /// The index of the centroid set in the batch that successfully produced the solution (or the index of the last set if all failed).
    pub crop_index: usize,
    /// The total elapsed time spent iterating through the batch and solving.
    pub solve_time: std::time::Duration,
}

impl FusedSolver {
    /// Initialize the unified solver.
    /// Takes in the database location, an optional IMU type (defaults to Auto),
    /// and an optional storage implementation. Location must be set later.
    pub fn new(
        database_path: &std::path::Path,
        imu_type: Option<ImuType>,
        storage: Option<Arc<dyn PersistentStorage>>,
    ) -> Result<Self, String> {
        let solver = Solver::load_database(database_path)
            .map_err(|e| format!("Failed to load database: {:?}", e))?;
        let extractor = Extractor::new();

        Ok(Self {
            solver: Arc::new(RwLock::new(Some(solver))),
            extractor: Arc::new(RwLock::new(Some(extractor))),
            fast_extractor: Arc::new(RwLock::new(None)),
            imu: Arc::new(RwLock::new(None)),
            imu_type: Arc::new(RwLock::new(imu_type.unwrap_or(ImuType::Auto))),
            storage,
            latest_solve_position: Arc::new(RwLock::new(None)),
            last_solve_failed: Arc::new(RwLock::new(false)),
            latitude: Arc::new(RwLock::new(None)),
            longitude: Arc::new(RwLock::new(None)),
        })
    }

    /// Sets the observer location. Must be called before IMU mapping can occur.
    pub fn set_observer_location(&self, lat: f64, lon: f64) {
        *self.latitude.write().unwrap() = Some(lat);
        *self.longitude.write().unwrap() = Some(lon);
    }

    /// Starts the IMU. Returns true if successful.
    pub fn start_imu(&self) -> Result<bool, String> {
        let mut imu_lock = self.imu.write().unwrap();
        if imu_lock.is_some() {
            return Err("IMU is already running.".into());
        }

        let mut imu_type = self.imu_type.write().unwrap();
        let imu_instance = match &mut *imu_type {
            ImuType::None => return Err("Cannot start IMU because ImuType is None.".into()),
            ImuType::Auto => {
                // Try Bno085 primary/alt
                if let Ok(dev) = olive_imu::bno085::Bno085Device::new(10, 0x4A, true, Some(3)) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else if let Ok(dev) =
                    olive_imu::bno085::Bno085Device::new(10, 0x4B, true, Some(3))
                {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else if let Ok(dev) = olive_imu::bno085::Bno085Device::new(10, 0x4A, true, None) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else if let Ok(dev) = olive_imu::bno085::Bno085Device::new(10, 0x4B, true, None) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else if let Ok(dev) = olive_imu::bno055::Bno055Device::new(10, 0x28) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else if let Ok(dev) = olive_imu::bno055::Bno055Device::new(10, 0x29) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else if let Ok(dev) = olive_imu::bmi160::Bmi160Device::new(0x68) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else if let Ok(dev) = olive_imu::bmi160::Bmi160Device::new(0x69) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else if let Ok(dev) = olive_imu::mpuxxxx::MpuXxxxDevice::new(10, 0x68) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else if let Ok(dev) = olive_imu::mpuxxxx::MpuXxxxDevice::new(10, 0x69) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else {
                    return Err("Auto mode could not find any supported IMU hardware.".into());
                }
            }
            ImuType::Bno085 => {
                if let Ok(dev) = olive_imu::bno085::Bno085Device::new(10, 0x4A, true, Some(3)) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else if let Ok(dev) =
                    olive_imu::bno085::Bno085Device::new(10, 0x4B, true, Some(3))
                {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else if let Ok(dev) = olive_imu::bno085::Bno085Device::new(10, 0x4A, true, None) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else if let Ok(dev) = olive_imu::bno085::Bno085Device::new(10, 0x4B, true, None) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else {
                    return Err("BNO085 hardware not found on I2C bus.".into());
                }
            }
            ImuType::Bno055 => {
                if let Ok(dev) = olive_imu::bno055::Bno055Device::new(10, 0x28) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else if let Ok(dev) = olive_imu::bno055::Bno055Device::new(10, 0x29) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else {
                    return Err("BNO055 hardware not found on I2C bus.".into());
                }
            }
            ImuType::Bmi160 => {
                if let Ok(dev) = olive_imu::bmi160::Bmi160Device::new(0x68) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else if let Ok(dev) = olive_imu::bmi160::Bmi160Device::new(0x69) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else {
                    return Err("BMI160 hardware not found on I2C bus.".into());
                }
            }
            ImuType::MpuXxxx => {
                if let Ok(dev) = olive_imu::mpuxxxx::MpuXxxxDevice::new(10, 0x68) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else if let Ok(dev) = olive_imu::mpuxxxx::MpuXxxxDevice::new(10, 0x69) {
                    Some(Imu::start(dev, self.storage.clone())?)
                } else {
                    return Err("MPU sensor hardware not found on I2C bus.".into());
                }
            }
            ImuType::Custom(_) => {
                let mut temp = ImuType::None;
                std::mem::swap(&mut *imu_type, &mut temp);
                if let ImuType::Custom(dev) = temp {
                    Some(Imu::start(CustomImuWrapper(dev), self.storage.clone())?)
                } else {
                    None
                }
            }
        };

        if let Some(instance) = imu_instance {
            *imu_lock = Some(Arc::new(instance));
            Ok(true)
        } else {
            Err("Failed to start IMU.".into())
        }
    }

    /// Stops the IMU and drops the background polling thread. Safe to call if not started.
    pub fn stop_imu(&self) -> Result<(), String> {
        let mut imu_lock = self.imu.write().unwrap();
        *imu_lock = None;
        Ok(())
    }

    /// Resets the internal IMU zero-bias and deletes the SVD calibration matrix from persistent storage.
    pub fn reset_calibration(&self) {
        if let Some(ref imu) = *self.imu.read().unwrap() {
            imu.clear_calibration();
            imu.reset_bias_calibration();
        }
    }

    // ==========================================
    // TETRA3 WRAPPERS
    // ==========================================

    /// Extracts star centroids using the standard pipeline.
    pub fn extract<S>(
        &self,
        image: &ArrayBase<S, Ix2>,
        options: ExtractOptions,
    ) -> Result<ExtractionResult, String>
    where
        S: Data<Elem = f32>,
    {
        let mut extractor_guard = self.extractor.write().unwrap();
        if let Some(extractor) = extractor_guard.as_mut() {
            Ok(extractor.extract(image, options))
        } else {
            Err("Extractor is not initialized.".into())
        }
    }

    /// Extracts star centroids using the fast sequential pipeline.
    pub fn extract_fast<S, T>(
        &self,
        image: &ArrayBase<S, Ix2>,
        options: FastExtractOptions,
    ) -> Result<FastExtractionResult, String>
    where
        S: Data<Elem = T>,
        T: FastPixel,
    {
        let mut extractor_guard = self.fast_extractor.write().unwrap();
        let (height, width) = image.dim();

        let reinit = match extractor_guard.as_ref() {
            Some(fe) => {
                fe.orig_width() != width || fe.orig_height() != height || fe.options() != &options
            }
            None => true,
        };

        if reinit {
            *extractor_guard = Some(FastExtractor::new(width, height, options));
        }

        let fe = extractor_guard.as_mut().unwrap();
        Ok(T::extract_fast(fe, image))
    }

    /// Extracts star centroids using multiple variants of fast extraction configurations.
    pub fn get_centroids_from_image_variants<S, T>(
        &self,
        image: &ArrayBase<S, Ix2>,
        options: FastExtractOptions,
        variants: &[tetra3::fast_extractor::FastExtractOptionsUpdate],
    ) -> Result<Vec<FastExtractionResult>, String>
    where
        S: Data<Elem = T>,
        T: FastPixel,
    {
        let mut extractor_guard = self.fast_extractor.write().unwrap();
        let (height, width) = image.dim();

        let reinit = match extractor_guard.as_ref() {
            Some(fe) => {
                fe.orig_width() != width || fe.orig_height() != height || fe.options() != &options
            }
            None => true,
        };

        if reinit {
            *extractor_guard = Some(FastExtractor::new(width, height, options));
        }

        let fe = extractor_guard.as_mut().unwrap();
        Ok(T::extract_variants_fast(fe, image, variants))
    }

    /// Performs a plate solve using pre-extracted centroids and given image dimensions.
    /// If successful, the solver automatically updates the IMU anchor internally.
    pub fn solve_from_centroids(
        &self,
        centroids: &Array2<f64>,
        size: (f64, f64),
        options: SolveOptions,
        timestamp: Option<SystemTime>,
    ) -> Result<Solution, String> {
        self.solve_from_centroids_batch(&[(centroids.clone(), None)], size, options, timestamp)
            .map(|batch_solution| batch_solution.solution)
    }

    /// Attempts to solve from multiple centroid sets in order, stopping when a solution is found.
    /// Updates internal solve state only when a solution is found or all given centroid sets fail.
    ///
    /// Returns a `BatchSolution` containing the successful `Solution`, the index of the
    /// centroid set in the batch that successfully produced it (or the index of the last set if all failed),
    /// and the total time taken to execute the batch.
    pub fn solve_from_centroids_batch(
        &self,
        centroids_batch: &[(Array2<f64>, Option<tetra3::extractor::Crop>)],
        main_size: (f64, f64),
        options: SolveOptions,
        timestamp: Option<SystemTime>,
    ) -> Result<BatchSolution, String> {
        let batch_start = std::time::Instant::now();
        let time = timestamp.unwrap_or_else(SystemTime::now);

        let mut last_solution = None;
        let mut last_index = 0;

        let mut solver_guard = self.solver.write().unwrap();
        let solver = solver_guard
            .as_mut()
            .ok_or_else(|| "Solver is not initialized.".to_string())?;

        let mut actual_options = options.clone();
        if actual_options.target_pixel.is_none() {
            actual_options.target_pixel = Some(
                ndarray::Array2::from_shape_vec((1, 2), vec![main_size.0 / 2.0, main_size.1 / 2.0])
                    .unwrap(),
            );
        }

        // Track the highest-confidence fallback match across all centroid crop attempts,
        // so we can attempt to rescue it using IMU verification if no perfect match is found.
        let mut best_low_conf_solution: Option<Solution> = None;
        let mut best_low_conf_prob = f64::INFINITY;
        let mut best_low_conf_index = 0;

        for (i, (centroids, crop_opt)) in centroids_batch.iter().enumerate() {
            let mut item_options = actual_options.clone();
            let mut item_size = main_size;
            let mut offset_x = 0.0;
            let mut offset_y = 0.0;

            if let Some(crop) = crop_opt {
                let (y_min, y_max, x_min, x_max) =
                    crop.bounds(main_size.1 as usize, main_size.0 as usize);
                item_size = ((y_max - y_min) as f64, (x_max - x_min) as f64);
                offset_y = y_min as f64;
                offset_x = x_min as f64;
                if let Some(fov) = item_options.fov_estimate {
                    // Calculate mathematically exact FOV based on gnomonic projection
                    let fov_rad = fov.to_radians();
                    let f = main_size.1 / (2.0 * (fov_rad / 2.0).tan());
                    let fov_crop_rad = 2.0 * ((item_size.1 / 2.0) / f).atan();
                    item_options.fov_estimate = Some(fov_crop_rad.to_degrees());
                }

                // Always allow out of bounds targets for crops, as a target on the main
                // image might naturally fall outside the crop's boundaries.
                item_options.allow_out_of_bounds_target_pixel = Some(true);

                // Disable radial distortion for virtual crops.
                // The optical center of the lens is outside the crop boundaries (for off-center crops).
                // Since tetra3 assumes the optical center is always at item_size / 2, leaving distortion
                // enabled will incorrectly warp the stars away from the crop's center instead of the lens center,
                // causing NoMatch. This also causes tetra3 to return distortion: None, which correctly
                // prevents engine.rs from feeding bogus values into the global distortion EMA.
                item_options.distortion = None;
            }

            // Adjust target_pixel for the crop offset
            if let Some(tp) = &mut item_options.target_pixel {
                for i in 0..tp.nrows() {
                    tp[[i, 0]] -= offset_y;
                    tp[[i, 1]] -= offset_x;
                }
            }

            // Shift centroids to crop origin
            let mut item_centroids = centroids.clone();
            if offset_y != 0.0 || offset_x != 0.0 {
                for i in 0..item_centroids.nrows() {
                    item_centroids[[i, 0]] -= offset_y;
                    item_centroids[[i, 1]] -= offset_x;
                }
            }

            let mut solution = solver.solve(&item_centroids, item_size, item_options);

            if solution.status == SolveStatus::MatchFound
                || solution.status == SolveStatus::LowConfidenceMatch
            {
                // Project target pixel coordinates back to the main image
                if let Some(ref mut target_y) = solution.target_y {
                    for y in target_y.iter_mut().flatten() {
                        *y += offset_y;
                    }
                }
                if let Some(ref mut target_x) = solution.target_x {
                    for x in target_x.iter_mut().flatten() {
                        *x += offset_x;
                    }
                }

                // If the client wants strict bounds checking against the main image size, enforce it
                if crop_opt.is_some() && !options.allow_out_of_bounds_target_pixel.unwrap_or(false)
                {
                    if let (Some(ref mut target_y), Some(ref mut target_x)) =
                        (&mut solution.target_y, &mut solution.target_x)
                    {
                        for (y, x) in target_y.iter_mut().zip(target_x.iter_mut()) {
                            if let (Some(vy), Some(vx)) = (*y, *x) {
                                if vy < 0.0 || vx < 0.0 || vy >= main_size.0 || vx >= main_size.1 {
                                    *y = None;
                                    *x = None;
                                }
                            }
                        }
                    }
                }
            }

            if solution.status == SolveStatus::MatchFound {
                last_solution = Some(solution);
                last_index = i;
                break;
            } else if solution.status == SolveStatus::LowConfidenceMatch {
                // Keep track of the lowest-probability LowConfidenceMatch so we can
                // fall back to it if no perfect MatchFound is discovered in this batch.
                let prob = solution.prob.unwrap_or(f64::INFINITY);
                if prob < best_low_conf_prob {
                    best_low_conf_prob = prob;
                    best_low_conf_solution = Some(solution.clone());
                    best_low_conf_index = i;
                }
            }

            last_solution = Some(solution);
            last_index = i;
        }

        let (mut solution, solve_crop_index) = if let Some(last) = last_solution {
            if last.status == SolveStatus::MatchFound {
                (last, last_index)
            } else if let Some(best_low_conf) = best_low_conf_solution {
                (best_low_conf, best_low_conf_index)
            } else {
                (last, last_index)
            }
        } else {
            return Err("No centroid sets provided.".to_string());
        };

        if solution.status == SolveStatus::MatchFound {
            let angle_moved = self.get_rotation_since_last_anchor(&time).unwrap_or(0.0);
            let max_allowed_error_deg = if angle_moved < STATIONARY_MOTION_THRESHOLD_DEG {
                STATIONARY_CAMERA_DEVIATION_TOLERANCE_DEG // Stationary since last anchor: reject false solves jumping too far
            } else {
                MOVING_CAMERA_DEVIATION_TOLERANCE_DEG // Physical slew occurred: tolerate gyro integration drift
            };

            let is_valid =
                match self.verify_solution_with_imu(&solution, time, max_allowed_error_deg) {
                    Some(valid) => valid,
                    None => true, // Initial bootstrap solve or no IMU: accept
                };

            if is_valid {
                *self.last_solve_failed.write().unwrap() = false;
                self.update_anchor_from_solution(&solution, time);
            } else {
                log::warn!(
                    "Rejected anomalous MatchFound: solve position deviated > {:.1}° from IMU (angle moved since last anchor: {:.2}°).",
                    max_allowed_error_deg,
                    angle_moved
                );
                *self.last_solve_failed.write().unwrap() = true;
                solution.status = SolveStatus::NoMatch;
            }
        } else if solution.status == SolveStatus::LowConfidenceMatch {
            // Only accept a low confidence fallback if the IMU can physically verify it
            // is within a tight 5-degree geometric cone of our expected orientation.
            let imu_verified = self.verify_solution_with_imu(&solution, time, 5.0);

            match imu_verified {
                Some(true) => {
                    solution.status = SolveStatus::MatchFound;
                    *self.last_solve_failed.write().unwrap() = false;

                    *self.latest_solve_position.write().unwrap() = Some(Position {
                        ra: solution.ra.unwrap_or(0.0),
                        dec: solution.dec.unwrap_or(0.0),
                        roll: solution.roll.unwrap_or(0.0),
                        source: PositionSource::Solver,
                        timestamp: time,
                    });

                    self.update_pointing_anchor_only_from_solution(&solution, time);
                }
                Some(false) => {
                    solution.status = SolveStatus::NoMatch;
                    *self.last_solve_failed.write().unwrap() = true;
                }
                None => {
                    *self.last_solve_failed.write().unwrap() = true;
                }
            }
        } else {
            *self.last_solve_failed.write().unwrap() = true;
        }

        Ok(BatchSolution {
            solution,
            crop_index: solve_crop_index,
            solve_time: batch_start.elapsed(),
        })
    }

    /// Evaluates a pre-computed `Solution` against an arbitrary list of centroids to
    /// determine the actual matching centroids.
    ///
    /// This is particularly useful when the original plate solve was performed on a
    /// truncated or cropped set of centroids (to improve performance), but you need
    /// to know which of the *full* image's centroids actually align with the
    /// recovered star field.
    ///
    /// # Arguments
    ///
    /// * `solution` - The `Solution` object returned from a successful plate solve.
    /// * `centroids` - An N x 2 array of `[y, x]` image centroids to test against the solution.
    /// * `size` - A tuple representing the `(height, width)` of the image.
    /// * `options` - The `SolveOptions` used to perform the solve, which supplies the matching radius.
    ///
    /// # Returns
    ///
    /// Returns `Some(Vec<[f64; 2]>)` containing the actual provided `image_centroids` that matched catalog stars,
    /// or `None` if the underlying solver has not been instantiated or the solution lacks
    /// a valid rotation matrix or FOV.
    pub fn get_matches_for_centroids(
        &self,
        solution: &Solution,
        centroids: &Array2<f64>,
        size: (f64, f64),
        options: &SolveOptions,
    ) -> Option<Vec<[f64; 2]>> {
        let mut solver_guard = self.solver.write().unwrap();
        let solver = solver_guard.as_mut()?;

        let mut centroids_vec = Vec::with_capacity(centroids.nrows());
        for row in centroids.rows() {
            centroids_vec.push([row[0], row[1]]);
        }

        solver.get_matches_for_centroids(solution, &centroids_vec, size, options)
    }

    /// Extracts star centroids from the image and performs a plate solve.
    /// If successful, the solver automatically updates the IMU anchor internally.
    ///
    /// Returns a tuple containing the `Solution` and the extraction time in milliseconds (`f64`).
    pub fn solve_from_image<S>(
        &self,
        image: &ArrayBase<S, Ix2>,
        extract_options: ExtractOptions,
        solve_options: SolveOptions,
        timestamp: Option<SystemTime>,
    ) -> Result<(Solution, f64), String>
    where
        S: Data<Elem = f32>,
    {
        let t0 = std::time::Instant::now();
        let centroids_result = self.extract(image, extract_options)?;
        let extract_time_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let num_centroids = centroids_result.centroids.len();
        let mut centroids_arr = Array2::zeros((num_centroids, 2));
        for (i, c) in centroids_result.centroids.iter().enumerate() {
            centroids_arr[[i, 0]] = c.y;
            centroids_arr[[i, 1]] = c.x;
        }

        let (height, width) = image.dim();
        let solution = self.solve_from_centroids(
            &centroids_arr,
            (height as f64, width as f64),
            solve_options,
            timestamp,
        )?;
        Ok((solution, extract_time_ms))
    }

    /// Extracts star centroids from the image using the fast pipeline and performs a plate solve.
    /// If virtual crops are provided, they are attempted first, and the full image is appended at the end as a fallback.
    ///
    /// Returns a tuple containing the `Solution` and the extraction time in milliseconds (`f64`).
    pub fn solve_from_image_fast<S, T>(
        &self,
        image: &ArrayBase<S, Ix2>,
        extract_options: FastExtractOptions,
        solve_options: SolveOptions,
        timestamp: Option<SystemTime>,
    ) -> Result<(Solution, f64), String>
    where
        S: Data<Elem = T>,
        T: FastPixel,
    {
        let t0 = std::time::Instant::now();
        let extract_result = self.extract_fast(image, extract_options.clone())?;
        let extract_time_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let mut centroid_arrays = Vec::new();

        if let (Some(vc), Some(crops)) = (
            &extract_result.virtual_crop_centroids,
            &extract_options.virtual_crops,
        ) {
            for (crop_cents, crop_def) in vc.iter().zip(crops.iter()) {
                let mut crop_arr = Array2::zeros((crop_cents.len(), 2));
                for (i, c) in crop_cents.iter().enumerate() {
                    crop_arr[[i, 0]] = c.y;
                    crop_arr[[i, 1]] = c.x;
                }
                centroid_arrays.push((crop_arr, Some(crop_def.clone())));
            }
        }

        let mut base_arr = Array2::zeros((extract_result.centroids.len(), 2));
        for (i, c) in extract_result.centroids.iter().enumerate() {
            base_arr[[i, 0]] = c.y;
            base_arr[[i, 1]] = c.x;
        }
        centroid_arrays.push((base_arr, None));

        let (height, width) = image.dim();
        let solution = self
            .solve_from_centroids_batch(
                &centroid_arrays,
                (height as f64, width as f64),
                solve_options,
                timestamp,
            )?
            .solution;
        Ok((solution, extract_time_ms))
    }

    fn update_anchor_from_solution(&self, solution: &Solution, time: SystemTime) {
        let ra = if let (Some(t_ra), Some(_t_dec)) = (&solution.target_ra, &solution.target_dec) {
            t_ra[0]
        } else {
            solution.ra.unwrap_or(0.0)
        };
        let dec = if let (Some(_t_ra), Some(t_dec)) = (&solution.target_ra, &solution.target_dec) {
            t_dec[0]
        } else {
            solution.dec.unwrap_or(0.0)
        };
        let roll = solution.roll.unwrap_or(0.0);

        if let Some(ref imu) = *self.imu.read().unwrap() {
            let lat_opt = *self.latitude.read().unwrap();
            let lon_opt = *self.longitude.read().unwrap();

            if let (Some(lat), Some(lon)) = (lat_opt, lon_opt) {
                let dt: chrono::DateTime<chrono::Utc> = time.into();

                let (alt, az, alt_az_roll) = ra_dec_to_alt_az(ra, dec, roll, lat, lon, dt);

                let mount_coords = MountCoordinates {
                    pitch: alt,
                    yaw: az,
                    roll: alt_az_roll,
                };
                imu.update_anchor(&mount_coords, &time);
            }
        }

        *self.latest_solve_position.write().unwrap() = Some(Position {
            ra,
            dec,
            roll,
            source: PositionSource::Solver,
            timestamp: time,
        });
    }

    fn update_pointing_anchor_only_from_solution(&self, solution: &Solution, time: SystemTime) {
        let ra = if let (Some(t_ra), Some(_t_dec)) = (&solution.target_ra, &solution.target_dec) {
            t_ra[0]
        } else {
            solution.ra.unwrap_or(0.0)
        };
        let dec = if let (Some(_t_ra), Some(t_dec)) = (&solution.target_ra, &solution.target_dec) {
            t_dec[0]
        } else {
            solution.dec.unwrap_or(0.0)
        };
        let roll = solution.roll.unwrap_or(0.0);

        if let Some(ref imu) = *self.imu.read().unwrap() {
            let lat_opt = *self.latitude.read().unwrap();
            let lon_opt = *self.longitude.read().unwrap();

            if let (Some(lat), Some(lon)) = (lat_opt, lon_opt) {
                let dt: chrono::DateTime<chrono::Utc> = time.into();
                let (alt, az, alt_az_roll) = ra_dec_to_alt_az(ra, dec, roll, lat, lon, dt);

                let mount_coords = MountCoordinates {
                    pitch: alt,
                    yaw: az,
                    roll: alt_az_roll,
                };
                imu.update_pointing_anchor_only(&mount_coords, &time);
            }
        }
    }

    fn verify_solution_with_imu(
        &self,
        solution: &Solution,
        time: SystemTime,
        max_dist_deg: f64,
    ) -> Option<bool> {
        let ra = if let (Some(t_ra), Some(_t_dec)) = (&solution.target_ra, &solution.target_dec) {
            t_ra[0]
        } else {
            solution.ra.unwrap_or(0.0)
        };
        let dec = if let (Some(_t_ra), Some(t_dec)) = (&solution.target_ra, &solution.target_dec) {
            t_dec[0]
        } else {
            solution.dec.unwrap_or(0.0)
        };

        let imu_guard = self.imu.read().unwrap();
        if let Some(ref imu) = *imu_guard {
            let lat_opt = *self.latitude.read().unwrap();
            let lon_opt = *self.longitude.read().unwrap();

            if let (Some(lat), Some(lon)) = (lat_opt, lon_opt) {
                if let Ok((est, _)) = imu.get_estimated_pointing(&time) {
                    let dt: chrono::DateTime<chrono::Utc> = time.into();
                    let (imu_ra, imu_dec, _) =
                        alt_az_to_ra_dec(est.pitch, est.yaw, est.roll, lat, lon, dt);

                    let dist = angular_separation_deg(ra, dec, imu_ra, imu_dec);
                    return Some(dist <= max_dist_deg);
                }
            }
        }
        None
    }

    // ==========================================
    // IMU STATUS WRAPPERS
    // ==========================================

    /// Retrieves the real-time calibration metrics from the IMU hardware, if running.
    pub fn get_calibration_status(&self) -> Option<olive_imu::TransformMetrics> {
        if let Some(ref imu) = *self.imu.read().unwrap() {
            imu.get_calibration_metrics()
        } else {
            None
        }
    }

    /// Retrieves the real-time motion stability state from the IMU hardware, if running.
    pub fn get_motion_state(&self) -> Option<olive_imu::MotionState> {
        (*self.imu.read().unwrap())
            .as_ref()
            .map(|imu| imu.get_motion_state())
    }

    /// Returns the angular movement (in degrees) measured by the IMU since the last plate-solve anchor was established.
    pub fn get_rotation_since_last_anchor(&self, timestamp: &SystemTime) -> Option<f64> {
        (*self.imu.read().unwrap())
            .as_ref()
            .and_then(|imu| imu.get_rotation_since_last_anchor(timestamp))
    }

    /// Fetches the current real-time hardware telemetry from the IMU.
    /// Returns None if the IMU is not running.
    pub fn get_sensor_data(&self) -> Option<SensorData> {
        if let Some(ref imu) = *self.imu.read().unwrap() {
            if let Some(state) = imu.get_latest_state() {
                return Some(SensorData {
                    timestamp: state.timestamp,
                    gyro: [state.gyro.x, state.gyro.y, state.gyro.z],
                    gravity_vector: state.gravity_vector.map(|a| [a.x, a.y, a.z]),
                    relative_gyro_quaternion: [
                        state.quaternion.i,
                        state.quaternion.j,
                        state.quaternion.k,
                        state.quaternion.w,
                    ],
                    hardware_quaternion: state.hardware_quaternion.map(|q| [q.i, q.j, q.k, q.w]),
                    motion_state: state.motion_state,
                });
            }
        }
        None
    }

    /// Fetches the latest known orientation of the device.
    /// If the IMU is actively tracking and has a valid plate solve anchor, this returns the real-time IMU estimate.
    /// Otherwise, it safely falls back to returning the position from the last successful plate solve.
    pub fn get_latest_position(&self) -> Option<Position> {
        let mut last_solve = self.latest_solve_position.read().unwrap().clone();
        let last_failed = *self.last_solve_failed.read().unwrap();

        if let Some(ref imu) = *self.imu.read().unwrap() {
            if let Ok((est, is_imu_estimate)) = imu.get_estimated_pointing(&SystemTime::now()) {
                let lat_opt = *self.latitude.read().unwrap();
                let lon_opt = *self.longitude.read().unwrap();

                if let (Some(lat), Some(lon)) = (lat_opt, lon_opt) {
                    let dt_now = chrono::Utc::now();
                    let (current_ra, current_dec, current_roll) =
                        alt_az_to_ra_dec(est.pitch, est.yaw, est.roll, lat, lon, dt_now);

                    let source = if is_imu_estimate {
                        PositionSource::Imu
                    } else if !last_failed {
                        PositionSource::Solver
                    } else if imu.is_calibrated() {
                        // In this case our position estimate is stale, but the IMU hasn't detected enough
                        // movement so we consider the estimate as coming from the IMU.
                        PositionSource::Imu
                    } else {
                        PositionSource::SolverStale
                    };

                    return Some(Position {
                        ra: current_ra,
                        dec: current_dec,
                        roll: current_roll,
                        source,
                        timestamp: SystemTime::now(),
                    });
                }
            }
        }

        if let Some(pos) = &mut last_solve {
            if last_failed {
                pos.source = PositionSource::SolverStale;
            }
        }

        last_solve
    }
}

// ==========================================
// MEEUS CELESTIAL TRANSFORMATIONS
// ==========================================

fn ra_dec_to_alt_az(
    ra_deg: f64,
    dec_deg: f64,
    roll_deg: f64,
    lat_deg: f64,
    lon_deg: f64,
    utc_time: chrono::DateTime<chrono::Utc>,
) -> (f64, f64, f64) {
    let y = utc_time.year() as f64;
    let m = utc_time.month() as f64;
    let d = utc_time.day() as f64;
    let h = utc_time.hour() as f64
        + utc_time.minute() as f64 / 60.0
        + utc_time.second() as f64 / 3600.0;

    let (mut y_jd, mut m_jd) = (y, m);
    if m <= 2.0 {
        y_jd -= 1.0;
        m_jd += 12.0;
    }

    let a = (y_jd / 100.0).floor();
    let b = 2.0 - a + (a / 4.0).floor();
    let jd =
        (365.25 * (y_jd + 4716.0)).floor() + (30.6001 * (m_jd + 1.0)).floor() + d + h / 24.0 + b
            - 1524.5;

    let d_jd = jd - 2451545.0; // days since J2000.0
    let mut gmst = 280.46061837 + 360.98564736629 * d_jd;
    gmst %= 360.0;
    if gmst < 0.0 {
        gmst += 360.0;
    }

    let mut lst = gmst + lon_deg;
    lst %= 360.0;
    if lst < 0.0 {
        lst += 360.0;
    }

    let mut ha = lst - ra_deg;
    ha %= 360.0;
    if ha < 0.0 {
        ha += 360.0;
    }

    let ha_rad = ha.to_radians();
    let dec_rad = dec_deg.to_radians();
    let lat_rad = lat_deg.to_radians();

    let sin_alt = dec_rad.sin() * lat_rad.sin() + dec_rad.cos() * lat_rad.cos() * ha_rad.cos();
    let alt_rad = sin_alt.clamp(-1.0, 1.0).asin();
    let alt_deg = alt_rad.to_degrees();

    let cos_az = (dec_rad.sin() - sin_alt * lat_rad.sin()) / (alt_rad.cos() * lat_rad.cos());
    let mut az_rad = cos_az.clamp(-1.0, 1.0).acos();
    if ha_rad.sin() > 0.0 {
        az_rad = std::f64::consts::TAU - az_rad;
    }
    let az_deg = az_rad.to_degrees();

    let q_rad = (ha_rad.sin()).atan2(lat_rad.tan() * dec_rad.cos() - dec_rad.sin() * ha_rad.cos());
    let q_deg = q_rad.to_degrees();

    let alt_az_roll = roll_deg + q_deg;

    (alt_deg, az_deg, alt_az_roll)
}

fn alt_az_to_ra_dec(
    alt_deg: f64,
    az_deg: f64,
    alt_az_roll: f64,
    lat_deg: f64,
    lon_deg: f64,
    utc_time: chrono::DateTime<chrono::Utc>,
) -> (f64, f64, f64) {
    let y = utc_time.year() as f64;
    let m = utc_time.month() as f64;
    let d = utc_time.day() as f64;
    let h = utc_time.hour() as f64
        + utc_time.minute() as f64 / 60.0
        + utc_time.second() as f64 / 3600.0;

    let (mut y_jd, mut m_jd) = (y, m);
    if m <= 2.0 {
        y_jd -= 1.0;
        m_jd += 12.0;
    }

    let a = (y_jd / 100.0).floor();
    let b = 2.0 - a + (a / 4.0).floor();
    let jd =
        (365.25 * (y_jd + 4716.0)).floor() + (30.6001 * (m_jd + 1.0)).floor() + d + h / 24.0 + b
            - 1524.5;

    let d_jd = jd - 2451545.0;
    let mut gmst = 280.46061837 + 360.98564736629 * d_jd;
    gmst %= 360.0;
    if gmst < 0.0 {
        gmst += 360.0;
    }

    let mut lst = gmst + lon_deg;
    lst %= 360.0;
    if lst < 0.0 {
        lst += 360.0;
    }

    let alt_rad = alt_deg.to_radians();
    let az_rad = az_deg.to_radians();
    let lat_rad = lat_deg.to_radians();

    let sin_dec = alt_rad.sin() * lat_rad.sin() + alt_rad.cos() * lat_rad.cos() * az_rad.cos();
    let dec_rad = sin_dec.clamp(-1.0, 1.0).asin();
    let dec_deg = dec_rad.to_degrees();

    let cos_ha = (alt_rad.sin() - sin_dec * lat_rad.sin()) / (dec_rad.cos() * lat_rad.cos());
    let mut ha_rad = cos_ha.clamp(-1.0, 1.0).acos();
    if az_rad.sin() > 0.0 {
        ha_rad = std::f64::consts::TAU - ha_rad;
    }
    let ha_deg = ha_rad.to_degrees();

    let mut ra_deg = lst - ha_deg;
    ra_deg %= 360.0;
    if ra_deg < 0.0 {
        ra_deg += 360.0;
    }

    let q_rad = (ha_rad.sin()).atan2(lat_rad.tan() * dec_rad.cos() - dec_rad.sin() * ha_rad.cos());
    let q_deg = q_rad.to_degrees();

    let eq_roll = alt_az_roll - q_deg;

    (ra_deg, dec_deg, eq_roll)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A dummy IMU for testing Custom IMU injections
    struct MockImu;
    impl ImuDevice for MockImu {
        fn init(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn poll(&mut self) -> Result<Vec<olive_imu::SensorEvent>, String> {
            Ok(vec![])
        }
        fn revive(&mut self) -> Result<(), String> {
            Ok(())
        }
    }

    #[test]
    fn test_start_imu_not_found() {
        let fs = FusedSolver {
            solver: Arc::new(RwLock::new(None)),
            extractor: Arc::new(RwLock::new(None)),
            fast_extractor: Arc::new(RwLock::new(None)),
            imu: Arc::new(RwLock::new(None)),
            imu_type: Arc::new(RwLock::new(ImuType::None)),
            storage: None,
            latest_solve_position: Arc::new(RwLock::new(None)),
            last_solve_failed: Arc::new(RwLock::new(false)),
            latitude: Arc::new(RwLock::new(None)),
            longitude: Arc::new(RwLock::new(None)),
        };

        let result = fs.start_imu();
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Cannot start IMU because ImuType is None."
        );
    }

    #[test]
    fn test_start_imu_double_start() {
        let fs = FusedSolver {
            solver: Arc::new(RwLock::new(None)),
            extractor: Arc::new(RwLock::new(None)),
            fast_extractor: Arc::new(RwLock::new(None)),
            imu: Arc::new(RwLock::new(None)),
            imu_type: Arc::new(RwLock::new(ImuType::Custom(Box::new(MockImu)))),
            storage: None,
            latest_solve_position: Arc::new(RwLock::new(None)),
            last_solve_failed: Arc::new(RwLock::new(false)),
            latitude: Arc::new(RwLock::new(None)),
            longitude: Arc::new(RwLock::new(None)),
        };

        let first = fs.start_imu();
        assert!(first.is_ok());

        let second = fs.start_imu();
        assert!(second.is_err());
        assert_eq!(second.unwrap_err(), "IMU is already running.");
    }

    #[test]
    fn test_get_sensor_data_empty_when_not_started() {
        let fs = FusedSolver {
            solver: Arc::new(RwLock::new(None)),
            extractor: Arc::new(RwLock::new(None)),
            fast_extractor: Arc::new(RwLock::new(None)),
            imu: Arc::new(RwLock::new(None)),
            imu_type: Arc::new(RwLock::new(ImuType::None)),
            storage: None,
            latest_solve_position: Arc::new(RwLock::new(None)),
            last_solve_failed: Arc::new(RwLock::new(false)),
            latitude: Arc::new(RwLock::new(None)),
            longitude: Arc::new(RwLock::new(None)),
        };

        let data = fs.get_sensor_data();
        assert!(data.is_none());
    }

    #[test]
    fn test_safe_stop() {
        let fs = FusedSolver {
            solver: Arc::new(RwLock::new(None)),
            extractor: Arc::new(RwLock::new(None)),
            fast_extractor: Arc::new(RwLock::new(None)),
            imu: Arc::new(RwLock::new(None)),
            imu_type: Arc::new(RwLock::new(ImuType::None)),
            storage: None,
            latest_solve_position: Arc::new(RwLock::new(None)),
            last_solve_failed: Arc::new(RwLock::new(false)),
            latitude: Arc::new(RwLock::new(None)),
            longitude: Arc::new(RwLock::new(None)),
        };

        let result = fs.stop_imu();
        assert!(result.is_ok()); // Safe to call even when not started
    }
    #[test]
    fn test_fallback_position() {
        let fs = FusedSolver {
            solver: Arc::new(RwLock::new(None)),
            extractor: Arc::new(RwLock::new(None)),
            fast_extractor: Arc::new(RwLock::new(None)),
            imu: Arc::new(RwLock::new(None)),
            imu_type: Arc::new(RwLock::new(ImuType::None)),
            storage: None,
            latest_solve_position: Arc::new(RwLock::new(Some(Position {
                ra: 10.0,
                dec: 20.0,
                roll: 0.0,
                source: PositionSource::Solver,
                timestamp: std::time::SystemTime::UNIX_EPOCH,
            }))),
            last_solve_failed: Arc::new(RwLock::new(false)),
            latitude: Arc::new(RwLock::new(None)),
            longitude: Arc::new(RwLock::new(None)),
        };

        let pos = fs.get_latest_position();
        assert!(pos.is_some());
        let pos = pos.unwrap();
        assert_eq!(pos.ra, 10.0);
        assert_eq!(pos.source, PositionSource::Solver);
    }

    #[test]
    fn test_coordinate_transforms() {
        use crate::{alt_az_to_ra_dec, ra_dec_to_alt_az};
        use chrono::TimeZone;

        let time = chrono::Utc.with_ymd_and_hms(2025, 1, 1, 12, 0, 0).unwrap();
        let lat = 34.0522; // Los Angeles
        let lon = -118.2437;

        let original_ra = 45.0;
        let original_dec = 30.0;
        let original_roll = 10.0;

        let (alt, az, roll) =
            ra_dec_to_alt_az(original_ra, original_dec, original_roll, lat, lon, time);

        let (ra, dec, r_roll) = alt_az_to_ra_dec(alt, az, roll, lat, lon, time);

        assert!((original_ra - ra).abs() < 1e-6 || (original_ra - ra).abs() > 360.0 - 1e-6);
        assert!((original_dec - dec).abs() < 1e-6);
        // Roll might be shifted by 360
        assert!(
            (original_roll - r_roll).abs() < 1e-6 || (original_roll - r_roll).abs() > 360.0 - 1e-6
        );
    }

    #[test]
    fn test_imu_fallback_when_no_anchor() {
        let fs = FusedSolver {
            solver: Arc::new(RwLock::new(None)),
            extractor: Arc::new(RwLock::new(None)),
            fast_extractor: Arc::new(RwLock::new(None)),
            imu: Arc::new(RwLock::new(None)),
            imu_type: Arc::new(RwLock::new(ImuType::Custom(Box::new(MockImu)))),
            storage: None,
            latest_solve_position: Arc::new(RwLock::new(None)),
            last_solve_failed: Arc::new(RwLock::new(false)),
            latitude: Arc::new(RwLock::new(None)),
            longitude: Arc::new(RwLock::new(None)),
        };

        fs.set_observer_location(34.0, -118.0);
        fs.start_imu().unwrap();

        let sol = tetra3::solver::Solution {
            ra: Some(100.0),
            dec: Some(50.0),
            roll: Some(0.0),
            status: tetra3::solver::SolveStatus::MatchFound,
            ..Default::default()
        };

        fs.update_anchor_from_solution(&sol, std::time::SystemTime::now());

        std::thread::sleep(std::time::Duration::from_millis(50));

        let pos = fs.get_latest_position();
        assert!(pos.is_some());
        let pos = pos.unwrap();
        // The IMU hasn't received any gyro data to establish a history, so update_anchor will silently fail.
        // Thus get_estimated_pointing will return Err, and we should correctly fallback to the solver's position.
        assert_eq!(pos.source, PositionSource::Solver);
    }

    #[test]
    fn test_uninitialized_solver_fails() {
        let fs = FusedSolver {
            solver: Arc::new(RwLock::new(None)), // Solver not initialized
            extractor: Arc::new(RwLock::new(None)),
            fast_extractor: Arc::new(RwLock::new(None)),
            imu: Arc::new(RwLock::new(None)),
            imu_type: Arc::new(RwLock::new(ImuType::None)),
            storage: None,
            latest_solve_position: Arc::new(RwLock::new(None)),
            last_solve_failed: Arc::new(RwLock::new(false)),
            latitude: Arc::new(RwLock::new(None)),
            longitude: Arc::new(RwLock::new(None)),
        };

        use tetra3::solver::SolveOptions;
        let options = SolveOptions::default();
        let result = fs.solve_from_centroids(
            &ndarray::Array2::zeros((0, 2)),
            (100.0, 100.0),
            options,
            None,
        );

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Solver is not initialized.");
    }

    #[test]
    fn test_observer_location_updates() {
        let fs = FusedSolver {
            solver: Arc::new(RwLock::new(None)),
            extractor: Arc::new(RwLock::new(None)),
            fast_extractor: Arc::new(RwLock::new(None)),
            imu: Arc::new(RwLock::new(None)),
            imu_type: Arc::new(RwLock::new(ImuType::None)),
            storage: None,
            latest_solve_position: Arc::new(RwLock::new(None)),
            last_solve_failed: Arc::new(RwLock::new(false)),
            latitude: Arc::new(RwLock::new(None)),
            longitude: Arc::new(RwLock::new(None)),
        };

        assert!(fs.latitude.read().unwrap().is_none());
        assert!(fs.longitude.read().unwrap().is_none());

        fs.set_observer_location(45.0, -90.0);

        assert_eq!(*fs.latitude.read().unwrap(), Some(45.0));
        assert_eq!(*fs.longitude.read().unwrap(), Some(-90.0));
    }

    #[test]
    fn test_imu_status_wrappers_without_imu() {
        let fs = FusedSolver {
            solver: Arc::new(RwLock::new(None)),
            extractor: Arc::new(RwLock::new(None)),
            fast_extractor: Arc::new(RwLock::new(None)),
            imu: Arc::new(RwLock::new(None)),
            imu_type: Arc::new(RwLock::new(ImuType::None)),
            storage: None,
            latest_solve_position: Arc::new(RwLock::new(None)),
            last_solve_failed: Arc::new(RwLock::new(false)),
            latitude: Arc::new(RwLock::new(None)),
            longitude: Arc::new(RwLock::new(None)),
        };

        // When IMU isn't running, these should safely return None without panicking
        assert!(fs.get_calibration_status().is_none());
        assert!(fs.get_motion_state().is_none());

        // Resetting calibration should also safely do nothing
        fs.reset_calibration();
    }

    #[test]
    fn test_virtual_crops_batch_solve() {
        use crate::FusedSolver;
        use std::fs::File;
        use std::path::Path;
        use tetra3::{SolveStatus, Solver, extractor::Crop};
        use zip::ZipArchive;

        let db_path = Path::new("tetra3/tests/fixtures/default_database.npz");
        let zip_path = Path::new("tetra3/tests/fixtures/solver_fixtures.zip");

        if !db_path.exists() {
            eprintln!("Skipping test: default_database.npz not found.");
            return;
        }
        if !zip_path.exists() {
            panic!(
                "Fixture zip not found! Run `cargo test generate_test_fixtures --release -- --ignored` first."
            );
        }

        let _solver = Solver::load_database(db_path).expect("Failed to load Tetra3 database");

        // We wrap it in a FusedSolver
        let fs = FusedSolver::new(db_path, None, None).unwrap();
        // FusedSolver::new already loaded the DB, so we don't need to overwrite it, but we can if we want.
        // *fs.solver.write().unwrap() = Some(solver);

        let zip_file = File::open(zip_path).expect("Failed to open solver_fixtures.zip");
        let mut archive = ZipArchive::new(zip_file).expect("Failed to open zip archive");

        // Read Input DTO
        let input_filename = format!("input_1.json");
        let mut input_buffer = Vec::new();
        {
            use std::io::Read;
            let mut req_file = archive.by_name(&input_filename).unwrap();
            req_file.read_to_end(&mut input_buffer).unwrap();
        }

        // Use a generic JSON value so we don't have to duplicate the DTO structs from validate_solver.rs
        let input_dto: serde_json::Value = serde_json::from_slice(&input_buffer).unwrap();

        let centroids = input_dto["centroids"].as_array().unwrap();
        let image_height = input_dto["image_height"].as_f64().unwrap();
        let image_width = input_dto["image_width"].as_f64().unwrap();

        let mut flat_cents = Vec::with_capacity(centroids.len() * 2);
        for c in centroids {
            let arr = c.as_array().unwrap();
            flat_cents.push(arr[0].as_f64().unwrap());
            flat_cents.push(arr[1].as_f64().unwrap());
        }
        let centroids_array =
            ndarray::Array2::from_shape_vec((centroids.len(), 2), flat_cents).unwrap();

        // To make the comparison identical, let's treat the "full image" as being larger,
        // and the "crop" as being the exact 512x512 region at offset (100, 100).
        let offset_y = 100.0;
        let offset_x = 100.0;
        let main_h = image_height + 200.0;
        let main_w = image_width + 200.0;
        let main_size = (main_h, main_w);

        let mut full_centroids = centroids_array.clone();
        for i in 0..full_centroids.nrows() {
            full_centroids[[i, 0]] += offset_y;
            full_centroids[[i, 1]] += offset_x;
        }

        // 1. Solve on the original FULL image blindly so it deduces its own true FOV and center
        let mut base_options = tetra3::solver::SolveOptions::default();
        base_options.fov_estimate = None;

        let result_init = fs
            .solve_from_centroids(&full_centroids, main_size, base_options.clone(), None)
            .unwrap();
        assert_eq!(result_init.status, SolveStatus::MatchFound);

        let ra = result_init.ra.unwrap();
        let dec = result_init.dec.unwrap();

        // Choose a target coordinate ~8 degrees away in Dec
        let target_dec = dec + 8.0;
        let target_ra = ra;

        let mut target_sky_coord = ndarray::Array2::<f64>::zeros((1, 2));
        target_sky_coord[[0, 0]] = target_ra;
        target_sky_coord[[0, 1]] = target_dec;
        base_options.target_sky_coord = Some(target_sky_coord);
        base_options.allow_out_of_bounds_target_pixel = Some(true);

        // Re-solve the full image with the target pixel so we have a baseline to compare against
        let result = fs
            .solve_from_centroids(&full_centroids, main_size, base_options.clone(), None)
            .unwrap();

        // 2. Pretend our centroids are actually from a virtual crop (e.g., offset by 100 pixels in X and Y)
        // Crop bounds. Crop::Region uses absolute coordinates in the main image.
        let crop_def = Crop::Region {
            height: image_height as usize,
            width: image_width as usize,
            offset_y: offset_y as isize,
            offset_x: offset_x as isize,
        };

        // Run the batch solver.
        // We will pass the crop! For a virtual crop, the engine passes the points in FULL coordinates!
        // Wait, does the batch solver expect full coordinates or cropped coordinates?
        // Let's check `olive-engine/src/engine.rs`: engine extracts crops, and the points are in the FULL image coordinate system.
        // And then in `solve_from_centroids_batch`, it shifts them: `item_centroids[[i, 0]] -= offset_y`.
        // So we just pass `full_centroids` to the batch solver!
        let batch = vec![(full_centroids, Some(crop_def))];

        // We expect the solver to find a match, AND correctly calculate target_y and target_x
        // because it overrides allow_out_of_bounds_target_pixel to true for crops!
        let crop_result = fs
            .solve_from_centroids_batch(&batch, main_size, base_options.clone(), None)
            .unwrap()
            .solution;

        assert_eq!(crop_result.status, SolveStatus::MatchFound);

        let x = crop_result.target_x.as_ref().unwrap()[0];
        let y = crop_result.target_y.as_ref().unwrap()[0];

        assert!(x.is_some());
        assert!(y.is_some());

        let x_val = x.unwrap();
        let y_val = y.unwrap();

        let base_x = result.target_x.as_ref().unwrap()[0].unwrap();
        let base_y = result.target_y.as_ref().unwrap()[0].unwrap();

        // Assert the target pixel from the crop solve correctly projected to the exact same place
        // on the main image coordinates as if it were solved natively on the main image.
        // Due to scaling and polynomial distortion differences between solving a 712x712 image vs a 512x512 image,
        // there is a tiny sub-pixel numerical difference over long projections (8 degrees away). We allow a 2.0px epsilon.
        let epsilon = 2.0;

        assert!(
            (x_val - base_x).abs() < epsilon,
            "Target X mismatched: crop={}, base={}",
            x_val,
            base_x
        );
        assert!(
            (y_val - base_y).abs() < epsilon,
            "Target Y mismatched: crop={}, base={}",
            y_val,
            base_y
        );
    }
}

/// Computes the great-circle angular distance between two sky coordinates in degrees.
pub fn angular_separation_deg(ra1_deg: f64, dec1_deg: f64, ra2_deg: f64, dec2_deg: f64) -> f64 {
    let ra1_rad = ra1_deg.to_radians();
    let dec1_rad = dec1_deg.to_radians();
    let ra2_rad = ra2_deg.to_radians();
    let dec2_rad = dec2_deg.to_radians();

    let cos_theta = dec1_rad.sin() * dec2_rad.sin()
        + dec1_rad.cos() * dec2_rad.cos() * (ra1_rad - ra2_rad).cos();
    cos_theta.clamp(-1.0, 1.0).acos().to_degrees()
}

#[cfg(test)]
mod new_tests {
    use super::*;

    struct MockImuWithData;
    impl ImuDevice for MockImuWithData {
        fn init(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn poll(&mut self) -> Result<Vec<olive_imu::SensorEvent>, String> {
            std::thread::sleep(std::time::Duration::from_millis(10));
            // Return identity quaternion constantly, meaning IMU reports no physical movement.
            Ok(vec![olive_imu::SensorEvent {
                gyro: Some(nalgebra::Vector3::new(0.0, 0.0, 0.0)),
                dt: Some(0.01),
                hardware_quaternion: Some(nalgebra::UnitQuaternion::from_quaternion(
                    nalgebra::Quaternion::new(1.0, 0.0, 0.0, 0.0),
                )),
                ..Default::default()
            }])
        }
        fn revive(&mut self) -> Result<(), String> {
            Ok(())
        }
    }

    #[test]
    fn test_imu_verification_low_confidence_match() {
        let fs = FusedSolver {
            solver: Arc::new(RwLock::new(None)),
            extractor: Arc::new(RwLock::new(None)),
            fast_extractor: Arc::new(RwLock::new(None)),
            imu: Arc::new(RwLock::new(None)),
            imu_type: Arc::new(RwLock::new(ImuType::Custom(Box::new(MockImuWithData)))),
            storage: None,
            latest_solve_position: Arc::new(RwLock::new(None)),
            last_solve_failed: Arc::new(RwLock::new(false)),
            latitude: Arc::new(RwLock::new(None)),
            longitude: Arc::new(RwLock::new(None)),
        };

        fs.set_observer_location(34.0, -118.0);
        fs.start_imu().unwrap();

        // Let the IMU thread accumulate some history
        std::thread::sleep(std::time::Duration::from_millis(3100)); // Must wait for IMU 3-second warm-up to finish!

        let t0 = std::time::SystemTime::now();

        // Anchor the IMU exactly at RA=100.0, Dec=50.0
        let anchor_sol = tetra3::solver::Solution {
            ra: Some(100.0),
            dec: Some(50.0),
            roll: Some(0.0),
            status: tetra3::solver::SolveStatus::MatchFound,
            ..Default::default()
        };
        fs.update_anchor_from_solution(&anchor_sol, t0);

        // Wait a tiny bit
        std::thread::sleep(std::time::Duration::from_millis(50));
        let t1 = std::time::SystemTime::now();

        // The IMU hasn't moved (always returning identity quaternion), so its estimate is still basically 100.0, 50.0.
        // Let's create a LowConfidenceMatch that is ~2 degrees away (within 5 degree cone).
        let candidate_verified = tetra3::solver::Solution {
            ra: Some(101.5), // ~1.5 degrees away in RA
            dec: Some(51.0), // ~1.0 degrees away in Dec
            roll: Some(0.0),
            status: tetra3::solver::SolveStatus::LowConfidenceMatch,
            ..Default::default()
        };

        let verified = fs.verify_solution_with_imu(&candidate_verified, t1, 5.0);
        assert_eq!(
            verified,
            Some(true),
            "Candidate should be verified as it is within 5 degrees"
        );

        // Let's test a candidate that is 10 degrees away
        let candidate_rejected = tetra3::solver::Solution {
            ra: Some(110.0),
            dec: Some(50.0),
            roll: Some(0.0),
            status: tetra3::solver::SolveStatus::LowConfidenceMatch,
            ..Default::default()
        };
        let rejected = fs.verify_solution_with_imu(&candidate_rejected, t1, 5.0);
        assert_eq!(
            rejected,
            Some(false),
            "Candidate should be rejected as it is >5 degrees away"
        );
    }

    #[test]
    fn test_imu_stationary_match_found_rejection() {
        let fs = FusedSolver {
            solver: Arc::new(RwLock::new(None)),
            extractor: Arc::new(RwLock::new(None)),
            fast_extractor: Arc::new(RwLock::new(None)),
            imu: Arc::new(RwLock::new(None)),
            imu_type: Arc::new(RwLock::new(ImuType::Custom(Box::new(MockImuWithData)))),
            storage: None,
            latest_solve_position: Arc::new(RwLock::new(None)),
            last_solve_failed: Arc::new(RwLock::new(false)),
            latitude: Arc::new(RwLock::new(None)),
            longitude: Arc::new(RwLock::new(None)),
        };

        fs.set_observer_location(34.0, -118.0);
        fs.start_imu().unwrap();

        // Let IMU initialize
        std::thread::sleep(std::time::Duration::from_millis(3100));

        let t0 = std::time::SystemTime::now();

        // Establish initial anchor at RA=100.0, Dec=50.0
        let anchor_sol = tetra3::solver::Solution {
            ra: Some(100.0),
            dec: Some(50.0),
            roll: Some(0.0),
            status: tetra3::solver::SolveStatus::MatchFound,
            ..Default::default()
        };
        fs.update_anchor_from_solution(&anchor_sol, t0);

        // IMU reports no motion since last anchor (angle_moved ≈ 0.0)
        let t1 = std::time::SystemTime::now();
        let angle_moved = fs.get_rotation_since_last_anchor(&t1).unwrap_or(0.0);
        assert!(
            angle_moved < STATIONARY_MOTION_THRESHOLD_DEG,
            "IMU should report < {}° motion since last anchor",
            STATIONARY_MOTION_THRESHOLD_DEG
        );

        // A false MatchFound solve (e.g. from tree leaves) jumping to RA=250.0, Dec=10.0 (155° away)
        let false_tree_solve = tetra3::solver::Solution {
            ra: Some(250.0),
            dec: Some(10.0),
            roll: Some(0.0),
            status: tetra3::solver::SolveStatus::MatchFound,
            ..Default::default()
        };

        // When stationary, verify should reject this 155° false solve.
        let max_allowed = if angle_moved < STATIONARY_MOTION_THRESHOLD_DEG {
            STATIONARY_CAMERA_DEVIATION_TOLERANCE_DEG
        } else {
            MOVING_CAMERA_DEVIATION_TOLERANCE_DEG
        };
        let is_valid = fs
            .verify_solution_with_imu(&false_tree_solve, t1, max_allowed)
            .unwrap_or(true);
        assert!(
            !is_valid,
            "False solve 155° away while stationary must be rejected"
        );

        // Valid solve 1.0° away while stationary should be accepted
        let valid_solve = tetra3::solver::Solution {
            ra: Some(100.5),
            dec: Some(50.5),
            roll: Some(0.0),
            status: tetra3::solver::SolveStatus::MatchFound,
            ..Default::default()
        };
        let is_valid_close = fs
            .verify_solution_with_imu(&valid_solve, t1, max_allowed)
            .unwrap_or(false);
        assert!(
            is_valid_close,
            "Valid solve 1° away while stationary must be accepted"
        );
    }
}
