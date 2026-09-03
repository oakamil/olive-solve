// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use log::{debug, error, info, warn};
use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};
use std::sync::RwLock;

use crate::storage::PersistentStorage;

const CALIBRATION_KEY: &str = "imu_calibration";

// --- SVD CONFIGURATION CONSTANTS ---
const SVD_MATURITY_SIZE: usize = 15;
// The minimum 3D volume required to trust a calibration matrix.
// 0.15 (15%) requires an intentional physical roll of the telescope.
const MIN_CALIBRATION_CONFIDENCE: f64 = 0.15;
// If the hardware shifts by more than this amount, force a recalibration override.
const HARDWARE_ALTERATION_THRESHOLD_DEG: f64 = 10.0;
// Time to allow sensor to settle before considering it to be motionless.
const SETTLE_TIME_MS: u64 = 100;

// --- ANCHOR VERIFICATION CONSTANTS ---
/// The maximum angular movement (in degrees) for the telescope to be considered stationary.
pub const STATIONARY_MOTION_THRESHOLD_DEG: f64 = 1.0;

/// The maximum allowed plate-solve deviation (in degrees) when the telescope is stationary.
pub const STATIONARY_CAMERA_DEVIATION_TOLERANCE_DEG: f64 = 5.0;

/// The maximum allowed discrepancy (in degrees) between the plate-solve motion and the IMU motion when slewing.
pub const MOVING_CAMERA_DEVIATION_TOLERANCE_DEG: f64 = 20.0;

/// Represents the current physical motion state of the IMU.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MotionState {
    /// IMU is starting up or calibrating
    Initializing,
    /// IMU is detecting physical movement
    Moving,
    /// IMU is stationary
    Stable,
}

/// Raw 3-axis sensor data (e.g. from gyroscope or accelerometer).
#[derive(Clone, Copy, Debug)]
pub struct RawSensorData {
    /// X axis measurement
    pub x: f64,
    /// Y axis measurement
    pub y: f64,
    /// Z axis measurement
    pub z: f64,
}

/// Euler angle representation of the mount's orientation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MountCoordinates {
    /// Roll angle in degrees
    pub roll: f64,
    /// Pitch angle in degrees
    pub pitch: f64,
    /// Yaw angle in degrees
    pub yaw: f64,
}

/// Metrics tracking the health and alignment of the IMU-to-Camera coordinate transformation.
#[derive(Clone, Debug)]
pub struct TransformMetrics {
    /// The proportion of total variation that cannot be explained by the current mount transform
    pub transform_error_fraction: f64,
    /// The gyro axis most aligned with the camera's view axis
    pub camera_view_gyro_axis: String,
    /// The angular misalignment between the camera view axis and the closest gyro axis (degrees)
    pub camera_view_misalignment: f64,
    /// The gyro axis most aligned with the camera's up axis
    pub camera_up_gyro_axis: String,
    /// The angular misalignment between the camera up axis and the closest gyro axis (degrees)
    pub camera_up_misalignment: f64,
}

/// A single snapshot of IMU state.
#[derive(Clone, Copy, Debug)]
pub struct ImuUpdate {
    /// Time when the update was recorded
    pub timestamp: SystemTime,
    /// Raw gyroscope readings
    pub gyro: RawSensorData,
    /// Smoothed accelerometer readings
    pub gravity_vector: Option<RawSensorData>,
    /// Current IMU orientation quaternion
    pub quaternion: UnitQuaternion<f64>,
    /// Absolute hardware-fused quaternion (if supported)
    pub hardware_quaternion: Option<UnitQuaternion<f64>>,
    /// Absolute angular velocity magnitude
    pub angular_velocity: f64,
    /// Classified state of motion
    pub motion_state: MotionState,
}

/// Internal state tracking the alignment and calibration between the IMU and camera.
#[derive(Clone)]
pub struct AlignmentState {
    /// The known true camera pointing from the most recent successful plate solve.
    pub last_camera_position: Option<MountCoordinates>,
    /// The IMU's raw quaternion recorded at the exact time the plate solve image was taken.
    pub imu_anchor_state: Option<UnitQuaternion<f64>>,
    /// The dynamically calculated physical mounting rotation of the IMU relative to the camera.
    pub mount_q: UnitQuaternion<f64>,
    /// The calculated calibration health between the camera and IMU.
    pub transform_metrics: Option<TransformMetrics>,
    /// Store distinct rotational axes to continuously refine the 3D calibration
    pub calibration_axes: Vec<(Vector3<f64>, Vector3<f64>)>,
    /// Flag to track if the current mount_q was loaded from disk or previously locked
    pub loaded_from_disk: bool,
    /// Flag to allow one-time hardware alteration check at session startup
    pub startup_hardware_check_pending: bool,
    /// The highest confidence score achieved by the currently locked mount_q
    pub best_calibration_confidence: f64,
    /// Counter to throttle SD card writes
    pub calibration_updates_since_save: usize,
    /// Rolling history of expected vs true error for metric tracking
    pub error_history: Vec<f64>,
}

impl Default for AlignmentState {
    fn default() -> Self {
        Self {
            last_camera_position: None,
            imu_anchor_state: None,
            mount_q: UnitQuaternion::identity(),
            transform_metrics: None,
            calibration_axes: Vec::new(),
            loaded_from_disk: false,
            startup_hardware_check_pending: false,
            best_calibration_confidence: 0.0,
            calibration_updates_since_save: 0,
            error_history: Vec::new(),
        }
    }
}

/// Represents a discrete reading from an IMU sensor.
///
/// Because sensors might emit data for different streams at different intervals (or over different I2C packets),
/// any given event can contain a sparse assortment of gyro, accel, and hardware quaternion data.
#[derive(Debug, Clone, Default)]
pub struct SensorEvent {
    /// Angular velocity from the gyroscope (in rad/s)
    pub gyro: Option<Vector3<f64>>,
    /// Linear acceleration from the accelerometer (in m/s^2)
    pub accel: Option<Vector3<f64>>,
    /// Absolute hardware-fused orientation (if supported)
    pub hardware_quaternion: Option<UnitQuaternion<f64>>,
    /// The integration time delta (in seconds). Only required if `gyro` or `accel` is present.
    pub dt: Option<f64>,
}

/// Generic interface for communicating with physical IMU hardware.
pub trait ImuDevice: Send + 'static {
    /// Initializes and configures the hardware device.
    fn init(&mut self) -> Result<(), String>;
    /// Reads the latest sensor events from the device.
    fn poll(&mut self) -> Result<Vec<SensorEvent>, String>;
    /// Attempts to soft-reset or revive an unresponsive device.
    fn revive(&mut self) -> Result<(), String>;
    /// Reports if the IMU needs to seed the bias_offset value prior to calibration.
    /// Some sensors have large stationary biases greater than our movement threshold. For these
    /// sensors a stable set of 100 samples is collected to generate the initial bias offset.
    fn needs_seeding(&self) -> bool {
        false
    }
}

struct CalibrationState {
    pub total_samples: usize,
    pub bias_offset: Vector3<f64>,
    pub is_seeded: bool,
    pub seed_samples: usize,
    pub seed_sum: Vector3<f64>,
    pub last_seed_reading: Vector3<f64>,
}

impl Default for CalibrationState {
    fn default() -> Self {
        Self {
            total_samples: 0,
            bias_offset: Vector3::zeros(),
            is_seeded: false,
            seed_samples: 0,
            seed_sum: Vector3::zeros(),
            last_seed_reading: Vector3::zeros(),
        }
    }
}

/// High-level interface that manages hardware communication, calibration, and state tracking.
pub struct Imu {
    state: Arc<RwLock<Option<ImuUpdate>>>,
    alignment: Arc<RwLock<AlignmentState>>,
    history: Arc<Mutex<VecDeque<(SystemTime, UnitQuaternion<f64>, MotionState, Vector3<f64>)>>>,
    calibration: Arc<Mutex<CalibrationState>>,
    storage: Option<Arc<dyn PersistentStorage>>,
}

impl Imu {
    /// Starts the background thread to poll the given IMU hardware continuously.
    pub fn start<D: ImuDevice>(
        mut device: D,
        storage: Option<Arc<dyn PersistentStorage>>,
    ) -> Result<Self, String> {
        let state = Arc::new(RwLock::new(None));
        let state_clone = Arc::clone(&state);

        let mut initial_alignment = AlignmentState::default();

        // Load previous calibration (Strictly requires the 5-part format)
        if let Some(data) = storage.as_ref().and_then(|s| s.get(CALIBRATION_KEY)) {
            let parts: Vec<&str> = data.trim().split(',').collect();
            if parts.len() == 5 {
                if let (Ok(x), Ok(y), Ok(z), Ok(w), Ok(confidence)) = (
                    parts[0].parse::<f64>(),
                    parts[1].parse::<f64>(),
                    parts[2].parse::<f64>(),
                    parts[3].parse::<f64>(),
                    parts[4].parse::<f64>(),
                ) {
                    initial_alignment.mount_q =
                        UnitQuaternion::new_normalize(nalgebra::Quaternion::new(w, x, y, z));
                    initial_alignment.loaded_from_disk = true; // Protect this saved matrix
                    initial_alignment.startup_hardware_check_pending = true;
                    initial_alignment.best_calibration_confidence = confidence;
                    info!(
                        "Successfully loaded saved calibration (Confidence: {:.1}%): {:.3}, {:.3}, {:.3}, {:.3}",
                        initial_alignment.best_calibration_confidence * 100.0,
                        x,
                        y,
                        z,
                        w
                    );
                }
            }
        }

        device.init()?;

        let alignment = Arc::new(RwLock::new(initial_alignment));

        // 3 seconds of IMU history (capped at 300 items for 100Hz)
        let history = Arc::new(Mutex::new(VecDeque::with_capacity(300)));
        let history_clone = Arc::clone(&history);

        let calibration = Arc::new(Mutex::new(CalibrationState::default()));
        let calibration_clone = Arc::clone(&calibration);

        // --- ASYNCHRONOUS POLLING THREAD ---
        // We spawn a dedicated OS thread for hardware polling because I2C operations are
        // blocking and we strictly do not want to stall the async Tokio runtime.
        std::thread::spawn(move || {
            let mut prev_quat = UnitQuaternion::identity();
            let mut ema_accel: Option<Vector3<f64>> = None;
            let boot_time = SystemTime::now();
            let warm_up_duration = Duration::from_secs(3);
            let mut last_msg_time = SystemTime::now(); // Hardware watchdog tracker
            let mut last_motion_time = SystemTime::now();

            // Main hardware polling loop. Extracts messages from the I2C bus.
            loop {
                // If the primary Imu struct was dropped, our state_clone is the only remaining reference.
                // This gracefully kills the ghost thread and releases the I2C device.
                if Arc::strong_count(&state_clone) <= 1 {
                    log::info!("Imu struct dropped. Terminating hardware polling thread.");
                    break;
                }

                std::thread::sleep(Duration::from_millis(10));

                match device.poll() {
                    Ok(events) => {
                        if events.is_empty() {
                            // Hardware watchdog: The BNO085 occasionally locks up over I2C.
                            // If we haven't seen a packet in 2 seconds, we re-send the enable command.
                            if last_msg_time.elapsed().unwrap_or_default() > Duration::from_secs(2)
                            {
                                if let Err(e) = device.revive() {
                                    error!("Failed to revive device: {}", e);
                                }
                                last_msg_time = SystemTime::now();
                            }
                            continue;
                        }

                        let now = SystemTime::now();
                        last_msg_time = now; // Kick the watchdog

                        let total_batch_dt: f64 = events.iter().filter_map(|e| e.dt).sum();

                        let mut current_event_time = now
                            .checked_sub(Duration::from_secs_f64(total_batch_dt))
                            .unwrap_or(now);

                        let mut current_motion_state = MotionState::Stable;
                        let mut final_gyro_vec_rad = Vector3::zeros();
                        let mut final_gyro_mag = 0.0;
                        let mut gyro_updated = false;
                        let mut latest_hw_quat = None;

                        for event in events {
                            let opt_accel = event.accel;
                            let opt_gyro = event.gyro;

                            if let Some(hwq) = event.hardware_quaternion {
                                latest_hw_quat = Some(hwq);
                            }

                            if let Some(accel) = opt_accel {
                                if let Some(mut current) = ema_accel {
                                    let alpha = 0.1;
                                    current.x = (accel.x * alpha) + (current.x * (1.0 - alpha));
                                    current.y = (accel.y * alpha) + (current.y * (1.0 - alpha));
                                    current.z = (accel.z * alpha) + (current.z * (1.0 - alpha));
                                    ema_accel = Some(current);
                                } else {
                                    ema_accel = Some(accel);
                                }
                            }

                            if let Some(raw_gyro) = opt_gyro {
                                gyro_updated = true;
                                // Calculate bias and magnitude
                                let mut is_seeding_active = false;
                                let bias = {
                                    let mut cal = calibration_clone.lock().unwrap();

                                    // Ensure that a baseline reading is seeded if necessary
                                    if !cal.is_seeded {
                                        if device.needs_seeding() {
                                            let seeding_threshold = 0.05; // rad/s max deviation

                                            if cal.seed_samples == 0 {
                                                cal.last_seed_reading = raw_gyro;
                                                cal.seed_sum = raw_gyro;
                                                cal.seed_samples = 1;
                                            } else {
                                                let dev = (raw_gyro - cal.last_seed_reading).norm();
                                                if dev > seeding_threshold {
                                                    // Reset if deviation is too large (sensor is moving)
                                                    cal.seed_samples = 1;
                                                    cal.seed_sum = raw_gyro;
                                                    cal.last_seed_reading = raw_gyro;
                                                } else {
                                                    cal.seed_sum += raw_gyro;
                                                    cal.seed_samples += 1;
                                                    cal.last_seed_reading = raw_gyro;

                                                    if cal.seed_samples >= 100 {
                                                        cal.bias_offset = cal.seed_sum / 100.0;
                                                        cal.is_seeded = true;
                                                        info!(
                                                            "Seeded initial IMU bias: {:?}",
                                                            cal.bias_offset
                                                        );
                                                    }
                                                }
                                            }

                                            if !cal.is_seeded {
                                                is_seeding_active = true;
                                                raw_gyro // Output raw_gyro as temporary bias so gyro_vec_rad is 0
                                            } else {
                                                cal.bias_offset
                                            }
                                        } else {
                                            cal.is_seeded = true;
                                            cal.bias_offset
                                        }
                                    } else {
                                        cal.bias_offset
                                    }
                                };

                                let gyro_vec_rad = raw_gyro - bias;
                                let gyro_mag = gyro_vec_rad.norm();
                                final_gyro_vec_rad = gyro_vec_rad;
                                final_gyro_mag = gyro_mag;

                                // 0.001 rad/s is approx 0.057 deg/s, a better deadband for stationary drift
                                let hw_dt = event.dt.unwrap_or(0.0);
                                let delta_q = if gyro_mag > 0.001 {
                                    UnitQuaternion::new(gyro_vec_rad * hw_dt)
                                } else {
                                    UnitQuaternion::identity()
                                };

                                prev_quat *= delta_q;

                                let is_warming_up = boot_time.elapsed().unwrap_or_default()
                                    < warm_up_duration
                                    || is_seeding_active;

                                if gyro_mag > 0.05 {
                                    last_motion_time = now;
                                }

                                // We enforce a time-based settling period after any motion ends. This
                                // prevents heavy vibrations or structural settling in the telescope mount
                                // from polluting our zero-bias baseline.
                                current_motion_state = if is_warming_up {
                                    MotionState::Initializing
                                } else if now.duration_since(last_motion_time).unwrap_or_default()
                                    < Duration::from_millis(SETTLE_TIME_MS)
                                {
                                    MotionState::Moving
                                } else {
                                    MotionState::Stable
                                };

                                // Continuous zero-bias EMA tracking
                                if current_motion_state == MotionState::Stable {
                                    let mut cal = calibration_clone.lock().unwrap();
                                    cal.total_samples += 1;

                                    // Dynamic Alpha:
                                    // Phase 1 (1.0 / samples): Acts as a mathematically pure cumulative average to
                                    // rapidly lock in a highly accurate baseline over the first ~50 seconds (2500 samples).
                                    // Phase 2 (max 0.0004): Permanently transforms into a rolling Exponential Moving Average
                                    // that slowly tracks thermal drift without dragging the heavy anchor of historical data.
                                    let alpha = (1.0 / (cal.total_samples as f64)).max(0.0004);

                                    cal.bias_offset.x =
                                        (raw_gyro.x * alpha) + (cal.bias_offset.x * (1.0 - alpha));
                                    cal.bias_offset.y =
                                        (raw_gyro.y * alpha) + (cal.bias_offset.y * (1.0 - alpha));
                                    cal.bias_offset.z =
                                        (raw_gyro.z * alpha) + (cal.bias_offset.z * (1.0 - alpha));
                                }

                                current_event_time = current_event_time
                                    .checked_add(Duration::from_secs_f64(hw_dt))
                                    .unwrap_or(now);
                                let gyro_vec_deg = gyro_vec_rad * (180.0 / std::f64::consts::PI);

                                {
                                    let mut hist = history_clone.lock().unwrap();
                                    hist.push_back((
                                        current_event_time,
                                        prev_quat,
                                        current_motion_state,
                                        gyro_vec_deg,
                                    ));
                                    if hist.len() > 500 {
                                        hist.pop_front();
                                    }
                                }
                            }
                        }

                        if gyro_updated {
                            let update = ImuUpdate {
                                timestamp: now,
                                gyro: RawSensorData {
                                    x: final_gyro_vec_rad.x,
                                    y: final_gyro_vec_rad.y,
                                    z: final_gyro_vec_rad.z,
                                },
                                gravity_vector: ema_accel.map(|a| RawSensorData {
                                    x: a.x,
                                    y: a.y,
                                    z: a.z,
                                }),
                                quaternion: prev_quat,
                                hardware_quaternion: latest_hw_quat,
                                angular_velocity: final_gyro_mag,
                                motion_state: current_motion_state,
                            };

                            // Drop any stale updates and push the newest one
                            *state_clone.write().unwrap() = Some(update);
                        }
                    }
                    Err(e) => {
                        error!("Device poll error: {}", e);
                    }
                }
            }
        });

        Ok(Self {
            state,
            alignment,
            history: Arc::clone(&history),
            calibration: Arc::clone(&calibration),
            storage,
        })
    }

    /// Returns the number of stationary samples collected for bias calibration.
    pub fn get_bias_samples(&self) -> usize {
        let cal = self.calibration.lock().unwrap();
        cal.total_samples
    }

    /// Returns the current hardware gyroscope bias vector.
    pub fn get_bias(&self) -> Vector3<f64> {
        let cal = self.calibration.lock().unwrap();
        cal.bias_offset
    }

    /// Resets the internal bias calibration, restarting the sampling process.
    pub fn reset_bias_calibration(&self) {
        let mut cal = self.calibration.lock().unwrap();
        cal.total_samples = 0;
        cal.bias_offset = Vector3::zeros();
        cal.is_seeded = false;
        cal.seed_samples = 0;
        cal.seed_sum = Vector3::zeros();
        cal.last_seed_reading = Vector3::zeros();
        info!("Reset IMU bias calibration baseline.");
    }

    /// Returns the angular movement (in degrees) measured by the IMU since the last plate-solve anchor was established.
    pub fn get_rotation_since_last_anchor(&self, timestamp: &SystemTime) -> Option<f64> {
        let align = self.alignment.read().unwrap();
        if let Some(old_quat) = align.imu_anchor_state {
            if let Some((hist_q, _)) = self.get_historical_quat(timestamp) {
                return Some((old_quat.conjugate() * hist_q).angle().to_degrees());
            }
        }
        None
    }

    // Helper to identify the closest primary IMU axis to a given camera vector
    fn get_closest_imu_axis(v: &Vector3<f64>) -> (String, f64) {
        let mut max_val = 0.0;
        let mut max_idx = 0;
        let mut sign = "+";

        for i in 0..3 {
            if v[i].abs() > max_val {
                max_val = v[i].abs();
                max_idx = i;
                sign = if v[i] >= 0.0 { "+" } else { "-" };
            }
        }

        let axis_name = match max_idx {
            0 => "X",
            1 => "Y",
            _ => "Z",
        };

        let mut pure_axis = Vector3::zeros();
        pure_axis[max_idx] = if sign == "+" { 1.0 } else { -1.0 };

        let misalignment = v.angle(&pure_axis).to_degrees();
        (format!("{}{}", sign, axis_name), misalignment)
    }

    // For real-time UI queries. Finds the literal closest timestamp without bracketing logic.
    fn get_historical_quat(
        &self,
        target_time: &SystemTime,
    ) -> Option<(UnitQuaternion<f64>, MotionState)> {
        let hist = self.history.lock().unwrap();
        if hist.is_empty() {
            return None;
        }

        let oldest_time = hist.front().unwrap().0;

        if *target_time < oldest_time {
            let diff = oldest_time.duration_since(*target_time).unwrap_or_default();
            if diff > Duration::from_secs(2) {
                debug!(
                    "Plate solve is {}s older than our oldest history frame. Data expired.",
                    diff.as_secs()
                );
                return None;
            }
        }

        let mut closest_quat = hist[0].1;
        let mut closest_motion = hist[0].2;
        let mut min_diff = Duration::from_secs(u64::MAX);

        for (time, quat, motion, _) in hist.iter() {
            let diff = if time > target_time {
                time.duration_since(*target_time).unwrap_or_default()
            } else {
                target_time.duration_since(*time).unwrap_or_default()
            };

            if diff < min_diff {
                min_diff = diff;
                closest_quat = *quat;
                closest_motion = *motion;
            }
        }

        if min_diff > Duration::from_secs(5) {
            warn!(
                "Nearest IMU frame is {}ms off. Rejecting out-of-sync timestamp.",
                min_diff.as_millis()
            );
            return None;
        }

        Some((closest_quat, closest_motion))
    }

    fn mount_to_quat(coord: &MountCoordinates) -> UnitQuaternion<f64> {
        UnitQuaternion::from_euler_angles(
            coord.roll.to_radians(),
            coord.pitch.to_radians(),
            coord.yaw.to_radians(),
        )
    }

    fn quat_to_mount(quat: &UnitQuaternion<f64>) -> MountCoordinates {
        let (roll, pitch, yaw) = quat.euler_angles();
        MountCoordinates {
            roll: roll.to_degrees().rem_euclid(360.0),
            pitch: pitch.to_degrees(),
            yaw: yaw.to_degrees().rem_euclid(360.0),
        }
    }

    // Helper to spin up a non-blocking save
    fn save_calibration_to_disk(&self, mount_q: UnitQuaternion<f64>, confidence: f64) {
        if let Some(storage) = self.storage.clone() {
            let data = format!(
                "{},{},{},{},{}",
                mount_q[0], mount_q[1], mount_q[2], mount_q[3], confidence
            );
            std::thread::spawn(move || {
                storage.set(CALIBRATION_KEY, &data);
                debug!("Successfully wrote calibration back to storage");
            });
        }
    }

    /// Updates the IMU anchor based on a true camera position from a successful plate solve.
    /// Returns early if no valid quaternion was found in history near `timestamp`.
    pub fn update_anchor(&self, camera_pointing: &MountCoordinates, timestamp: &SystemTime) {
        let imu_state = *self.state.read().unwrap();

        if imu_state.is_some() {
            // Find the exact historical quaternion that matches the image timestamp
            let historical_imu_q = self.get_historical_quat(timestamp);

            if let Some((hist_q, motion_state)) = historical_imu_q {
                if motion_state != MotionState::Stable {
                    debug!(
                        "Image was captured during movement. Rejecting anchor to prevent timestamp and rolling shutter artifacts."
                    );
                    return;
                }

                let mut align = self.alignment.write().unwrap();
                let new_true_q = Self::mount_to_quat(camera_pointing);

                if let (Some(old_mount), Some(old_quat)) =
                    (align.last_camera_position, align.imu_anchor_state)
                {
                    let old_true_q = Self::mount_to_quat(&old_mount);

                    // Continuous SVD calibration (Wahba's Problem)
                    let q_true_delta = old_true_q.conjugate() * new_true_q;
                    let q_imu_delta = old_quat.conjugate() * hist_q;

                    let angle_cam = q_true_delta.angle().to_degrees();
                    let angle_imu = q_imu_delta.angle().to_degrees();

                    // SVD Calibration Eligibility:
                    // Both sensors must detect significant physical motion and agree on the slew magnitude.
                    let svd_eligible = angle_cam >= STATIONARY_MOTION_THRESHOLD_DEG
                        && angle_imu >= STATIONARY_MOTION_THRESHOLD_DEG
                        && (angle_cam - angle_imu).abs() <= MOVING_CAMERA_DEVIATION_TOLERANCE_DEG;

                    if svd_eligible {
                        if let (Some(axis_true), Some(axis_imu)) =
                            (q_true_delta.axis(), q_imu_delta.axis())
                        {
                            let t_vec = axis_true.into_inner();
                            let i_vec = axis_imu.into_inner();

                            let mut replaced = false;
                            for (existing_t, existing_i) in align.calibration_axes.iter_mut() {
                                // If the new physical axis is highly parallel to an existing one (cos(11 deg) ≈ 0.98)
                                if existing_t.dot(&t_vec).abs() > 0.98 {
                                    // Overwrite it! This keeps the calibration mathematically fresh
                                    // without destroying our hard-earned 3D spatial diversity!
                                    *existing_t = t_vec;
                                    *existing_i = i_vec;
                                    replaced = true;
                                    break;
                                }
                            }

                            if !replaced {
                                align.calibration_axes.push((t_vec, i_vec));
                                if align.calibration_axes.len() > 100 {
                                    align.calibration_axes.remove(0);
                                }
                            }

                            let mut is_rank_sufficient = false;
                            for i in 0..align.calibration_axes.len() {
                                for j in (i + 1)..align.calibration_axes.len() {
                                    if align.calibration_axes[i]
                                        .0
                                        .dot(&align.calibration_axes[j].0)
                                        .abs()
                                        < 0.95
                                    {
                                        is_rank_sufficient = true;
                                        break;
                                    }
                                }
                                if is_rank_sufficient {
                                    break;
                                }
                            }

                            if is_rank_sufficient {
                                let mut b = Matrix3::zeros();

                                // SVD Matrix Construction
                                for (t, i) in &align.calibration_axes {
                                    b += t * i.transpose();
                                }

                                let svd = b.svd(true, true);
                                if let (Some(u), Some(v_t)) = (svd.u, svd.v_t) {
                                    // --- CALIBRATION CONFIDENCE SCORING ---
                                    // Extract singular values (sigma_1 is max, sigma_3 is min).
                                    // The ratio of min to max defines the true 3D geometric volume of the calibration.
                                    let sigma_1 = svd.singular_values[0];
                                    let sigma_3 = svd.singular_values[2];

                                    let new_pool_confidence = if sigma_1 > 0.0 {
                                        sigma_3 / sigma_1
                                    } else {
                                        0.0
                                    };

                                    let det = (u * v_t).determinant();
                                    let mut d = Matrix3::identity();
                                    if det < 0.0 {
                                        d[(2, 2)] = -1.0;
                                    }

                                    let r_mount = u * d * v_t;

                                    if r_mount.iter().all(|val| val.is_finite()) {
                                        let calculated_q = UnitQuaternion::from_rotation_matrix(
                                            &Rotation3::from_matrix_unchecked(r_mount),
                                        );

                                        let is_mature =
                                            align.calibration_axes.len() >= SVD_MATURITY_SIZE;
                                        let hardware_shift_deg = (align.mount_q.inverse()
                                            * calculated_q)
                                            .angle()
                                            .to_degrees();

                                        if !align.loaded_from_disk {
                                            // Bootstrapping phase. Update fluidly to get the UI tracking immediately.
                                            align.mount_q = calculated_q;
                                            align.best_calibration_confidence = new_pool_confidence;

                                            // Save to disk immediately so progress isn't lost if the app closes
                                            self.save_calibration_to_disk(
                                                align.mount_q,
                                                align.best_calibration_confidence,
                                            );

                                            // Only lock to High Water Mark mode once we have enough diverse data points
                                            if is_mature {
                                                align.loaded_from_disk = true; // Upgrade status to protected
                                                info!(
                                                    "Bootstrapping complete. Calibration Locked! Confidence: {:.1}%",
                                                    align.best_calibration_confidence * 100.0
                                                );
                                            }
                                        } else if align.startup_hardware_check_pending {
                                            // Startup verification: detect if hardware was physically remounted between sessions
                                            if is_mature
                                                && new_pool_confidence >= MIN_CALIBRATION_CONFIDENCE
                                                && hardware_shift_deg
                                                    > HARDWARE_ALTERATION_THRESHOLD_DEG
                                            {
                                                warn!(
                                                    "Hardware alteration detected at startup! New matrix differs by {:.2}°. Updating calibration.",
                                                    hardware_shift_deg
                                                );
                                                align.mount_q = calculated_q;
                                                align.best_calibration_confidence =
                                                    new_pool_confidence;
                                                self.save_calibration_to_disk(
                                                    align.mount_q,
                                                    align.best_calibration_confidence,
                                                );
                                            }
                                            if is_mature {
                                                align.startup_hardware_check_pending = false;
                                            }
                                        } else if new_pool_confidence
                                            > align.best_calibration_confidence
                                            && hardware_shift_deg <= 10.0
                                        {
                                            // Incremental upgrade during runtime tracking
                                            info!(
                                                "Upgrading calibration matrix! Confidence increased from {:.1}% to {:.1}% (Shift: {:.2}°)",
                                                align.best_calibration_confidence * 100.0,
                                                new_pool_confidence * 100.0,
                                                hardware_shift_deg
                                            );
                                            align.mount_q = calculated_q;
                                            align.best_calibration_confidence = new_pool_confidence;

                                            align.calibration_updates_since_save += 1;
                                            if align.calibration_updates_since_save % 5 == 0 {
                                                self.save_calibration_to_disk(
                                                    align.mount_q,
                                                    align.best_calibration_confidence,
                                                );
                                            }
                                        } else {
                                            // Coasting Phase. The active hardware is identical, and the new pool is flatter than our historical best.
                                            // Ignore the SVD calculation to protect the High Water Mark matrix.
                                        }
                                    } else {
                                        warn!("SVD generated NaNs. Keeping previous safe mount_q.");
                                    }
                                }
                            } else {
                                debug!(
                                    "Calibration pool lacks distinct axes. Need movement on a different plane to run SVD."
                                );
                            }
                        }
                    } else {
                        debug!(
                            "Movement (cam: {:.2}°, imu: {:.2}°) not eligible for SVD calibration.",
                            angle_cam, angle_imu
                        );
                    }

                    // Recalculate error metric post-SVD refinement for reporting
                    let imu_local_delta = old_quat.conjugate() * hist_q;
                    let cam_local_delta =
                        align.mount_q * imu_local_delta * align.mount_q.conjugate();
                    let final_expected = old_true_q * cam_local_delta;

                    let final_error_quat = final_expected.inverse() * new_true_q;
                    let final_error_angle = final_error_quat.angle().to_degrees();

                    if angle_cam > 5.0 {
                        align.error_history.push(final_error_angle);
                        if align.error_history.len() > 20 {
                            align.error_history.remove(0);
                        }
                        let avg_error: f64 = align.error_history.iter().sum::<f64>()
                            / (align.error_history.len() as f64);

                        // --- ALT/AZ ERROR COMPONENT LOGGING ---
                        let expected_coords = Self::quat_to_mount(&final_expected);
                        let true_coords = Self::quat_to_mount(&new_true_q);

                        let alt_error = true_coords.pitch - expected_coords.pitch;

                        let mut az_error = true_coords.yaw - expected_coords.yaw;
                        if az_error > 180.0 {
                            az_error -= 360.0;
                        }
                        if az_error < -180.0 {
                            az_error += 360.0;
                        }

                        let mut roll_error = true_coords.roll - expected_coords.roll;
                        if roll_error > 180.0 {
                            roll_error -= 360.0;
                        }
                        if roll_error < -180.0 {
                            roll_error += 360.0;
                        }

                        info!(
                            "Expected vs True error: {:.3}° (Rolling Avg: {:.3}°) | Alt Err: {:.3}°, Az Err: {:.3}°, Roll Err: {:.3}° | Confidence: {:.1}%",
                            final_error_angle,
                            avg_error,
                            alt_error,
                            az_error,
                            roll_error,
                            align.best_calibration_confidence * 100.0
                        );
                    }

                    // Assume standard optical camera axes: View = +Z, Up = +Y.
                    // Rotate the camera axes into the IMU's reference frame using our calibration matrix.
                    let cam_view_in_imu = align.mount_q.conjugate() * Vector3::new(0.0, 0.0, 1.0);
                    let cam_up_in_imu = align.mount_q.conjugate() * Vector3::new(0.0, 1.0, 0.0);

                    let (view_axis, view_misalign) = Self::get_closest_imu_axis(&cam_view_in_imu);
                    let (up_axis, up_misalign) = Self::get_closest_imu_axis(&cam_up_in_imu);

                    align.transform_metrics = Some(TransformMetrics {
                        transform_error_fraction: final_error_angle / angle_cam.max(0.001),
                        camera_view_gyro_axis: view_axis,
                        camera_view_misalignment: view_misalign,
                        camera_up_gyro_axis: up_axis,
                        camera_up_misalignment: up_misalign,
                    });

                    // Pointing Anchor Update
                    let is_valid_anchor_update = if angle_imu < STATIONARY_MOTION_THRESHOLD_DEG {
                        angle_cam <= STATIONARY_CAMERA_DEVIATION_TOLERANCE_DEG
                    } else {
                        (angle_cam - angle_imu).abs() <= MOVING_CAMERA_DEVIATION_TOLERANCE_DEG
                    };

                    if is_valid_anchor_update {
                        align.last_camera_position = Some(*camera_pointing);
                        align.imu_anchor_state = Some(hist_q);
                    } else {
                        debug!(
                            "Rejecting anchor update: discrepancy too large (cam: {:.2}°, imu: {:.2}°)",
                            angle_cam, angle_imu
                        );
                    }
                } else {
                    info!("Initial plate-solve anchor locked in.");

                    let cam_view_in_imu = align.mount_q.conjugate() * Vector3::new(0.0, 0.0, 1.0);
                    let cam_up_in_imu = align.mount_q.conjugate() * Vector3::new(0.0, 1.0, 0.0);

                    let (view_axis, view_misalign) = Self::get_closest_imu_axis(&cam_view_in_imu);
                    let (up_axis, up_misalign) = Self::get_closest_imu_axis(&cam_up_in_imu);

                    align.transform_metrics = Some(TransformMetrics {
                        transform_error_fraction: 0.0,
                        camera_view_gyro_axis: view_axis,
                        camera_view_misalignment: view_misalign,
                        camera_up_gyro_axis: up_axis,
                        camera_up_misalignment: up_misalign,
                    });

                    align.last_camera_position = Some(*camera_pointing);
                    align.imu_anchor_state = Some(hist_q);
                }
            }
        }
    }

    /// Updates the positional IMU anchor from a true camera position without adding
    /// the movement vector to the SVD calibration pool. This safely cures gyro drift
    /// using low-confidence plate solves without risking permanent geometric corruption.
    pub fn update_pointing_anchor_only(
        &self,
        camera_pointing: &MountCoordinates,
        timestamp: &SystemTime,
    ) {
        let imu_state = *self.state.read().unwrap();

        if imu_state.is_some() {
            let historical_imu_q = self.get_historical_quat(timestamp);

            if let Some((hist_q, motion_state)) = historical_imu_q {
                if motion_state != MotionState::Stable {
                    debug!(
                        "Image was captured during movement. Rejecting pointing anchor update to prevent timestamp and rolling shutter artifacts."
                    );
                    return;
                }

                let mut align = self.alignment.write().unwrap();
                align.last_camera_position = Some(*camera_pointing);
                align.imu_anchor_state = Some(hist_q);

                debug!(
                    "IMU pointing anchor successfully updated from low-confidence solve. SVD calibration bypassed."
                );
            }
        }
    }

    // Clears the active session anchors and telemetry, but strictly preserves
    // the SVD calibration pool, mount_q, and disk file to support file-less recalibration and
    // uninterrupted EQ tracking.
    /// Resets the IMU anchor state, clearing the `last_camera_position`.
    pub fn reset_anchors(&self) {
        debug!("reset called. Clearing anchors but preserving calibration matrix.");
        let mut align = self.alignment.write().unwrap();
        align.last_camera_position = None;
        align.imu_anchor_state = None;
        align.transform_metrics = None;
        align.error_history.clear();
    }

    // Clears the entire calibration matrix, history, and deletes the persistent file.
    /// Clears the 3D calibration between the IMU and camera, resetting `mount_q` to identity.
    pub fn clear_calibration(&self) {
        let mut align = self.alignment.write().unwrap();
        align.mount_q = UnitQuaternion::identity();
        align.calibration_axes.clear();
        align.loaded_from_disk = false;
        align.best_calibration_confidence = 0.0;

        if let Some(storage) = self.storage.clone() {
            std::thread::spawn(move || {
                storage.remove(CALIBRATION_KEY);
                info!("Deleted IMU calibration file from persistent storage.");
            });
        }
        info!("IMU calibration matrix and history have been cleared.");
    }

    // IMU-derived estimate of camera pointing as of the given time.
    // The boolean in the Result tuple is `true` if the returned position is an updated estimate
    // using IMU data, and `false` if it is falling back to the static anchor itself.
    /// Computes the camera's estimated pointing by advancing the last known
    /// camera position using the IMU's rotational delta since the anchor time.
    pub fn get_estimated_pointing(
        &self,
        timestamp: &SystemTime,
    ) -> Result<(MountCoordinates, bool), &'static str> {
        let align = self.alignment.read().unwrap().clone();

        if let (Some(anchor_horiz), Some(anchor_quat)) =
            (align.last_camera_position, align.imu_anchor_state)
        {
            if !align.loaded_from_disk && align.calibration_axes.len() < 3 {
                return Ok((anchor_horiz, false));
            }

            // 1. Fetch the raw IMU state for the requested time
            let target_q = self
                .get_historical_quat(timestamp)
                .map(|h| h.0)
                .unwrap_or_else(|| {
                    debug!("get_estimated falling back to real-time quaternion");
                    self.state.read().unwrap().unwrap().quaternion
                });

            // 2. Calculate the raw physical movement of the IMU chip itself.
            // This rotational delta is strictly in the IMU's local reference frame,
            // meaning it includes any arbitrary physical mounting offsets (roll, pitch, yaw)
            // between how the IMU is mounted versus how the camera sensor is oriented.
            let imu_local_delta = anchor_quat.conjugate() * target_q;

            // Reject tiny movements to avoid reporting stationary drift
            if imu_local_delta.angle().to_degrees() < 0.05 {
                return Ok((anchor_horiz, false));
            }

            // 3. Coordinate Transformation (Similarity Transform / Change of Basis).
            // We must convert the IMU's local movement into the camera's optical reference frame.
            // The mathematically universal transform is: Camera_Delta = Mount * IMU_Delta * Mount_Inverse.
            // This isolates the actual telescope movement by "untwisting" whatever arbitrary physical
            // mounting offset exists between the two sensors, preventing movements on one axis
            // from bleeding into another (e.g., preventing Azimuth pans from dipping into Altitude).
            let cam_local_delta = align.mount_q * imu_local_delta * align.mount_q.conjugate();

            // 4. Convert our known optical anchor (from the last successful plate solve) into a quaternion
            let anchor_true_q = Self::mount_to_quat(&anchor_horiz);

            // 5. Apply the correctly transformed movement delta directly to the true sky anchor
            let est_q = anchor_true_q * cam_local_delta;

            Ok((Self::quat_to_mount(&est_q), true))
        } else {
            Err("No plate solve anchor established yet.")
        }
    }

    /// Returns the current classification of the IMU's physical motion.
    pub fn get_motion_state(&self) -> MotionState {
        self.state
            .read()
            .unwrap()
            .as_ref()
            .map(|s| s.motion_state)
            .unwrap_or(MotionState::Initializing)
    }

    /// Returns the active metrics describing the health of the hardware calibration.
    pub fn get_calibration_metrics(&self) -> Option<TransformMetrics> {
        self.alignment.read().unwrap().transform_metrics.clone()
    }

    /// Returns `true` if a full 3D calibration has been established.
    pub fn is_calibrated(&self) -> bool {
        let align = self.alignment.read().unwrap();
        align.loaded_from_disk || align.calibration_axes.len() >= 3
    }

    /// Retrieves the latest snapshot of the IMU's orientation and motion state.
    pub fn get_latest_state(&self) -> Option<ImuUpdate> {
        *self.state.read().unwrap()
    }
}
