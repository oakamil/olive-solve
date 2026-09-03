// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

#[cfg(any(target_os = "linux", target_os = "android"))]
mod hardware {
    use bno080::interface::i2c::I2cInterface;
    use bno080::wrapper::BNO080;
    use linux_embedded_hal::{Delay, I2cdev};
    use log::{info, warn};
    use nalgebra::{Quaternion, UnitQuaternion, Vector3};

    use std::time::{Duration, SystemTime};

    use crate::imu::{ImuDevice, SensorEvent};

    pub struct Bno085Device {
        imu: BNO080<I2cInterface<I2cdev>>,
        delay: Delay,
        report_interval_ms: u16,
        use_calibrated: bool,
        last_gyro_time: Option<SystemTime>,
        last_accel_time: Option<SystemTime>,
    }

    impl Bno085Device {
        pub fn new(
            report_interval_ms: u16,
            address: u8,
            use_calibrated: bool,
            i2c_bus: Option<u8>,
        ) -> Result<Self, String> {
            let bus = i2c_bus.unwrap_or(1);
            let bus_path = format!("/dev/i2c-{}", bus);
            info!(
                "Initializing BNO085 hardware over I2C ({}) at address 0x{:X}...",
                bus_path, address
            );
            let i2c = I2cdev::new(&bus_path).map_err(|e| format!("I2cdev::new failed: {:?}", e))?;
            let interface = I2cInterface::new(i2c, address);
            let mut imu = BNO080::new_with_interface(interface);
            let mut delay = Delay {};

            imu.init(&mut delay)
                .map_err(|e| format!("Failed to initialize BNO085 over I2C: {:?}", e))?;

            let mode_str = if use_calibrated {
                "Calibrated"
            } else {
                "Uncalibrated"
            };

            if use_calibrated {
                imu.enable_gyro_calibrated(report_interval_ms)
                    .map_err(|e| format!("Failed to enable Calibrated Gyroscope: {:?}", e))?;
            } else {
                imu.enable_gyro(report_interval_ms)
                    .map_err(|e| format!("Failed to enable Uncalibrated Gyroscope: {:?}", e))?;
            }

            std::thread::sleep(Duration::from_millis(50));

            imu.enable_accelerometer(report_interval_ms)
                .map_err(|e| format!("Failed to enable Accelerometer: {:?}", e))?;

            std::thread::sleep(Duration::from_millis(50));

            imu.enable_rotation_vector(report_interval_ms)
                .map_err(|e| format!("Failed to enable Rotation Vector: {:?}", e))?;

            info!(
                "Hardware initialized at {}ms using {} Gyroscope.",
                report_interval_ms, mode_str
            );

            Ok(Self {
                imu,
                delay,
                report_interval_ms,
                use_calibrated,
                last_gyro_time: None,
                last_accel_time: None,
            })
        }
    }

    impl ImuDevice for Bno085Device {
        fn init(&mut self) -> Result<(), String> {
            Ok(())
        }

        fn poll(&mut self) -> Result<Vec<SensorEvent>, String> {
            let _msg_count = self.imu.handle_all_messages(&mut self.delay, 1);

            let mut events = Vec::new();

            let (accel_len, accel_queue) = self.imu.accel_queue();
            if accel_len > 0 {
                let now = SystemTime::now();
                let fallback_dt = (self.report_interval_ms as f64) / 1000.0;
                for i in 0..accel_len {
                    let steps_backward = (accel_len - 1 - i) as u32;
                    let sample_time = now
                        .checked_sub(Duration::from_secs_f64(
                            fallback_dt * (steps_backward as f64),
                        ))
                        .unwrap_or(now);

                    let (_timestamp, accel_data) = accel_queue[i];
                    let ax = accel_data[0] as f64;
                    let ay = accel_data[1] as f64;
                    let az = accel_data[2] as f64;

                    let dt = if let Some(last) = self.last_accel_time {
                        sample_time
                            .duration_since(last)
                            .unwrap_or(Duration::from_secs_f64(fallback_dt))
                            .as_secs_f64()
                    } else {
                        fallback_dt
                    };

                    let safe_dt = if dt <= 0.0 { fallback_dt } else { dt };
                    self.last_accel_time = Some(sample_time);
                    events.push(SensorEvent {
                        accel: Some(Vector3::new(ax, ay, az)),
                        dt: Some(safe_dt),
                        ..Default::default()
                    });
                }
            }

            // Since we are polling over I2C without a hardware interrupt (HINT) pin, the BNO085's
            // internal timestamps reset on packet boundaries, making them unusable for absolute time.
            // Instead, we use "back-dating": we anchor the *last* sample in the queue to the host's
            // current wall-clock time (`now`), and step backwards by the requested hardware interval
            // for each preceding sample. This forces the boundary sample to absorb any I2C loop jitter
            // keeping the integration timeline aligned with real-world physical time.
            let (gyro_len, gyro_queue) = if self.use_calibrated {
                self.imu.calibrated_gyro_queue()
            } else {
                self.imu.gyro_queue()
            };

            if gyro_len > 0 {
                let now = SystemTime::now();
                let fallback_dt = (self.report_interval_ms as f64) / 1000.0;
                for i in 0..gyro_len {
                    let steps_backward = (gyro_len - 1 - i) as u32;
                    let sample_time = now
                        .checked_sub(Duration::from_secs_f64(
                            fallback_dt * (steps_backward as f64),
                        ))
                        .unwrap_or(now);

                    let (_timestamp, gyro_data) = gyro_queue[i];
                    let wx = gyro_data[0] as f64;
                    let wy = gyro_data[1] as f64;
                    let wz = gyro_data[2] as f64;

                    let dt = if let Some(last) = self.last_gyro_time {
                        sample_time
                            .duration_since(last)
                            .unwrap_or(Duration::from_secs_f64(fallback_dt))
                            .as_secs_f64()
                    } else {
                        fallback_dt
                    };

                    let safe_dt = if dt <= 0.0 { fallback_dt } else { dt };
                    self.last_gyro_time = Some(sample_time);
                    events.push(SensorEvent {
                        gyro: Some(Vector3::new(wx, wy, wz)),
                        dt: Some(safe_dt),
                        ..Default::default()
                    });
                }
            }

            if let Ok(q) = self.imu.rotation_quaternion() {
                // Only process the quaternion if it has been populated by the sensor (not all zeros)
                if q[0] != 0.0 || q[1] != 0.0 || q[2] != 0.0 || q[3] != 0.0 {
                    let quat = UnitQuaternion::new_normalize(Quaternion::new(
                        q[3] as f64,
                        q[0] as f64,
                        q[1] as f64,
                        q[2] as f64,
                    ));
                    if let Some(last) = events.last_mut() {
                        last.hardware_quaternion = Some(quat);
                    } else {
                        events.push(SensorEvent {
                            hardware_quaternion: Some(quat),
                            ..Default::default()
                        });
                    }
                }
            }

            Ok(events)
        }

        fn revive(&mut self) -> Result<(), String> {
            warn!("Sensor unresponsive. Sending hardware revive command...");
            if self.use_calibrated {
                self.imu
                    .enable_gyro_calibrated(self.report_interval_ms)
                    .map_err(|e| format!("Failed to revive: {:?}", e))?;
            } else {
                self.imu
                    .enable_gyro(self.report_interval_ms)
                    .map_err(|e| format!("Failed to revive: {:?}", e))?;
            }
            std::thread::sleep(Duration::from_millis(50));
            self.imu
                .enable_accelerometer(self.report_interval_ms)
                .map_err(|e| format!("Failed to revive accel: {:?}", e))?;
            std::thread::sleep(Duration::from_millis(50));
            self.imu
                .enable_rotation_vector(self.report_interval_ms)
                .map_err(|e| format!("Failed to revive rotation vector: {:?}", e))?;
            Ok(())
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub use hardware::*;

#[cfg(not(any(target_os = "linux", target_os = "android")))]
mod stub {
    use crate::imu::{ImuDevice, SensorEvent};
    use nalgebra::Vector3;

    /// Implementation of `ImuDevice` for the BNO085 IMU.
    pub struct Bno085Device;

    impl Bno085Device {
        /// Creates a new `Bno085Device`.
        pub fn new(
            _interval: u16,
            _address: u8,
            _calib: bool,
            _i2c_bus: Option<u8>,
        ) -> Result<Self, String> {
            Err("Hardware I2C is only supported on Linux/Android".into())
        }
    }

    impl ImuDevice for Bno085Device {
        fn init(&mut self) -> Result<(), String> {
            Err("Unsupported".into())
        }
        fn poll(&mut self) -> Result<Vec<SensorEvent>, String> {
            Err("Unsupported".into())
        }
        fn revive(&mut self) -> Result<(), String> {
            Err("Unsupported".into())
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub use stub::*;
