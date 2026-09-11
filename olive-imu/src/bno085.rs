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
        last_poll_time: Option<SystemTime>,
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
                last_poll_time: None,
            })
        }
    }

    fn pair_bno085_samples(
        accel_queue: &[(u32, [f32; 3])],
        gyro_queue: &[(u32, [f32; 3])],
        report_interval_ms: u16,
        now: SystemTime,
        last_poll_time: Option<SystemTime>,
    ) -> (Vec<SensorEvent>, SystemTime) {
        let mut events = Vec::new();
        let max_len = std::cmp::max(accel_queue.len(), gyro_queue.len());
        if max_len == 0 {
            return (events, last_poll_time.unwrap_or(now));
        }

        let fallback_dt = (report_interval_ms as f64) / 1000.0;

        // Determine the expected length primarily from gyro, fallback to accel
        let expected_len = if !gyro_queue.is_empty() {
            gyro_queue.len()
        } else {
            accel_queue.len()
        } as f64;
        let expected_total_time = expected_len * fallback_dt;

        let mut total_dt = if let Some(last) = last_poll_time {
            now.duration_since(last)
                .unwrap_or(Duration::from_secs_f64(expected_total_time))
                .as_secs_f64()
        } else {
            expected_total_time
        };

        // Clamp to [0.5 * expected, 2.5 * expected] to absorb jitter but reject stalls/sleeps
        if total_dt < 0.5 * expected_total_time {
            total_dt = 0.5 * expected_total_time;
        } else if total_dt > 2.5 * expected_total_time {
            total_dt = 2.5 * expected_total_time;
        }

        let gyro_dt = if !gyro_queue.is_empty() {
            total_dt / (gyro_queue.len() as f64)
        } else {
            0.0 // dt is exclusively assigned to gyro frames
        };

        for i in 0..max_len {
            let accel_vec = accel_queue
                .get(i)
                .map(|(_, data)| Vector3::new(data[0] as f64, data[1] as f64, data[2] as f64));

            let gyro_vec = gyro_queue
                .get(i)
                .map(|(_, data)| Vector3::new(data[0] as f64, data[1] as f64, data[2] as f64));

            let dt = if gyro_vec.is_some() {
                Some(gyro_dt)
            } else {
                None
            };

            events.push(SensorEvent {
                accel: accel_vec,
                gyro: gyro_vec,
                dt,
                ..Default::default()
            });
        }

        let next_time = if !gyro_queue.is_empty() {
            now
        } else {
            last_poll_time.unwrap_or(now)
        };

        (events, next_time)
    }

    impl ImuDevice for Bno085Device {
        fn init(&mut self) -> Result<(), String> {
            Ok(())
        }

        fn poll(&mut self) -> Result<Vec<SensorEvent>, String> {
            let _msg_count = self.imu.handle_all_messages(&mut self.delay, 1);

            let (accel_len, accel_queue) = self.imu.accel_queue();
            let (gyro_len, gyro_queue) = if self.use_calibrated {
                self.imu.calibrated_gyro_queue()
            } else {
                self.imu.gyro_queue()
            };

            let now = SystemTime::now();
            let (mut events, new_time) = pair_bno085_samples(
                &accel_queue[0..accel_len],
                &gyro_queue[0..gyro_len],
                self.report_interval_ms,
                now,
                self.last_poll_time,
            );
            self.last_poll_time = Some(new_time);

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

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::time::{Duration, SystemTime};

        #[test]
        fn test_pair_bno085_samples_empty_poll() {
            let now = SystemTime::now();
            let last = now.checked_sub(Duration::from_millis(20)).unwrap();

            let (events, next_time) = pair_bno085_samples(&[], &[], 10, now, Some(last));

            assert_eq!(events.len(), 0);
            assert_eq!(next_time, last); // Should NOT advance to now
        }

        #[test]
        fn test_pair_bno085_samples_accel_only() {
            let now = SystemTime::now();
            let last = now.checked_sub(Duration::from_millis(20)).unwrap();

            let accel = vec![(0, [1.0, 2.0, 3.0])];
            let (events, next_time) = pair_bno085_samples(&accel, &[], 10, now, Some(last));

            assert_eq!(events.len(), 1);
            assert_eq!(events[0].dt, None); // dt should be none for accel-only
            assert_eq!(next_time, last); // Should NOT advance to now
        }

        #[test]
        fn test_pair_bno085_samples_symmetric() {
            let now = SystemTime::now();
            let last = now.checked_sub(Duration::from_millis(20)).unwrap();

            let accel = vec![(0, [1.0, 2.0, 3.0]), (1, [4.0, 5.0, 6.0])];
            let gyro = vec![(0, [0.1, 0.2, 0.3]), (1, [0.4, 0.5, 0.6])];

            let (events, next_time) = pair_bno085_samples(&accel, &gyro, 10, now, Some(last));

            assert_eq!(events.len(), 2);
            assert_eq!(events[0].dt, Some(0.01));
            assert_eq!(events[1].dt, Some(0.01));
            assert_eq!(next_time, now);
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
