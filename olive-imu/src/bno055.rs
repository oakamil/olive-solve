// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

#[cfg(any(target_os = "linux", target_os = "android"))]
mod hardware {
    use bno055::{BNO055OperationMode, Bno055};
    use linux_embedded_hal::{Delay, I2cdev};
    use log::{info, warn};
    use nalgebra::{Quaternion, UnitQuaternion, Vector3};

    use std::time::{Duration, SystemTime};

    use crate::imu::{ImuDevice, SensorEvent};

    pub struct Bno055Device {
        imu: Bno055<I2cdev>,
        delay: Delay,
        report_interval_ms: u16,
        last_system_time: Option<SystemTime>,
    }

    impl Bno055Device {
        pub fn new(report_interval_ms: u16, address: u8) -> Result<Self, String> {
            info!(
                "Initializing BNO055 hardware over I2C at address 0x{:X}...",
                address
            );
            // i2c-1 is typical on Raspberry Pi I2C buses
            let i2c =
                I2cdev::new("/dev/i2c-1").map_err(|e| format!("I2cdev::new failed: {:?}", e))?;

            // The bno055 crate defaults to 0x28. If the alternate address is requested,
            // we configure the driver to use it.
            let mut imu = bno055::Bno055::new(i2c);
            if address == 0x29 {
                imu = imu.with_alternative_address();
            } else if address != 0x28 {
                return Err(format!("Unsupported BNO055 I2C address: 0x{:X}", address));
            }

            let mut delay = Delay {};

            imu.init(&mut delay)
                .map_err(|e| format!("Failed to initialize BNO055 over I2C: {:?}", e))?;

            // Set mode to NDOF
            imu.set_mode(BNO055OperationMode::NDOF, &mut delay)
                .map_err(|e| format!("Failed to set BNO055 mode to NDOF: {:?}", e))?;

            info!(
                "Hardware initialized at {}ms using NDOF mode.",
                report_interval_ms
            );

            Ok(Self {
                imu,
                delay,
                report_interval_ms,
                last_system_time: None,
            })
        }
    }

    impl ImuDevice for Bno055Device {
        fn init(&mut self) -> Result<(), String> {
            Ok(())
        }

        fn poll(&mut self) -> Result<Vec<SensorEvent>, String> {
            let mut readings = Vec::new();

            // The BNO055 does not support an internal hardware queue over I2C.
            // We simply poll the latest gyroscope reading from the data registers.
            let gyro_data = match self.imu.gyro_data() {
                Ok(data) => data,
                Err(_) => return Ok(readings), // If read fails, return empty to let caller handle watchdog
            };

            let accel_data = match self.imu.accel_data() {
                Ok(data) => data,
                Err(_) => return Ok(readings),
            };

            let now = SystemTime::now();
            let fallback_dt = (self.report_interval_ms as f64) / 1000.0;

            let dt = if let Some(last) = self.last_system_time {
                now.duration_since(last)
                    .unwrap_or(Duration::from_secs_f64(fallback_dt))
                    .as_secs_f64()
            } else {
                fallback_dt
            };

            let safe_dt = if dt <= 0.0 { fallback_dt } else { dt };
            self.last_system_time = Some(now);

            // The BNO055 defaults to Degrees Per Second (DPS).
            // We must convert this to Radians Per Second (rad/s) to match the system expectation.
            let to_rad = std::f64::consts::PI / 180.0;
            let wx = gyro_data.x as f64 * to_rad;
            let wy = gyro_data.y as f64 * to_rad;
            let wz = gyro_data.z as f64 * to_rad;
            let vec_g = Vector3::new(wx, wy, wz);

            let ax = accel_data.x as f64;
            let ay = accel_data.y as f64;
            let az = accel_data.z as f64;
            let vec_a = Vector3::new(ax, ay, az);

            let hw_quat = if let Ok(q) = self.imu.quaternion() {
                if q.s != 0.0 || q.v.x != 0.0 || q.v.y != 0.0 || q.v.z != 0.0 {
                    Some(UnitQuaternion::new_normalize(Quaternion::new(
                        q.s as f64,
                        q.v.x as f64,
                        q.v.y as f64,
                        q.v.z as f64,
                    )))
                } else {
                    None
                }
            } else {
                None
            };

            readings.push(SensorEvent {
                gyro: Some(vec_g),
                accel: Some(vec_a),
                hardware_quaternion: hw_quat,
                dt: Some(safe_dt),
                ..Default::default()
            });

            Ok(readings)
        }

        fn revive(&mut self) -> Result<(), String> {
            warn!("Sensor unresponsive. Sending hardware revive command...");
            self.imu
                .set_mode(BNO055OperationMode::NDOF, &mut self.delay)
                .map_err(|e| format!("Failed to revive: {:?}", e))?;
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

    /// Implementation of `ImuDevice` for the BNO055 IMU.
    pub struct Bno055Device;

    impl Bno055Device {
        /// Creates a new `Bno055Device`.
        pub fn new(_interval: u16, _address: u8) -> Result<Self, String> {
            Err("Hardware I2C is only supported on Linux/Android".into())
        }
    }

    impl ImuDevice for Bno055Device {
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
