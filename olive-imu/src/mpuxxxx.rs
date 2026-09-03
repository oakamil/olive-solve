// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

#[cfg(any(target_os = "linux", target_os = "android"))]
mod hardware {
    use crate::imu::{ImuDevice, SensorEvent};
    use linux_embedded_hal::I2cdev;
    use log::{info, warn};
    use mpu6050_driver::{Address, Dlpf, GyroRange, Mpu6050};
    use nalgebra::Vector3;
    use std::time::{Duration, SystemTime};

    pub struct MpuXxxxDevice {
        mpu: Mpu6050<I2cdev>,
        report_interval_ms: u16,
        last_system_time: Option<SystemTime>,
    }

    impl MpuXxxxDevice {
        pub fn new(report_interval_ms: u16, addr_u8: u8) -> Result<Self, String> {
            info!(
                "Initializing MPU series hardware over I2C at address 0x{:X}...",
                addr_u8
            );
            let i2c =
                I2cdev::new("/dev/i2c-1").map_err(|e| format!("I2cdev::new failed: {:?}", e))?;

            let addr = if addr_u8 == 0x68 {
                Address::Ad0Low
            } else if addr_u8 == 0x69 {
                Address::Ad0High
            } else {
                return Err(format!("Invalid MPU address: 0x{:X}", addr_u8));
            };
            let mut mpu = Mpu6050::new(i2c, addr);

            // Wake up the device (this also validates I2C communication)
            mpu.wake()
                .map_err(|_| "Failed to wake up MPU (device not responding)".to_string())?;

            // Configure Gyro range and Digital Low Pass Filter
            mpu.set_gyro_range(GyroRange::Dps2000).ok();
            mpu.set_dlpf(Dlpf::Cfg3).ok(); // ~41Hz bandwidth

            // Configure sample rate divider for ~100Hz.
            // Cfg1 base rate is 1kHz. 1000 / (1 + 9) = 100Hz.
            mpu.set_sample_rate_divider(9).ok();

            Ok(Self {
                mpu,
                report_interval_ms,
                last_system_time: None,
            })
        }
    }

    impl ImuDevice for MpuXxxxDevice {
        fn init(&mut self) -> Result<(), String> {
            Ok(())
        }

        fn poll(&mut self) -> Result<Vec<SensorEvent>, String> {
            let mut readings = Vec::new();

            // Real-time register read
            let raw = match self.mpu.read_raw_accel_gyro_temp() {
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

            // Convert to rad/sec based on 2000 dps scale (Scale factor: 16.4 LSB/dps)
            let scale = 16.4;
            let deg2rad = std::f64::consts::PI / 180.0;
            let wx = (raw.gyro[0] as f64 / scale) * deg2rad;
            let wy = (raw.gyro[1] as f64 / scale) * deg2rad;
            let wz = (raw.gyro[2] as f64 / scale) * deg2rad;
            let vec_g = Vector3::new(wx, wy, wz);

            // Convert accel to m/s^2. MPU6050 default accel scale is +-2g (16384 LSB/g)
            let accel_scale = 16384.0 / 9.81;
            let ax = raw.accel[0] as f64;
            let ay = raw.accel[1] as f64;
            let az = raw.accel[2] as f64;
            let vec_a = Vector3::new(ax / accel_scale, ay / accel_scale, az / accel_scale);

            readings.push(SensorEvent {
                gyro: Some(vec_g),
                accel: Some(vec_a),
                dt: Some(safe_dt),
                ..Default::default()
            });

            Ok(readings)
        }

        fn revive(&mut self) -> Result<(), String> {
            warn!("Sensor unresponsive. Resetting hardware...");
            self.mpu
                .reset()
                .map_err(|e| format!("Failed to reset: {:?}", e))?;
            // Wait for reset to complete
            std::thread::sleep(Duration::from_millis(100));
            self.mpu.wake().ok();
            self.mpu.set_gyro_range(GyroRange::Dps2000).ok();
            self.mpu.set_dlpf(Dlpf::Cfg3).ok();
            self.mpu.set_sample_rate_divider(9).ok();
            Ok(())
        }

        fn needs_seeding(&self) -> bool {
            true
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub use hardware::*;

#[cfg(not(any(target_os = "linux", target_os = "android")))]
mod stub {
    use crate::imu::{ImuDevice, SensorEvent};
    use nalgebra::Vector3;

    pub struct MpuXxxxDevice;

    // Use a generic type or u8 to avoid pulling in the driver
    impl MpuXxxxDevice {
        pub fn new(_interval: u16, _address: u8) -> Result<Self, String> {
            Err("Hardware I2C is only supported on Linux/Android".into())
        }
    }

    impl ImuDevice for MpuXxxxDevice {
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
