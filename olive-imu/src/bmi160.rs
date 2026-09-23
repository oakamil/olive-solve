// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use crate::imu::SensorEvent;
use nalgebra::Vector3;

/// Pure function to parse raw BMI160 FIFO data into discrete `SensorEvent` instances.
///
/// # Arguments
/// * `fifo_data` - The exact byte slice of FIFO payload (excluding any I2C register prefix).
/// * `last_sensor_time` - The 24-bit sensortime timestamp recorded from the previous gyro poll.
///
/// # Returns
/// A tuple containing:
/// * `Vec<SensorEvent>` - Decoded sensor events with little-endian conversion, scaling, and dt.
/// * `Option<u32>` - The updated sensortime anchor for subsequent polls.
pub fn parse_bmi160_fifo(
    fifo_data: &[u8],
    last_sensor_time: Option<u32>,
) -> (Vec<SensorEvent>, Option<u32>) {
    let mut i = 0;
    let mut frames = Vec::new();
    let mut current_sensor_time: Option<u32> = None;
    let mut next_last_time = last_sensor_time;

    while i < fifo_data.len() {
        let header = fifo_data[i];
        i += 1;

        // Mask out interrupt tag bits (bits 1:0) for regular frames
        let base_header = header & 0xFC;

        if base_header == 0x88 {
            // Gyro-only frame (Header + 6 bytes)
            if i + 6 <= fifo_data.len() {
                let gx = (fifo_data[i] as u16 | ((fifo_data[i + 1] as u16) << 8)) as i16;
                let gy = (fifo_data[i + 2] as u16 | ((fifo_data[i + 3] as u16) << 8)) as i16;
                let gz = (fifo_data[i + 4] as u16 | ((fifo_data[i + 5] as u16) << 8)) as i16;
                let deg_to_rad = std::f64::consts::PI / 180.0;
                // 2000 deg/s range = 16.4 LSB per deg/s
                let wx = (gx as f64 / 16.4) * deg_to_rad;
                let wy = (gy as f64 / 16.4) * deg_to_rad;
                let wz = (gz as f64 / 16.4) * deg_to_rad;
                frames.push((Some(Vector3::new(wx, wy, wz)), None));
                i += 6;
            } else {
                break; // Malformed / truncated frame
            }
        } else if base_header == 0x84 {
            // Accel-only frame (Header + 6 bytes)
            if i + 6 <= fifo_data.len() {
                let ax = (fifo_data[i] as u16 | ((fifo_data[i + 1] as u16) << 8)) as i16;
                let ay = (fifo_data[i + 2] as u16 | ((fifo_data[i + 3] as u16) << 8)) as i16;
                let az = (fifo_data[i + 4] as u16 | ((fifo_data[i + 5] as u16) << 8)) as i16;
                // 2G range = 16384 LSB/g, 1g = 9.81 m/s^2
                let scale = 16384.0 / 9.81;
                let vec_a = Vector3::new(ax as f64 / scale, ay as f64 / scale, az as f64 / scale);
                frames.push((None, Some(vec_a)));
                i += 6;
            } else {
                break; // Malformed / truncated frame
            }
        } else if base_header == 0x8C {
            // Gyro + Accel frame (Header + 12 bytes: 6 Gyro, then 6 Accel)
            if i + 12 <= fifo_data.len() {
                let gx = (fifo_data[i] as u16 | ((fifo_data[i + 1] as u16) << 8)) as i16;
                let gy = (fifo_data[i + 2] as u16 | ((fifo_data[i + 3] as u16) << 8)) as i16;
                let gz = (fifo_data[i + 4] as u16 | ((fifo_data[i + 5] as u16) << 8)) as i16;
                let deg_to_rad = std::f64::consts::PI / 180.0;
                let wx = (gx as f64 / 16.4) * deg_to_rad;
                let wy = (gy as f64 / 16.4) * deg_to_rad;
                let wz = (gz as f64 / 16.4) * deg_to_rad;

                let ax = (fifo_data[i + 6] as u16 | ((fifo_data[i + 7] as u16) << 8)) as i16;
                let ay = (fifo_data[i + 8] as u16 | ((fifo_data[i + 9] as u16) << 8)) as i16;
                let az = (fifo_data[i + 10] as u16 | ((fifo_data[i + 11] as u16) << 8)) as i16;
                let scale = 16384.0 / 9.81;
                let vec_a = Vector3::new(ax as f64 / scale, ay as f64 / scale, az as f64 / scale);

                frames.push((Some(Vector3::new(wx, wy, wz)), Some(vec_a)));
                i += 12;
            } else {
                break; // Malformed / truncated frame
            }
        } else if header == 0x44 {
            // Sensortime frame (Header + 3 bytes)
            if i + 3 <= fifo_data.len() {
                let time = (fifo_data[i] as u32)
                    | ((fifo_data[i + 1] as u32) << 8)
                    | ((fifo_data[i + 2] as u32) << 16);
                current_sensor_time = Some(time);
                i += 3;
            } else {
                break;
            }
        } else if header == 0x40 {
            // Skip frame (Header + 1 byte skip count)
            if i < fifo_data.len() {
                i += 1;
            } else {
                break;
            }
        } else if header == 0x80 {
            // Empty / Invalid (end of valid data in FIFO)
            break;
        } else {
            // Unknown header
            log::warn!("BMI160 unknown header: 0x{:X} at index {}", header, i - 1);
            break;
        }
    }

    let num_gyro = frames.iter().filter(|(g, _)| g.is_some()).count();

    let total_dt = if let Some(time) = current_sensor_time {
        let dt = if let Some(last_time) = last_sensor_time {
            let mut diff = time as i64 - last_time as i64;
            if diff < -8_000_000 {
                // 24-bit counter wrapped past 0x1000000
                diff += 0x1000000;
            } else if diff < 0 {
                // Negative clock jitter or duplicate timestamp
                diff = 0;
            }
            // SENSORTIME runs at 25.6 kHz (exactly 39.0625 us per tick)
            (diff as f64) * 39.0625e-6
        } else {
            // Initial poll fallback: assume 10ms per gyro frame (100Hz ODR)
            0.01 * (num_gyro as f64).max(1.0)
        };

        // Accel-Only Time Slip Fix: only advance anchor if gyro frames consumed the time
        if num_gyro > 0 {
            next_last_time = Some(time);
        }
        dt
    } else {
        // Missing Sensortime Double-Count Fix: invalidate anchor if gyro had to use fallback
        if num_gyro > 0 {
            next_last_time = None;
        }
        0.01 * (num_gyro as f64).max(1.0)
    };

    // Per-Frame Clamping: enforce safe bounds per integration step
    let gyro_dt = if num_gyro > 0 {
        let raw_gyro_dt = total_dt / (num_gyro as f64);
        raw_gyro_dt.clamp(0.0, 0.050)
    } else {
        0.0
    };

    let mut readings = Vec::new();
    for (opt_g, opt_a) in frames {
        if let (Some(g), Some(a)) = (opt_g, opt_a) {
            readings.push(SensorEvent {
                gyro: Some(g),
                accel: Some(a),
                dt: Some(gyro_dt),
                ..Default::default()
            });
        } else if let Some(g) = opt_g {
            readings.push(SensorEvent {
                gyro: Some(g),
                dt: Some(gyro_dt),
                ..Default::default()
            });
        } else if let Some(a) = opt_a {
            readings.push(SensorEvent {
                accel: Some(a),
                dt: None, // Accel-only events must never carry dt
                ..Default::default()
            });
        }
    }

    (readings, next_last_time)
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod hardware {
    use super::parse_bmi160_fifo;
    use bmi160::{
        AccelerometerPowerMode, AccelerometerRange, Bmi160, GyroscopeBwp, GyroscopeOdr,
        GyroscopePowerMode, GyroscopeRange, SlaveAddr, interface::I2cInterface,
    };
    use linux_embedded_hal::I2cdev;
    use log::{info, warn};

    use crate::imu::{ImuDevice, SensorEvent};

    pub struct Bmi160Device {
        imu: Bmi160<I2cInterface<I2cdev>>,
        last_sensor_time: Option<u32>,
        enable_accel: bool,
    }

    impl Bmi160Device {
        pub fn new(address_u8: u8, enable_accel: bool) -> Result<Self, String> {
            info!(
                "Initializing BMI160 hardware over I2C at address 0x{:X}...",
                address_u8
            );
            let mut i2c =
                I2cdev::new("/dev/i2c-1").map_err(|e| format!("I2cdev::new failed: {:?}", e))?;

            // 1. Primary Check: BMI160 Chip ID
            use embedded_hal::i2c::I2c;
            let mut chip_id = [0u8; 1];
            if i2c.write_read(address_u8, &[0x00], &mut chip_id).is_err() || chip_id[0] != 0xD1 {
                return Err(format!("BMI160 chip ID mismatch at 0x{:X}", address_u8));
            }

            // 2. Secondary Check: MPU WHO_AM_I elimination
            let mut mpu_id = [0u8; 1];
            if i2c.write_read(address_u8, &[0x75], &mut mpu_id).is_ok() {
                // Reject if it identifies as an MPU
                if matches!(mpu_id[0], 0x68 | 0x70 | 0x71 | 0x73 | 0x75) {
                    return Err(
                        "Sensor matches an MPU series identity, rejecting BMI160 initialization"
                            .into(),
                    );
                }
            }
            let address = if address_u8 == 0x69 {
                SlaveAddr::Alternative(true)
            } else {
                SlaveAddr::Default
            };
            let imu = Bmi160::new_with_i2c(i2c, address);

            Ok(Self {
                imu,
                last_sensor_time: None,
                enable_accel,
            })
        }
    }

    impl ImuDevice for Bmi160Device {
        fn init(&mut self) -> Result<(), String> {
            // Set gyro range to 2000 deg/s for high movement applications
            self.imu
                .set_gyro_range(GyroscopeRange::Scale2000)
                .map_err(|_| "Failed to set BMI160 gyro range".to_string())?;

            // Configure hardware DLPF for maximum smoothing (100Hz ODR, OSR4)
            self.imu
                .set_gyro_conf(GyroscopeOdr::Hz100, GyroscopeBwp::Osr4)
                .map_err(|_| "Failed to configure BMI160 gyro filtering".to_string())?;

            self.imu
                .set_accel_range(AccelerometerRange::G2)
                .map_err(|_| "Failed to set BMI160 accel range".to_string())?;

            // Turn on the gyro
            self.imu
                .set_gyro_power_mode(GyroscopePowerMode::Normal)
                .map_err(|_| "Failed to enable BMI160 gyro".to_string())?;

            // BMI160 needs ~100ms for gyro to fully turn on from suspend
            std::thread::sleep(std::time::Duration::from_millis(100));

            if self.enable_accel {
                // Turn on the accel
                self.imu
                    .set_accel_power_mode(AccelerometerPowerMode::Normal)
                    .map_err(|_| "Failed to enable BMI160 accel".to_string())?;

                // Accel needs at least 10ms to transition
                std::thread::sleep(std::time::Duration::from_millis(10));
            }

            // Configure FIFO for Header mode + Gyro + Time
            self.imu
                .config_fifo()
                .map_err(|_| "Failed to configure BMI160 FIFO".to_string())?;

            info!(
                "BMI160 initialized. Gyroscope running at 2000 deg/s range, Accel at 2G, with hardware FIFO enabled."
            );
            Ok(())
        }

        fn poll(&mut self) -> Result<Vec<SensorEvent>, String> {
            let mut buffer = [0u8; 1024];

            match self.imu.read_fifo(&mut buffer) {
                Ok(len) if len > 0 => {
                    let (readings, new_time) =
                        parse_bmi160_fifo(&buffer[1..=len], self.last_sensor_time);
                    self.last_sensor_time = new_time;
                    Ok(readings)
                }
                Ok(_) => Ok(Vec::new()), // len == 0
                Err(e) => {
                    log::warn!("BMI160 FIFO read error: {:?}", e);
                    // Return empty vec on transient read errors rather than crashing the system.
                    // The Imu watchdog will handle revive() if it drops too many packets.
                    Ok(Vec::new())
                }
            }
        }

        fn revive(&mut self) -> Result<(), String> {
            warn!("BMI160 unresponsive. Sending hardware revive command...");
            // Re-assert power mode and reconfigure FIFO in an attempt to wake up the sensor
            self.imu
                .set_gyro_power_mode(GyroscopePowerMode::Normal)
                .map_err(|_| "Failed to revive BMI160 gyro".to_string())?;

            std::thread::sleep(std::time::Duration::from_millis(100));

            if self.enable_accel {
                self.imu
                    .set_accel_power_mode(AccelerometerPowerMode::Normal)
                    .map_err(|_| "Failed to revive BMI160 accel".to_string())?;

                std::thread::sleep(std::time::Duration::from_millis(10));
            }

            let _ = self.imu.config_fifo();

            std::thread::sleep(std::time::Duration::from_millis(100));
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

    /// Implementation of `ImuDevice` for the BMI160 IMU.
    pub struct Bmi160Device;

    impl Bmi160Device {
        /// Creates a new `Bmi160Device`.
        pub fn new(_address: u8, _enable_accel: bool) -> Result<Self, String> {
            Err("Hardware I2C is only supported on Linux/Android".into())
        }
    }

    impl ImuDevice for Bmi160Device {
        fn init(&mut self) -> Result<(), String> {
            Err("Unsupported".into())
        }
        fn poll(&mut self) -> Result<Vec<SensorEvent>, String> {
            Err("Unsupported".into())
        }
        fn revive(&mut self) -> Result<(), String> {
            Err("Unsupported".into())
        }

        fn needs_seeding(&self) -> bool {
            true
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub use stub::*;

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    #[test]
    fn test_bmi160_parse_standard_frames() {
        // Frame: 0x8C (Gyro + Accel) + 0x44 (Sensortime: 256 ticks = 10ms)
        let mut data = vec![0x8C];
        // Gyro: gx=164, gy=-164, gz=0 -> wx=10 deg/s, wy=-10 deg/s
        data.extend_from_slice(&164i16.to_le_bytes());
        data.extend_from_slice(&(-164i16).to_le_bytes());
        data.extend_from_slice(&0i16.to_le_bytes());
        // Accel: ax=0, ay=0, az=16384 -> az=9.81 m/s^2 (1G)
        data.extend_from_slice(&0i16.to_le_bytes());
        data.extend_from_slice(&0i16.to_le_bytes());
        data.extend_from_slice(&16384i16.to_le_bytes());
        // Sensortime: 256 ticks
        data.extend_from_slice(&[0x44, 0x00, 0x01, 0x00]);

        let (events, next_time) = parse_bmi160_fifo(&data, Some(0));
        assert_eq!(events.len(), 1);

        let g = events[0].gyro.expect("Expected gyro");
        let a = events[0].accel.expect("Expected accel");

        let expected_rad = 10.0 * (std::f64::consts::PI / 180.0);
        assert!((g.x - expected_rad).abs() < 1e-4);
        assert!((g.y - -expected_rad).abs() < 1e-4);
        assert!((g.z - 0.0).abs() < 1e-4);

        assert!((a.z - 9.81).abs() < 1e-3);

        let dt = events[0].dt.expect("Expected dt");
        assert!((dt - 0.010).abs() < EPSILON);
        assert_eq!(next_time, Some(256));
    }

    #[test]
    fn test_bmi160_parse_independent_frames() {
        // Accel only (0x84) followed by Sensortime (500)
        let data = vec![0x84, 0, 0, 0, 0, 0, 64, 0x44, 0xF4, 0x01, 0x00];
        let (events, next_time) = parse_bmi160_fifo(&data, Some(100));

        assert_eq!(events.len(), 1);
        assert!(events[0].accel.is_some());
        assert!(events[0].gyro.is_none());
        assert_eq!(events[0].dt, None); // Accel-only events must never carry dt
        assert_eq!(next_time, Some(100)); // Anchor must NOT advance without gyro
    }

    #[test]
    fn test_bmi160_interrupt_tag_masking() {
        // Tagged frames: 0x85 (Accel+tag), 0x89 (Gyro+tag), 0x8D (Both+tag1), 0x8E (Both+tag2)
        for header in [0x85, 0x89, 0x8D, 0x8E] {
            let mut data = vec![header];
            let payload_len = if (header & 0xFC) == 0x8C { 12 } else { 6 };
            data.extend_from_slice(&vec![0u8; payload_len]);

            let (events, _) = parse_bmi160_fifo(&data, None);
            assert_eq!(
                events.len(),
                1,
                "Failed to parse tagged header 0x{:X}",
                header
            );
        }
    }

    #[test]
    fn test_bmi160_sensortime_wraparound() {
        // Rollover from 0x00FF_FF80 to 0x0000_0080 (exactly 256 ticks elapsed = 10ms)
        let mut data = vec![0x88, 0, 0, 0, 0, 0, 0];
        data.extend_from_slice(&[0x44, 0x80, 0x00, 0x00]);

        let (events, next_time) = parse_bmi160_fifo(&data, Some(0x00FF_FF80));
        assert_eq!(events.len(), 1);

        let dt = events[0].dt.unwrap();
        assert!((dt - 0.010).abs() < EPSILON);
        assert_eq!(next_time, Some(0x0000_0080));
    }

    #[test]
    fn test_bmi160_per_frame_stall_clamping() {
        // Simulate a 200ms thread stall with 1 surviving gyro frame (5120 ticks)
        let mut data = vec![0x88, 0, 0, 0, 0, 0, 0];
        data.extend_from_slice(&[0x44, 0x00, 0x14, 0x00]); // 5120 ticks = 200ms

        let (events, next_time) = parse_bmi160_fifo(&data, Some(0));
        assert_eq!(events.len(), 1);

        // Clamped strictly to 50ms maximum per frame to protect filter stability
        let dt = events[0].dt.unwrap();
        assert!((dt - 0.050).abs() < EPSILON);
        assert_eq!(next_time, Some(5120));
    }

    #[test]
    fn test_bmi160_missing_sensortime_fallback() {
        // Gyro frame without a trailing 0x44 sensortime frame
        let data = vec![0x88, 0, 0, 0, 0, 0, 0];
        let (events, next_time) = parse_bmi160_fifo(&data, Some(1000));

        assert_eq!(events.len(), 1);
        assert!((events[0].dt.unwrap() - 0.010).abs() < EPSILON);
        assert_eq!(next_time, None); // Anchor invalidated to avoid double-counting
    }

    #[test]
    fn test_bmi160_multi_frame_and_skip() {
        // Frame 1 (0x8C) + Skip Frame (0x40 + 1 count) + Frame 2 (0x8C) + Sensortime (512 ticks = 20ms)
        let mut data = vec![0x8C];
        data.extend_from_slice(&[0u8; 12]);
        data.extend_from_slice(&[0x40, 0x01]); // Skip 1 frame
        data.push(0x8C);
        data.extend_from_slice(&[0u8; 12]);
        data.extend_from_slice(&[0x44, 0x00, 0x02, 0x00]); // 512 ticks

        let (events, next_time) = parse_bmi160_fifo(&data, Some(0));
        assert_eq!(events.len(), 2);
        // Total 20ms distributed across 2 gyro frames = 10ms per frame
        assert!((events[0].dt.unwrap() - 0.010).abs() < EPSILON);
        assert!((events[1].dt.unwrap() - 0.010).abs() < EPSILON);
        assert_eq!(next_time, Some(512));
    }

    #[test]
    fn test_bmi160_sequential_multipoll_lifecycle() {
        // Poll 1: Gyro+Accel at T=1000
        let mut p1 = vec![0x8C];
        p1.extend_from_slice(&[0u8; 12]);
        p1.extend_from_slice(&[0x44, 0xE8, 0x03, 0x00]); // 1000
        let (ev1, t1) = parse_bmi160_fifo(&p1, None);
        assert_eq!(t1, Some(1000));
        assert!((ev1[0].dt.unwrap() - 0.010).abs() < EPSILON);

        // Poll 2: Accel-only at T=1256 (+10ms)
        let mut p2 = vec![0x84];
        p2.extend_from_slice(&[0u8; 6]);
        p2.extend_from_slice(&[0x44, 0xE8, 0x04, 0x00]); // 1256
        let (ev2, t2) = parse_bmi160_fifo(&p2, t1);
        assert_eq!(ev2[0].dt, None);
        assert_eq!(t2, Some(1000)); // Preserved!

        // Poll 3: Gyro-only at T=1512 (+20ms total since Poll 1)
        let mut p3 = vec![0x88];
        p3.extend_from_slice(&[0u8; 6]);
        p3.extend_from_slice(&[0x44, 0xE8, 0x05, 0x00]); // 1512
        let (ev3, t3) = parse_bmi160_fifo(&p3, t2);
        assert_eq!(t3, Some(1512));
        // Conserves elapsed 20ms (512 ticks) since Poll 1
        assert!((ev3[0].dt.unwrap() - 0.020).abs() < EPSILON);
    }

    #[test]
    fn test_bmi160_malformed_buffer() {
        // Truncated 0x8C frame (only 3 bytes of payload)
        let truncated = vec![0x8C, 0x01, 0x02, 0x03];
        let (events, _) = parse_bmi160_fifo(&truncated, None);
        assert_eq!(events.len(), 0); // Safely aborts without panicking

        // Empty FIFO sentinel (0x80)
        let empty_fifo = vec![0x80, 0xFF, 0xFF];
        let (events, _) = parse_bmi160_fifo(&empty_fifo, Some(500));
        assert_eq!(events.len(), 0);

        // Zero-length buffer
        let (events, next_time) = parse_bmi160_fifo(&[], Some(500));
        assert_eq!(events.len(), 0);
        assert_eq!(next_time, Some(500));
    }
}
