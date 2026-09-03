// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

//! `test_bias` is a CLI diagnostic tool for analyzing IMU sensor drift.
//!
//! Note: As of the continuous-EMA update to `olive-imu`, the core engine now automatically
//! subtracts the rolling zero-bias in real-time before broadcasting `ImuUpdate`.
//! Therefore, when running this tool against the modern engine, the EMA calculated here
//! represents the *residual* uncorrected drift (which should aggressively converge to 0.0),
//! proving that the internal engine is correctly clamping the baseline.

use std::time::Duration;

use olive_imu::{
    Imu, bmi160::Bmi160Device, bno055::Bno055Device, bno085::Bno085Device, mpuxxxx::MpuXxxxDevice,
};
use pico_args::Arguments;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut pargs = Arguments::from_env();
    let seconds: u64 = pargs
        .opt_value_from_str(["-s", "--seconds"])
        .unwrap_or(None)
        .unwrap_or(5);

    let use_calibrated: bool = pargs.contains(["-c", "--calibrated"]);

    println!("Probing I2C bus for IMU sensors...");

    let mut imu_engine = None;

    // Try BNO085 at 0x4B on bus 3
    if let Ok(device) = Bno085Device::new(10, 0x4B, use_calibrated, Some(3)) {
        if let Ok(engine) = Imu::start(device, None) {
            println!("BNO085 successfully initialized at 0x4B!");
            imu_engine = Some(engine);
        }
    }

    // Try BNO085 at 0x4A on bus 3
    if imu_engine.is_none() {
        if let Ok(device) = Bno085Device::new(10, 0x4A, use_calibrated, Some(3)) {
            if let Ok(engine) = Imu::start(device, None) {
                println!("BNO085 successfully initialized at 0x4A!");
                imu_engine = Some(engine);
            }
        }
    }

    // Try BNO085 at 0x4B
    if let Ok(device) = Bno085Device::new(10, 0x4B, use_calibrated, None) {
        if let Ok(engine) = Imu::start(device, None) {
            println!("BNO085 successfully initialized at 0x4B!");
            imu_engine = Some(engine);
        }
    }

    // Try BNO085 at 0x4A
    if imu_engine.is_none() {
        if let Ok(device) = Bno085Device::new(10, 0x4A, use_calibrated, None) {
            if let Ok(engine) = Imu::start(device, None) {
                println!("BNO085 successfully initialized at 0x4A!");
                imu_engine = Some(engine);
            }
        }
    }

    // Try BNO055 at 0x28
    if imu_engine.is_none() {
        if let Ok(device) = Bno055Device::new(10, 0x28) {
            if let Ok(engine) = Imu::start(device, None) {
                println!("BNO055 successfully initialized at 0x28!");
                imu_engine = Some(engine);
            }
        }
    }

    // Try BNO055 at 0x29
    if imu_engine.is_none() {
        if let Ok(device) = Bno055Device::new(10, 0x29) {
            if let Ok(engine) = Imu::start(device, None) {
                println!("BNO055 successfully initialized at 0x29!");
                imu_engine = Some(engine);
            }
        }
    }

    // Try BMI160 at 0x68
    if imu_engine.is_none() {
        if let Ok(device) = Bmi160Device::new(0x68) {
            if let Ok(engine) = Imu::start(device, None) {
                println!("BMI160 successfully initialized at 0x68!");
                imu_engine = Some(engine);
            }
        }
    }

    // Try BMI160 at 0x69
    if imu_engine.is_none() {
        if let Ok(device) = Bmi160Device::new(0x69) {
            if let Ok(engine) = Imu::start(device, None) {
                println!("BMI160 successfully initialized at 0x69!");
                imu_engine = Some(engine);
            }
        }
    }

    // Try MPUXXXX at 0x68
    if imu_engine.is_none() {
        if let Ok(device) = MpuXxxxDevice::new(10, 0x68) {
            if let Ok(engine) = Imu::start(device, None) {
                println!("MPUXXXX successfully initialized at 0x68!");
                imu_engine = Some(engine);
            }
        }
    }

    // Try MPUXXXX at 0x69
    if imu_engine.is_none() {
        if let Ok(device) = MpuXxxxDevice::new(10, 0x69) {
            if let Ok(engine) = Imu::start(device, None) {
                println!("MPUXXXX successfully initialized at 0x69!");
                imu_engine = Some(engine);
            }
        }
    }

    let engine = match imu_engine {
        Some(e) => e,
        None => {
            println!("No IMU sensor found on standard I2C addresses.");
            return Ok(());
        }
    };

    println!("--------------------------------------------------");
    println!(
        "Starting continuous bias calibration loop ({} seconds). Press Ctrl-C to exit.",
        seconds
    );
    println!("--------------------------------------------------");

    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            println!("\nCtrl-C received. Shutting down gracefully...");
        }
        _ = async {
            let mut ema_bias = (0.0, 0.0, 0.0);
            let mut total_samples = 0_usize;
            let mut strict_cumulative_sum = (0.0, 0.0, 0.0);

            let mut current_sum = (0.0, 0.0, 0.0);
            let mut current_samples = 0_usize;
            let mut last_timestamp = None;
            let mut last_print = tokio::time::Instant::now();
            let mut previous_motion_state = olive_imu::MotionState::Initializing;

            loop {
                if let Some(update) = engine.get_latest_state() {
                    if Some(update.timestamp) != last_timestamp {
                        last_timestamp = Some(update.timestamp);

                        // Print state transitions so we can see what's happening in real-time
                        if update.motion_state != previous_motion_state {
                            match update.motion_state {
                                olive_imu::MotionState::Initializing => println!("*** IMU State: Initializing (Warming up) ***"),
                                olive_imu::MotionState::Moving => println!("*** IMU State: Moving (EMA frozen) ***"),
                                olive_imu::MotionState::Stable => println!("*** IMU State: Stable (EMA resumed) ***"),
                            }
                            previous_motion_state = update.motion_state;
                        }

                        // We only process stationary samples.
                        if update.motion_state == olive_imu::MotionState::Stable {
                            total_samples += 1;

                            // Dynamic Alpha: Acts exactly like a true cumulative average for the first 2500 samples,
                            // then clamps at 0.0004 to become an Exponential Moving Average (rolling window).
                            let alpha = (1.0 / (total_samples as f64)).max(0.0004);

                            ema_bias.0 = (update.gyro.x * alpha) + (ema_bias.0 * (1.0 - alpha));
                            ema_bias.1 = (update.gyro.y * alpha) + (ema_bias.1 * (1.0 - alpha));
                            ema_bias.2 = (update.gyro.z * alpha) + (ema_bias.2 * (1.0 - alpha));

                            strict_cumulative_sum.0 += update.gyro.x;
                            strict_cumulative_sum.1 += update.gyro.y;
                            strict_cumulative_sum.2 += update.gyro.z;

                            // For display purposes, also track the current interval's simple average
                            current_sum.0 += update.gyro.x;
                            current_sum.1 += update.gyro.y;
                            current_sum.2 += update.gyro.z;
                            current_samples += 1;
                        }
                    }
                }

                if last_print.elapsed().as_secs() >= seconds {
                    if current_samples > 0 && total_samples > 0 {
                        // Metric 1: Window Average
                        // A simple average of only the samples collected during this specific print interval (e.g. the last 2 seconds).
                        // Highly susceptible to noise and short-term thermal fluctuations.
                        let current_bias = (
                            current_sum.0 / current_samples as f64,
                            current_sum.1 / current_samples as f64,
                            current_sum.2 / current_samples as f64,
                        );

                        // Metric 2: Strict Cumulative Average
                        // A mathematically pure average of all samples since boot.
                        // Very stable, but totally blind to slow thermal drift over time because historical samples weigh it down.
                        let strict_bias = (
                            strict_cumulative_sum.0 / total_samples as f64,
                            strict_cumulative_sum.1 / total_samples as f64,
                            strict_cumulative_sum.2 / total_samples as f64,
                        );

                        // Metric 3: EMA Baseline
                        // The dynamic alpha rolling window that perfectly balances short-term noise rejection with long-term drift tracking.

                        let accel_str = if let Some(update) = engine.get_latest_state() {
                            if let Some(accel) = update.gravity_vector {
                                let mag = (accel.x * accel.x + accel.y * accel.y + accel.z * accel.z).sqrt();
                                format!("[{:.5}, {:.5}, {:.5}] ({:.2} m/s^2)", accel.x, accel.y, accel.z, mag)
                            } else {
                                "None".to_string()
                            }
                        } else {
                            "None".to_string()
                        };

                        let quat_str = if let Some(update) = engine.get_latest_state() {
                            if let Some(q) = update.hardware_quaternion {
                                format!("[w:{:.3}, i:{:.3}, j:{:.3}, k:{:.3}]", q.w, q.i, q.j, q.k)
                            } else {
                                "None".to_string()
                            }
                        } else {
                            "None".to_string()
                        };

                        println!(
                            "Gyro Window: [{:.5}, {:.5}, {:.5}] | Gyro EMA: [{:.5}, {:.5}, {:.5}] | Gyro Strict: [{:.5}, {:.5}, {:.5}] | Accel EMA: {} | HW Quat: {} ({} samples)",
                            current_bias.0, current_bias.1, current_bias.2,
                            ema_bias.0, ema_bias.1, ema_bias.2,
                            strict_bias.0, strict_bias.1, strict_bias.2,
                            accel_str,
                            quat_str,
                            total_samples
                        );
                    }

                    // Reset current sum for the next interval
                    current_sum = (0.0, 0.0, 0.0);
                    current_samples = 0;
                    last_print = tokio::time::Instant::now();
                }

                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        } => {}
    }

    Ok(())
}
