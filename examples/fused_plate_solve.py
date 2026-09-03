#!/usr/bin/env python3
# Copyright (c) 2026 Omair Kamil
# See LICENSE file in root directory for license terms.

"""
Example: Live Fused Astronomical Plate Solving in Python

This example demonstrates how to integrate `olive-solve` with a live camera
using `Picamera2` in raw sensor mode (e.g. SRGGB10 for Sony IMX290) and hardware
IMU orientation tracking.

Architecture:
- Thread 1 (Capture & Solve Pipeline):
    1. Drains accumulated camera frames and captures the freshest exposure.
    2. Extracts the 8-bit MSB raw Bayer frame.
    3. Runs the fast star centroid extractor with block-median background
       subtraction and 2x2 Bayer superpixel binning.
    4. Plate-solves the centroids to calculate RA/Dec pointing.
    5. Logs the solve result prefixed with [Solver].
- Thread 2 (Fused Polling Pipeline):
    1. Periodically queries the FusedSolver at a configurable rate (default 10 Hz).
    2. Logs the unified attitude estimate and its source (IMU, Solver, or SolverStale)
       prefixed with [Fused].

Prerequisites:
- Install the olive_solve wheel:
    pip install dist/olive_solve-*.whl
- Install Picamera2 and dependencies:
    sudo apt install python3-picamera2
"""

import sys
from unittest.mock import MagicMock

# Mock PyAV and PyQt5 to avoid pulling in X11/libGL dependencies on headless systems
sys.modules["av"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["PyQt5"] = MagicMock()
sys.modules["PyQt5.QtCore"] = MagicMock()
sys.modules["PyQt5.QtWidgets"] = MagicMock()
sys.modules["PyQt5.QtGui"] = MagicMock()

import argparse
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from picamera2 import Picamera2

import olive_solve


def setup_logging() -> None:
    """Configures standard thread-safe logging with millisecond timestamps."""
    # Suppress libcamera C++ stdout/stderr log spam
    os.environ["LIBCAMERA_LOG_LEVELS"] = "*:ERROR"
    try:
        Picamera2.set_logging(logging.ERROR)
    except Exception:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_database_path(db_arg: str) -> str:
    """
    Resolves the star database path with comprehensive fallbacks:
    1. Direct user-provided path or default path if it exists.
    2. PyInstaller internal bundle directory (if frozen with PyInstaller).
    3. Adjacent to the binary/script (e.g. ./default_database.npz).
    4. Current working directory.
    5. Standard tetra3 test fixture path in repo.
    6. Standard ~/data/default_database.npz.
    """
    candidate = Path(db_arg).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    search_dirs = []

    # 1. PyInstaller bundled temporary directory
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        search_dirs.append(Path(sys._MEIPASS))

    # 2. Executable / script directory
    exe_dir = Path(sys.argv[0]).resolve().parent
    search_dirs.append(exe_dir)

    # 3. Source script parent directory
    script_dir = Path(__file__).resolve().parent
    search_dirs.append(script_dir)
    search_dirs.append(script_dir.parent)

    # 4. Current working directory
    search_dirs.append(Path.cwd())

    # 5. User data directory
    search_dirs.append(Path.home() / "data")

    # Search candidates in all potential root directories
    for root in search_dirs:
        for fname in [db_arg, Path(db_arg).name, "default_database.npz"]:
            alt = (root / fname).resolve()
            if alt.exists() and alt.is_file():
                return str(alt)
        # Also check tetra3 fixture subpath
        alt_fixture = root / "tetra3" / "tests" / "fixtures" / "default_database.npz"
        if alt_fixture.exists():
            return str(alt_fixture.resolve())

    raise FileNotFoundError(
        f"Star database npz file not found (tried '{db_arg}' and standard locations). "
        "Please provide the path using --database <path_to_npz>."
    )


def initialize_camera(
    picam2: Picamera2,
    raw_format: str,
    exposure_ms: float,
) -> Tuple[int, int]:
    """
    Configures the primary camera in raw stream mode with maximum analog gain.

    Returns:
        (height, width) of the sensor stream.
    """
    # Query sensor modes to select full native resolution (e.g. 1920x1080 for IMX290)
    modes = picam2.sensor_modes
    if modes:
        best_mode = max(modes, key=lambda m: m["size"][0] * m["size"][1])
        width, height = best_mode["size"]
    else:
        width, height = 1920, 1080

    logging.info(
        f"Configuring camera in raw mode: format={raw_format}, size={width}x{height}"
    )
    cfg = picam2.create_video_configuration(
        raw={"format": raw_format, "size": (width, height)}
    )
    picam2.configure(cfg)

    # Discover and set maximum analogue gain supported by hardware
    controls = picam2.camera_controls
    max_gain = 16.0
    if "AnalogueGain" in controls:
        gain_range = controls["AnalogueGain"]
        max_gain = gain_range[1]
    logging.info(f"Discovered max analogue gain: {max_gain}")

    exposure_us = int(exposure_ms * 1000.0)
    logging.info(f"Setting manual exposure: {exposure_ms:.1f} ms ({exposure_us} us)")

    picam2.set_controls(
        {
            "AeEnable": False,
            "AnalogueGain": float(max_gain),
            "ExposureTime": exposure_us,
            "FrameDurationLimits": (exposure_us, max(exposure_us, 1_000_000)),
        }
    )

    picam2.start()
    return height, width


def capture_and_solve_worker(
    picam2: Picamera2,
    solver: olive_solve.FusedSolver,
    height: int,
    width: int,
    downsample: int,
    stop_event: threading.Event,
) -> None:
    """
    Single worker thread executing the capture -> extract -> solve pipeline:
    1. Drains accumulated camera exposures to ensure latest frame.
    2. Converts raw 16-bit unpacked data to 8-bit MSB.
    3. Runs fast extraction with block median background subtraction & 2x2 binning.
    4. Plate-solves extracted centroids and logs the result.
    """
    logging.info("Starting capture and solve worker thread...")

    while not stop_event.is_set():
        try:
            # Drain any stale frames accumulated during solving and wait for latest exposure
            req = picam2.capture_request(flush=True)
            if req is None:
                continue

            arr = req.make_array("raw")

            if arr is None or arr.size == 0:
                req.release()
                continue

            # Libcamera outputs SRGGB10 as unpacked 16-bit LSB-aligned words.
            if arr.ndim == 2 and arr.shape[1] == width * 2:
                # Ultrafast in-place shift: move the 10-bit LSB-aligned data right by 2.
                # This places the 8-bit MSB (bits 2-9) squarely into the lower byte (bits 0-7).
                arr_uint16 = arr.view(np.uint16)
                np.right_shift(arr_uint16, 2, out=arr_uint16)

                # Copy the lower bytes into a contiguous array to safely release the camera buffer
                raw_8bit = np.ascontiguousarray(arr[:, 0::2])
            elif arr.ndim == 2 and arr.shape == (height, width):
                raw_8bit = np.ascontiguousarray(arr)
            else:
                # Fallback for alternative array shapes
                raw_8bit = np.ascontiguousarray(arr[:height, :width])

            # Release the camera buffer now that we have copied our 8-bit array
            req.release()

            t_start = time.perf_counter()

            # Fast extraction with block median background subtraction and 2x2 Bayer binning
            centroids = solver.get_centroids_from_image_fast(
                raw_8bit,
                bg_sub_mode="block_median",
                downsample=downsample,
            )

            # Perform celestial plate solve
            solve_result = solver.solve_from_centroids(
                centroids.astype(np.float64),
                (float(height), float(width)),
            )

            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            status = solve_result.get("status", "Unknown")

            if status == "MatchFound" and solve_result.get("RA") is not None:
                ra = solve_result["RA"]
                dec = solve_result["Dec"]
                roll = solve_result.get("Roll", 0.0)
                matches = solve_result.get("Matches", 0)
                logging.info(
                    f"[Solver] Match found: RA={ra:.4f} deg, Dec={dec:.4f} deg, "
                    f"Roll={roll:.4f} deg, Matches={matches}, "
                    f"Centroids={len(centroids)}, PipelineTime={t_elapsed_ms:.1f}ms"
                )
            else:
                logging.info(
                    f"[Solver] No match (status: {status}), "
                    f"Centroids={len(centroids)}, PipelineTime={t_elapsed_ms:.1f}ms"
                )

        except Exception as e:
            if not stop_event.is_set():
                logging.error(f"[Solver] Pipeline error: {e}", exc_info=True)
                time.sleep(0.05)


def fused_polling_worker(
    solver: olive_solve.FusedSolver,
    polling_rate_hz: float,
    stop_event: threading.Event,
) -> None:
    """
    Polling worker thread that queries the FusedSolver at a steady rate
    and logs the current attitude estimate and its source (IMU/Solver/SolverStale).
    """
    interval = 1.0 / max(1.0, polling_rate_hz)
    logging.info(f"Starting fused polling thread at {polling_rate_hz:.1f} Hz...")

    last_await_log = 0.0

    while not stop_event.is_set():
        t0 = time.perf_counter()
        try:
            pos = solver.get_latest_position()
            ra = pos["ra"]
            dec = pos["dec"]
            roll = pos["roll"]
            source = pos["source"]

            logging.info(
                f"[Fused] Position (source: {source}): "
                f"RA={ra:.4f} deg, Dec={dec:.4f} deg, Roll={roll:.4f} deg"
            )
        except RuntimeError:
            # Throttled message when awaiting initial plate solve anchor or IMU tracking
            now = time.time()
            if now - last_await_log >= 2.0:
                logging.info(
                    "[Fused] Awaiting initial position estimate from solver or IMU..."
                )
                last_await_log = now
        except Exception as e:
            if not stop_event.is_set():
                logging.error(f"[Fused] Polling error: {e}")

        # Maintain steady polling rate
        elapsed = time.perf_counter() - t0
        sleep_time = interval - elapsed
        if sleep_time > 0:
            stop_event.wait(timeout=sleep_time)


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Live Python plate solving with Picamera2 and FusedSolver IMU tracking."
    )
    parser.add_argument(
        "--exposure",
        "--exposure-ms",
        dest="exposure_ms",
        type=float,
        default=100.0,
        help="Camera exposure time in milliseconds (default: 100.0 ms)",
    )
    parser.add_argument(
        "--lat",
        "--latitude",
        dest="latitude",
        type=float,
        default=37.35,
        help="Observer latitude in degrees (default: 37.35)",
    )
    parser.add_argument(
        "--lon",
        "--longitude",
        dest="longitude",
        type=float,
        default=-121.96,
        help="Observer longitude in degrees (default: -121.96)",
    )
    parser.add_argument(
        "--rate",
        "--polling-rate",
        dest="polling_rate",
        type=float,
        default=10.0,
        help="Fused solver position polling rate in Hz (default: 10.0 Hz)",
    )
    parser.add_argument(
        "--database",
        "--database-path",
        dest="database_path",
        type=str,
        default="./tetra3/tests/fixtures/default_database.npz",
        help="Path to star database npz file (default: ./tetra3/tests/fixtures/default_database.npz)",
    )
    parser.add_argument(
        "--raw-format",
        type=str,
        default="SRGGB10",
        help="Camera raw format string (default: SRGGB10)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        choices=[1, 2, 4],
        default=2,
        help="Downsampling factor for extraction (1=None, 2=2x2 Bayer binning, 4=4x4 binning) (default: 2)",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_arguments()

    logging.info("==================================================")
    logging.info(" Olive-Solve Python Fused Plate-Solving Example   ")
    logging.info("==================================================")
    logging.info(
        f"Observer Location: Lat {args.latitude} deg, Lon {args.longitude} deg"
    )
    logging.info(f"Exposure Time    : {args.exposure_ms:.1f} ms")
    logging.info(f"Polling Rate     : {args.polling_rate:.1f} Hz")
    logging.info(f"Raw Format       : {args.raw_format}")
    logging.info(f"Downsample       : {args.downsample}")

    db_path = resolve_database_path(args.database_path)
    logging.info(f"Loading star database: {db_path}")

    # 1. Initialize FusedSolver and configure observer location
    solver = olive_solve.FusedSolver(db_path, imu_type="auto")
    solver.set_observer_location(args.latitude, args.longitude)

    # 2. Auto-detect and start hardware IMU
    logging.info("Probing and starting hardware IMU...")
    try:
        imu_started = solver.start_imu()
        logging.info(f"IMU started successfully: {imu_started}")
    except Exception as e:
        logging.warning(
            f"Could not initialize hardware IMU: {e}. Proceeding in solver-only mode."
        )

    # 3. Initialize Picamera2 in raw mode
    picam2 = Picamera2()
    try:
        height, width = initialize_camera(picam2, args.raw_format, args.exposure_ms)
    except Exception as e:
        logging.error(f"Failed to initialize camera: {e}")
        picam2.close()
        solver.stop_imu()
        sys.exit(1)

    stop_event = threading.Event()

    # 4. Handle termination signals (Ctrl+C, SIGTERM)
    def handle_signal(sig, frame):
        logging.info("Termination signal received. Stopping threads...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # 5. Launch worker threads
    # Thread 1: Camera capture, 8-bit MSB extraction, and plate solving
    solve_thread = threading.Thread(
        target=capture_and_solve_worker,
        args=(picam2, solver, height, width, args.downsample, stop_event),
        name="CaptureSolveThread",
        daemon=True,
    )

    # Thread 2: Fused position polling at specified Hz
    poll_thread = threading.Thread(
        target=fused_polling_worker,
        args=(solver, args.polling_rate, stop_event),
        name="FusedPollingThread",
        daemon=True,
    )

    solve_thread.start()
    poll_thread.start()

    # Keep main thread alive waiting for stop signal
    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received.")
        stop_event.set()

    logging.info("Shutting down workers and hardware resources...")
    solve_thread.join(timeout=3.0)
    poll_thread.join(timeout=2.0)

    try:
        picam2.stop()
        picam2.close()
        logging.info("Camera stopped and closed.")
    except Exception as e:
        logging.warning(f"Error stopping camera: {e}")

    try:
        solver.stop_imu()
        logging.info("IMU stopped.")
    except Exception as e:
        logging.warning(f"Error stopping IMU: {e}")

    logging.info("Example terminated cleanly.")


if __name__ == "__main__":
    main()
