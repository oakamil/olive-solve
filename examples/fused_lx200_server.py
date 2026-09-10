#!/usr/bin/env python3
# Copyright (c) 2026 Omair Kamil
# See LICENSE file in root directory for license terms.

"""
Example: Standalone Fused Plate Solver with LX200 Server

This example demonstrates how to integrate `olive-solve` with a live camera
and host an LX200 protocol telescope server on port 4030.

Architecture:
- Thread 1 (Capture & Solve Pipeline):
    1. Reads raw camera frames (e.g. SRGGB10) with maximum analog gain.
    2. Uses an optimized unpacking and buffer reuse path.
    3. Runs the fast star centroid extractor with global median background
       subtraction and 2x2 Bayer binning, plate-solving each exposure.
- Thread 2 (LX200 Server on port 4030):
    1. Supports a single client connection (e.g. Stellarium, SkySafari).
    2. Queries the FusedSolver's latest position without locks.
    3. Handles coordinate epoch conversions (J2000 vs JNow).
    4. When time/date are provided, synchronizes the system clock.
    5. When observer location is provided, configures the solver and starts the IMU.
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
from datetime import datetime, timezone, timedelta
import logging
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
from typing import Optional, Tuple

import numpy as np
from picamera2 import Picamera2

import olive_solve


def setup_logging() -> None:
    """Configures standard thread-safe logging with millisecond timestamps."""
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
    """Resolves the star database path with standard fallbacks."""
    candidate = Path(db_arg).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    search_dirs = []
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        search_dirs.append(Path(sys._MEIPASS))

    search_dirs.append(Path(sys.argv[0]).resolve().parent)
    script_dir = Path(__file__).resolve().parent
    search_dirs.append(script_dir)
    search_dirs.append(script_dir.parent)
    search_dirs.append(Path.cwd())
    search_dirs.append(Path.home() / "data")

    for root in search_dirs:
        for fname in [db_arg, Path(db_arg).name, "default_database.npz"]:
            alt = (root / fname).resolve()
            if alt.exists() and alt.is_file():
                return str(alt)
        alt_fixture = root / "tetra3" / "tests" / "fixtures" / "default_database.npz"
        if alt_fixture.exists():
            return str(alt_fixture.resolve())

    raise FileNotFoundError(
        f"Star database npz file not found (tried '{db_arg}' and standard locations). "
        "Please provide the path using --database <path_to_npz>."
    )


# ==========================================
# ASTRONOMICAL COORDINATE & PRECESSION UTILS
# ==========================================

def to_hms(n: float) -> Tuple[int, int, int]:
    """
    Converts decimal degrees or hours to (hours_or_degrees, minutes, seconds).
    Rounds seconds and cascades overflow to avoid 60-second / 60-minute anomalies.
    """
    n_abs = abs(n)
    hours = int(math.floor(n_abs))
    h_rem = (n_abs - hours) * 60.0
    minutes = int(math.floor(h_rem))
    m_rem = (h_rem - minutes) * 60.0
    seconds = int(round(m_rem))

    if seconds == 60:
        seconds = 0
        minutes += 1
        if minutes == 60:
            minutes = 0
            hours += 1
    if hours == 24:
        hours = 0
    if n < 0.0:
        hours = -hours
    return hours, minutes, seconds


def parse_coordinates(d_str: str, m_str: str, s_str: str) -> Optional[float]:
    """Parses degree/hour, minute, and second substrings into decimal degrees."""
    try:
        deg = int(d_str)
        minutes = int(m_str)
        seconds = int(s_str)
    except ValueError:
        return None

    is_negative = deg < 0 or d_str.startswith("-")
    val = abs(deg) + (minutes / 60.0) + (seconds / 3600.0)
    return -val if is_negative else val


def parse_location(deg_str: str, min_str: str) -> Optional[float]:
    """Parses degrees and minutes into decimal degrees."""
    return parse_coordinates(deg_str, min_str, "0")


def _precession_angles_iau1976(t_centuries: float) -> Tuple[float, float, float]:
    """
    Calculates the IAU 1976 precession angles (zeta, z, theta) in radians
    for t Julian centuries from J2000.0.
    """
    t = t_centuries
    t2 = t * t
    t3 = t2 * t
    sec_to_rad = math.pi / (180.0 * 3600.0)
    zeta = (2306.2181 * t + 0.30188 * t2 + 0.017998 * t3) * sec_to_rad
    z = (2306.2181 * t + 1.09468 * t2 + 0.018203 * t3) * sec_to_rad
    theta = (2004.3109 * t - 0.42665 * t2 - 0.041833 * t3) * sec_to_rad
    return zeta, z, theta


def _precession_matrix_j2000_to_epoch(epoch: float) -> np.ndarray:
    """Returns the 3x3 rotation matrix from J2000.0 to the specified Julian epoch."""
    t = (epoch - 2000.0) / 100.0
    zeta, z, theta = _precession_angles_iau1976(t)

    cz, sz = math.cos(zeta), math.sin(zeta)
    ct, st = math.cos(theta), math.sin(theta)
    czz, szz = math.cos(z), math.sin(z)

    # R_z(-z) * R_y(theta) * R_z(-zeta)
    p = np.array([
        [cz * ct * czz - sz * szz, -sz * ct * czz - cz * szz, -st * czz],
        [cz * ct * szz + sz * czz, -sz * ct * szz + cz * czz, -st * szz],
        [cz * st,                 -sz * st,                  ct],
    ], dtype=np.float64)
    return p


def precess_coordinates(
    ra: float,
    dec: float,
    epoch_from: float,
    epoch_to: float,
) -> Tuple[float, float]:
    """
    Precesses RA and Dec in degrees from epoch_from to epoch_to.
    """
    if abs(epoch_from - epoch_to) < 1e-5:
        return ra, dec

    p_from = _precession_matrix_j2000_to_epoch(epoch_from)
    p_to = _precession_matrix_j2000_to_epoch(epoch_to)
    rot_matrix = p_to @ p_from.T

    ra_rad = math.radians(ra)
    dec_rad = math.radians(dec)

    v_init = np.array([
        math.cos(dec_rad) * math.cos(ra_rad),
        math.cos(dec_rad) * math.sin(ra_rad),
        math.sin(dec_rad),
    ], dtype=np.float64)

    v_prec = rot_matrix @ v_init

    prec_ra = math.degrees(math.atan2(v_prec[1], v_prec[0])) % 360.0
    prec_dec = math.degrees(math.asin(max(-1.0, min(1.0, v_prec[2]))))
    return prec_ra, prec_dec


# ==========================================
# PIPELINE METRICS & TRACKING
# ==========================================

class PipelineTracker:
    """Thread-safe tracker for camera capture & plate solve pipeline metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.pipeline_time_ms: float = 0.0
        self.status: str = "Unknown"
        self.matches: int = 0
        self.ra: Optional[float] = None
        self.dec: Optional[float] = None
        self.roll: Optional[float] = None
        self.timestamp: float = 0.0

    def update(self, dt_ms: float, solve_result: dict) -> None:
        with self._lock:
            self.pipeline_time_ms = dt_ms
            self.status = solve_result.get("status", "Unknown")
            self.matches = solve_result.get("Matches", 0)
            self.ra = solve_result.get("RA")
            self.dec = solve_result.get("Dec")
            self.roll = solve_result.get("Roll")
            self.timestamp = time.time()

    def snapshot(self) -> Tuple[float, str, int, Optional[float], Optional[float], Optional[float], float]:
        with self._lock:
            return (
                self.pipeline_time_ms,
                self.status,
                self.matches,
                self.ra,
                self.dec,
                self.roll,
                self.timestamp,
            )


# ==========================================
# LX200 CONTROLLER LOGIC
# ==========================================

class Lx200Controller:
    """
    Handles LX200 protocol parsing, telescope pointing queries, and command execution.
    Mirrors the command handling and coordinate state machine in telescope_server.rs.
    """

    def __init__(self, solver: olive_solve.FusedSolver, pipeline_tracker: PipelineTracker):
        self.solver = solver
        self.pipeline_tracker = pipeline_tracker
        self.client_buffer = bytearray()
        self.is_stellarium = False

        # Current date and time tracked by controller
        now_dt = datetime.now().astimezone()
        self.datetime = now_dt
        self.jnow_epoch = round((now_dt.year + (now_dt.timetuple().tm_yday - 1) / 365.0) * 10.0) / 10.0

        # Observer site strings formatted for LX200 responses
        self.latitude_str = "+00:00"
        self.longitude_str = "000:00"
        self.site_lat: Optional[float] = None
        self.site_lon: Optional[float] = None
        self.imu_started = False

        # Target coordinates and pending time commands
        self.target_ra: Optional[float] = None
        self.target_dec: Optional[float] = None
        self.pending_timezone: Optional[str] = None
        self.pending_time: Optional[str] = None

        # Slew state and atomic Dec snapshot for :GD# queries
        self.is_slewing = False
        self.snapshot_dec_jnow: Optional[float] = None
        self.last_gr_log_time: float = 0.0
        self.last_is_calibrated: bool = False


    def convert_to_jnow(self, ra: float, dec: float) -> Tuple[float, float]:
        """Converts coordinates from J2000 to JNow unless connected client is Stellarium."""
        if self.is_stellarium:
            return ra, dec
        return precess_coordinates(ra, dec, 2000.0, self.jnow_epoch)

    def convert_to_j2000(self, ra: float, dec: float) -> Tuple[float, float]:
        """Converts coordinates from JNow to J2000 unless connected client is Stellarium."""
        if self.is_stellarium:
            return ra, dec
        return precess_coordinates(ra, dec, self.jnow_epoch, 2000.0)

    def set_system_time(self, dt: datetime) -> bool:
        """Sets the operating system real-time clock if permitted."""
        ts_secs = dt.timestamp()
        try:
            time.clock_settime(time.CLOCK_REALTIME, ts_secs)
            logging.info(f"System clock updated via clock_settime to {dt.isoformat()}")
            return True
        except (PermissionError, OSError):
            pass

        try:
            # Fallback to date command if clock_settime fails
            dt_formatted = dt.strftime("%Y-%m-%d %H:%M:%S")
            subprocess.run(["date", "-s", dt_formatted], check=True, capture_output=True)
            logging.info(f"System clock updated via date command to {dt_formatted}")
            return True
        except Exception as e:
            logging.warning(f"Unable to set system clock (requires root/CAP_SYS_TIME): {e}")
            return False

    def update_site_and_start_imu_if_ready(self) -> None:
        """Sets observer location on solver and starts IMU once both latitude and longitude are known."""
        if self.site_lat is not None and self.site_lon is not None:
            self.solver.set_observer_location(self.site_lat, self.site_lon)
            if not self.imu_started:
                try:
                    res = self.solver.start_imu()
                    self.imu_started = True
                    cal_status = self.solver.get_calibration_status()
                    is_cal = cal_status is not None
                    motion = self.solver.get_motion_state() or "Initializing"
                    logging.info(
                        f"Started hardware IMU after observer location set: {res} "
                        f"(Motion: {motion}, Calibrated: {is_cal})"
                    )
                    if not is_cal:
                        logging.info(
                            "[IMU] Status: Uncalibrated. Perform 3 distinct slews (e.g. pan azimuth, tilt altitude) "
                            "with steady plate-solves between movements to establish mount SVD calibration."
                        )
                except Exception as e:
                    logging.warning(f"Could not start hardware IMU: {e}. Running in solver-only mode.")

    def process_client_data(self, data: bytes) -> bytes:
        """
        Consumes streaming bytes, extracts commands delimited by '#',
        handles special handshakes (0x06, $$$), and produces responses.
        """
        response = bytearray()
        for b in data:
            self.client_buffer.append(b)

            # Stellarium ACK command
            if len(self.client_buffer) == 1 and self.client_buffer[0] == 0x06:
                self.is_stellarium = True
                response.extend(b"A")
                self.client_buffer.clear()
                continue

            # SkySafari initialization sequence
            if self.client_buffer == b"$$$":
                self.client_buffer.clear()
                continue

            # Guard against buffer overflow from invalid input
            if len(self.client_buffer) > 1024:
                self.client_buffer.clear()
                continue

            # End of LX200 command packet
            if b == ord("#"):
                if len(self.client_buffer) > 1:
                    raw_str = self.client_buffer.decode("latin1", errors="replace")
                    res_str = self.process_input(raw_str)
                    if res_str:
                        response.extend(res_str.encode("latin1"))
                self.client_buffer.clear()

        return bytes(response)

    def extract_command(self, s: str) -> Optional[str]:
        """Extracts the uppercase command name prefix after the leading colon."""
        colon_idx = s.find(":")
        if colon_idx == -1:
            return None
        s = s[colon_idx:]

        for idx, c in enumerate(s[1:], start=1):
            if not c.isalpha():
                if idx == 1:
                    return None
                return s[1:idx]
        return s[1:]

    def process_input(self, in_data: str) -> Optional[str]:
        """Dispatches an LX200 command string to the corresponding handler."""
        colon_idx = in_data.find(":")
        if colon_idx == -1:
            return None
        cmd_slice = in_data[colon_idx:].rstrip("#")
        command = self.extract_command(cmd_slice)

        if not command:
            return None

        if command == "GR":
            return self.get_ra()
        elif command == "GD":
            return self.get_dec()
        elif command == "CM":
            return self.sync()
        elif command == "MS":
            return self.slew()
        elif command == "Q":
            self.abort()
            return None
        elif command == "D":
            return self.get_distance_bars()
        elif command == "GC":
            return self.get_date()
        elif command == "GL":
            return self.get_time()
        elif command == "GG":
            return self.get_hours_to_utc()
        elif command == "Gg":
            return self.get_longitude()
        elif command == "Gt":
            return self.get_latitude()
        elif command == "GVD":
            return "Nov 14 2025#"
        elif command == "GVN":
            return "01.0#"
        elif command == "GVP":
            return "Lx200Server#"
        elif command == "GVT":
            return "23:00:00#"
        elif command == "GW":
            return "AT1"
        elif command == "Sr":
            return self.set_target_ra(cmd_slice)
        elif command == "Sd":
            return self.set_target_dec(cmd_slice)
        elif command == "St":
            return self.set_latitude(cmd_slice)
        elif command == "Sg":
            return self.set_longitude(cmd_slice)
        elif command == "SG":
            return self.set_hours_to_utc(cmd_slice)
        elif command == "SL":
            return self.set_time(cmd_slice)
        elif command == "SC":
            return self.set_date(cmd_slice)
        elif command == "Sw":
            return "1"
        elif command in ("Me", "Mn", "Ms", "Mw", "Qe", "Qn", "Qs", "Qw", "RS", "U"):
            return None
        else:
            logging.debug(f"Unrecognized LX200 command: {cmd_slice}")
            return None

    def get_ra(self) -> str:
        """
        Returns current telescope RA in JNow, snapping the entire position
        so the next :GD# call returns Dec from the exact same snapshot.
        Logs the plate solving status, pipeline time, and IMU context (rate-limited to 1 Hz).
        """
        try:
            pos = self.solver.get_latest_position()
            tel_ra = pos["ra"]
            tel_dec = pos["dec"]
            source = pos.get("source", "Unknown")
            pos_ts = pos.get("timestamp", time.time())
        except Exception:
            tel_ra = 0.0
            tel_dec = 0.0
            source = "None"
            pos_ts = 0.0

        ra_jnow, dec_jnow = self.convert_to_jnow(tel_ra, tel_dec)
        self.snapshot_dec_jnow = dec_jnow

        self.log_solver_and_imu_status(source, pos_ts)

        h, m, s = to_hms((ra_jnow / 15.0) % 24.0)
        return f"{h:02d}:{m:02d}:{s:02d}#"

    def get_dec(self) -> str:
        """
        Returns telescope Declination matching the snapshot taken during the preceding :GR# call.
        Clears the snapshot once consumed.
        """
        if self.snapshot_dec_jnow is not None:
            dec_jnow = self.snapshot_dec_jnow
            self.snapshot_dec_jnow = None
        else:
            try:
                pos = self.solver.get_latest_position()
                tel_ra = pos["ra"]
                tel_dec = pos["dec"]
            except Exception:
                tel_ra = 0.0
                tel_dec = 0.0
            _, dec_jnow = self.convert_to_jnow(tel_ra, tel_dec)

        h, m, s = to_hms(dec_jnow)
        sign = "-" if dec_jnow < 0.0 and (h != 0 or m != 0 or s != 0) else "+"
        return f"{sign}{abs(h):02d}*{m:02d}'{s:02d}#"

    def log_solver_and_imu_status(self, current_source: str, pos_ts: float) -> None:
        """
        Logs plate-solving status, pipeline latency, and IMU tracking state upon :GR# query.
        Rate-limited to 1 Hz to prevent console flooding from fast planetarium polling.
        """
        cal_status = self.solver.get_calibration_status()
        is_calibrated = cal_status is not None

        # Instant logging for calibration state transitions (bypasses rate limiter)
        if is_calibrated and not self.last_is_calibrated:
            err = cal_status.get("transform_error_fraction", 0.0)
            view_axis = cal_status.get("camera_view_gyro_axis")
            view_mis = cal_status.get("camera_view_misalignment", 0.0)
            up_axis = cal_status.get("camera_up_gyro_axis")
            up_mis = cal_status.get("camera_up_misalignment", 0.0)
            logging.info(
                f"[IMU] Calibration established! Error: {err:.4f}, "
                f"View axis: {view_axis} (misalign: {view_mis:.1f}°), "
                f"Up axis: {up_axis} (misalign: {up_mis:.1f}°)"
            )
        elif not is_calibrated and self.last_is_calibrated:
            logging.info("[IMU] Calibration lost / reset")
        self.last_is_calibrated = is_calibrated

        # Rate limit recurring status output to 1 Hz so 10 Hz queries don't spam console
        now = time.time()
        if now - self.last_gr_log_time < 1.0:
            return
        self.last_gr_log_time = now

        dt_ms, status, matches, ra, dec, roll, pipeline_ts = self.pipeline_tracker.snapshot()
        current_motion = self.solver.get_motion_state() or "Off"
        cal_str = "Calibrated" if is_calibrated else "Uncalibrated"
        
        imu_warn = " [STALE POS]" if (pos_ts > 0 and now - pos_ts > 3.0) else ""
        pipe_warn = " [STALE PIPELINE]" if (pipeline_ts > 0 and now - pipeline_ts > 3.0) else ""

        imu_info = f" | [IMU] Motion: {current_motion}, {cal_str}, Source: {current_source}{imu_warn}"

        if status == "MatchFound" and ra is not None:
            logging.info(
                f"[Solver] Match: RA={ra:.4f}°, Dec={dec:.4f}°, Roll={roll:.4f}°, "
                f"Matches={matches}, Time={dt_ms:.1f}ms{pipe_warn}{imu_info}"
            )
        else:
            logging.info(f"[Solver] No match ({status}), Time={dt_ms:.1f}ms{pipe_warn}{imu_info}")


    def sync(self) -> str:
        """Canned sync acknowledgment string expected by LX200 planetarium software."""
        if self.target_ra is not None and self.target_dec is not None:
            logging.info(f"[LX200] Sync command: RA={self.target_ra:.4f}°, Dec={self.target_dec:.4f}°")
        else:
            logging.info("[LX200] Sync command received")
        self.target_ra = None
        self.target_dec = None
        return " M31 EX GAL MAG 3.5 SZ178.0'#"

    def slew(self) -> str:
        """Simulates slew start if target coordinates are defined."""
        if self.target_ra is not None and self.target_dec is not None:
            self.is_slewing = True
            logging.info(f"[LX200] Slew command: RA={self.target_ra:.4f}°, Dec={self.target_dec:.4f}°")
            self.target_ra = None
            self.target_dec = None
            return "0"
        logging.warning("[LX200] Slew command received but no target coordinates set")
        return "1No object#"

    def abort(self) -> None:
        """Stops any active slew."""
        logging.info("[LX200] Stop / abort slew command")
        self.is_slewing = False

    def get_distance_bars(self) -> str:
        """Returns distance indicator if slewing, or empty indicator when stopped."""
        return "\x7f#" if self.is_slewing else "#"

    def set_target_ra(self, cmd: str) -> str:
        """Parses target RA command :SrHH:MM:SS."""
        if len(cmd) < 11:
            return "0"
        hours = parse_coordinates(cmd[3:5], cmd[6:8], cmd[9:11])
        if hours is not None:
            self.target_ra = hours * 15.0
            logging.info(f"[LX200] Set target RA: {cmd[3:11]} ({self.target_ra:.4f}°)")
            return "1"
        return "0"

    def set_target_dec(self, cmd: str) -> str:
        """Parses target Dec command :SdsDD*MM:SS or :SdsDD*MM'SS."""
        if len(cmd) < 12:
            return "0"
        deg = parse_coordinates(cmd[3:6], cmd[7:9], cmd[10:12])
        if deg is not None:
            self.target_dec = deg
            logging.info(f"[LX200] Set target Dec: {cmd[3:12]} ({self.target_dec:.4f}°)")
            return "1"
        return "0"

    def set_latitude(self, cmd: str) -> str:
        """Parses site latitude :StsDD:MM."""
        if len(cmd) < 9:
            return "0"
        loc = parse_location(cmd[3:6], cmd[7:9])
        if loc is not None:
            self.latitude_str = cmd[3:9]
            self.site_lat = loc
            logging.info(f"Configured observer latitude: {loc:.4f}°")
            self.update_site_and_start_imu_if_ready()
            return "1"
        return "0"

    def set_longitude(self, cmd: str) -> str:
        """Parses site longitude :SgDDD:MM (degrees West of Prime Meridian)."""
        if len(cmd) < 9:
            return "0"
        loc = parse_location(cmd[3:6], cmd[7:9])
        if loc is not None:
            self.longitude_str = cmd[3:9]
            # Convert degrees West (0-360) to signed East (+/-180)
            self.site_lon = 360.0 - loc if loc > 180.0 else -loc
            logging.info(f"Configured observer longitude: {self.site_lon:.4f}°")
            self.update_site_and_start_imu_if_ready()
            return "1"
        return "0"

    def set_hours_to_utc(self, cmd: str) -> str:
        """Parses UTC offset command :SGsHH.H (sign inverted per LX200 standard)."""
        if len(cmd) < 8:
            return "0"
        sign = "+" if cmd[3] == "-" else "-"
        hours = cmd[4:6]
        fraction = cmd[6:8]
        mins_map = {".0": "00", ".2": "15", ".5": "30", ".8": "45"}
        if fraction not in mins_map:
            return "0"
        self.pending_timezone = f"{sign}{hours}{mins_map[fraction]}"
        return "1"

    def set_time(self, cmd: str) -> str:
        """Parses local time command :SLHH:MM:SS."""
        if len(cmd) < 11:
            return "0"
        self.pending_time = cmd[3:11]
        return "1"

    def set_date(self, cmd: str) -> str:
        """
        Parses date command :SCMM/DD/YY.
        Combines with pending time and timezone to update controller time and system clock.
        """
        if len(cmd) < 11:
            return "0"
        date_part = cmd[3:11]

        time_part = self.pending_time or datetime.now().strftime("%H:%M:%S")
        tz_part = self.pending_timezone or datetime.now().astimezone().strftime("%z")

        self.pending_time = None
        self.pending_timezone = None

        dt_str = f"{tz_part}{time_part}{date_part}"
        try:
            parsed_dt = datetime.strptime(dt_str, "%z%H:%M:%S%m/%d/%y")
            self.datetime = parsed_dt
            self.jnow_epoch = round((parsed_dt.year + (parsed_dt.timetuple().tm_yday - 1) / 365.0) * 10.0) / 10.0
            logging.info(f"Updated controller date/time to {parsed_dt.isoformat()}, JNow={self.jnow_epoch}")
            self.set_system_time(parsed_dt)
            return "1Updating Planetary Data# #"
        except Exception as e:
            logging.warning(f"Failed to parse date/time string '{dt_str}': {e}")
            return "0"

    def get_hours_to_utc(self) -> str:
        """Formats the UTC offset with inverted sign."""
        tz_str = self.datetime.astimezone().strftime("%z")
        if not tz_str or len(tz_str) < 5:
            return "+00.0#"
        sign = "+" if tz_str[0] == "-" else "-"
        hours = tz_str[1:3]
        mins = tz_str[3:5]
        partial = {
            "15": ".2",
            "30": ".5",
            "45": ".8",
        }.get(mins, ".0")
        return f"{sign}{hours}{partial}#"

    def get_time(self) -> str:
        """Formats current local time."""
        return self.datetime.strftime("%H:%M:%S#")

    def get_date(self) -> str:
        """Formats current local date."""
        return self.datetime.strftime("%m/%d/%y#")

    def get_latitude(self) -> str:
        """Returns stored latitude string."""
        return f"{self.latitude_str}#"

    def get_longitude(self) -> str:
        """Returns stored longitude string."""
        return f"{self.longitude_str}#"


def lx200_server_worker(
    solver: olive_solve.FusedSolver,
    pipeline_tracker: PipelineTracker,
    host: str,
    port: int,
    stop_event: threading.Event,
) -> None:
    """
    Worker thread hosting the single-client LX200 TCP server on port 4030.
    """
    controller = Lx200Controller(solver, pipeline_tracker)
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.settimeout(1.0)

    try:
        server_sock.bind((host, port))
        server_sock.listen(1)
        logging.info(f"LX200 server listening on {host}:{port}")
    except Exception as e:
        logging.error(f"Failed to bind LX200 server on {host}:{port}: {e}")
        return

    while not stop_event.is_set():
        try:
            client_sock, client_addr = server_sock.accept()
        except socket.timeout:
            continue
        except Exception as e:
            if not stop_event.is_set():
                logging.warning(f"Error accepting connection: {e}")
            break

        client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        client_sock.settimeout(1.0)

        while not stop_event.is_set():
            try:
                data = client_sock.recv(512)
                if not data:
                    break

                response = controller.process_client_data(data)
                if response:
                    client_sock.sendall(response)
            except socket.timeout:
                continue
            except Exception:
                break

        try:
            client_sock.close()
        except Exception:
            pass

    try:
        server_sock.close()
    except Exception:
        pass
    logging.info("LX200 server thread stopped.")


# ==========================================
# CAMERA CAPTURE & SOLVER PIPELINE
# ==========================================

def initialize_camera(
    picam2: Picamera2,
    raw_format: str,
    exposure_ms: float,
) -> Tuple[int, int]:
    """Configures primary camera in raw stream mode with max gain."""
    modes = picam2.sensor_modes
    if modes:
        best_mode = max(modes, key=lambda m: m["size"][0] * m["size"][1])
        width, height = best_mode["size"]
    else:
        width, height = 1920, 1080

    logging.info(f"Configuring camera in raw mode: format={raw_format}, size={width}x{height}")
    cfg = picam2.create_video_configuration(raw={"format": raw_format, "size": (width, height)})
    picam2.configure(cfg)

    controls = picam2.camera_controls
    max_gain = 16.0
    if "AnalogueGain" in controls:
        max_gain = controls["AnalogueGain"][1]
    logging.info(f"Discovered max analogue gain: {max_gain}")

    exposure_us = int(exposure_ms * 1000.0)
    logging.info(f"Setting fixed exposure: {exposure_ms:.1f} ms ({exposure_us} us)")

    picam2.set_controls({
        "AeEnable": False,
        "AnalogueGain": float(max_gain),
        "ExposureTime": exposure_us,
        "FrameDurationLimits": (exposure_us, max(exposure_us, 1_000_000)),
    })

    picam2.start()
    return height, width


def capture_and_solve_worker(
    picam2: Picamera2,
    solver: olive_solve.FusedSolver,
    pipeline_tracker: PipelineTracker,
    height: int,
    width: int,
    downsample: int,
    bg_sub_mode: str,
    stop_event: threading.Event,
) -> None:
    """
    Worker thread executing the camera exposure read and celestial plate solve.
    Drains stale frames, extracts 8-bit MSBs, and runs the fast solver pipeline.
    """
    logging.info("Starting camera capture and plate solve worker thread...")
    # Preallocate contiguous 8-bit frame buffer to avoid memory allocations in hot loop
    raw_8bit = np.empty((height, width), dtype=np.uint8)

    while not stop_event.is_set():
        try:
            # Drain stale frames and capture the freshest exposure
            req = picam2.capture_request(flush=True)
            if req is None:
                logging.warning("[Camera] Capture request failed (no frame returned)")
                continue

            arr = req.make_array("raw")
            if arr is None or arr.size == 0:
                logging.warning("[Camera] Capture request returned empty frame array")
                req.release()
                continue

            # Optimized unpacking for 10-bit raw sensor data (e.g. SRGGB10)
            if arr.ndim == 2 and arr.shape[1] == width * 2:
                arr_u16 = arr.view(np.uint16)
                np.right_shift(arr_u16, 2, out=arr_u16)
                np.copyto(raw_8bit, arr[:, 0::2])
            elif arr.ndim == 2 and arr.shape == (height, width):
                if arr.dtype == np.uint16:
                    np.right_shift(arr, 2, out=arr)
                np.copyto(raw_8bit, arr)
            else:
                np.copyto(raw_8bit, arr[:height, :width])

            # Release camera buffer immediately back to the pool
            req.release()

            t_start = time.perf_counter()

            # Execute fast centroid extraction and plate solve
            try:
                solve_result = solver.solve_from_image_fast(
                    raw_8bit,
                    bg_sub_mode=bg_sub_mode,
                    downsample=downsample,
                )
            except (AttributeError, TypeError):
                # Fallback path if solve_from_image_fast wrapper is not directly present
                centroids = solver.get_centroids_from_image_fast(
                    raw_8bit,
                    bg_sub_mode=bg_sub_mode,
                    downsample=downsample,
                )
                solve_result = solver.solve_from_centroids(
                    centroids.astype(np.float64),
                    (float(height), float(width)),
                )

            dt_ms = (time.perf_counter() - t_start) * 1000.0
            pipeline_tracker.update(dt_ms, solve_result)

        except Exception as e:
            if not stop_event.is_set():
                logging.error(f"[Solver] Pipeline error: {e}", exc_info=True)
                time.sleep(0.05)


# ==========================================
# CLI & MAIN ENTRYPOINT
# ==========================================

def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Standalone Fused Plate Solver with LX200 Server on port 4030."
    )
    parser.add_argument(
        "--exposure",
        "--exposure-ms",
        dest="exposure_ms",
        type=float,
        default=50.0,
        help="Fixed camera exposure time in milliseconds (default: 50.0 ms)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4030,
        help="TCP port to host LX200 telescope server (default: 4030)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind address for LX200 server (default: 0.0.0.0)",
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
        help="Downsampling factor for fast extractor (default: 2)",
    )
    parser.add_argument(
        "--bg-sub-mode",
        type=str,
        default="global_median",
        help="Background subtraction mode (default: global_median)",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_arguments()

    logging.info("==================================================")
    logging.info(" Fused Plate Solver & LX200 Server                ")
    logging.info("==================================================")
    logging.info(f"Fixed Exposure   : {args.exposure_ms:.1f} ms")
    logging.info(f"LX200 Server Port: {args.port}")
    logging.info(f"Raw Format       : {args.raw_format}")
    logging.info(f"Downsampling     : {args.downsample}x")
    logging.info(f"BG Subtraction   : {args.bg_sub_mode}")

    db_path = resolve_database_path(args.database_path)
    logging.info(f"Star Database    : {db_path}")

    # Initialize unified FusedSolver with auto IMU probing
    solver = olive_solve.FusedSolver(db_path, imu_type="auto")

    # Initialize Picamera2 in raw sensor mode
    picam2 = Picamera2()
    try:
        height, width = initialize_camera(picam2, args.raw_format, args.exposure_ms)
    except Exception as e:
        logging.error(f"Failed to initialize camera: {e}")
        picam2.close()
        sys.exit(1)

    stop_event = threading.Event()

    def handle_signal(sig, frame):
        logging.info("Termination signal received. Shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    pipeline_tracker = PipelineTracker()

    # Thread 1: Camera exposure reading and plate solving
    solve_thread = threading.Thread(
        target=capture_and_solve_worker,
        args=(picam2, solver, pipeline_tracker, height, width, args.downsample, args.bg_sub_mode, stop_event),
        name="CaptureSolveThread",
        daemon=True,
    )

    # Thread 2: LX200 TCP server
    lx200_thread = threading.Thread(
        target=lx200_server_worker,
        args=(solver, pipeline_tracker, args.host, args.port, stop_event),
        name="Lx200ServerThread",
        daemon=True,
    )

    solve_thread.start()
    lx200_thread.start()

    try:
        while not stop_event.is_set():
            if not solve_thread.is_alive():
                logging.error("Camera solver thread died unexpectedly. Shutting down...")
                stop_event.set()
                break
            if not lx200_thread.is_alive():
                logging.error("LX200 server thread died unexpectedly. Shutting down...")
                stop_event.set()
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received.")
        stop_event.set()

    logging.info("Cleaning up resources...")
    solve_thread.join(timeout=3.0)
    lx200_thread.join(timeout=2.0)

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

    logging.info("Terminated cleanly.")


if __name__ == "__main__":
    main()
