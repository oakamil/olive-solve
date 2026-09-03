# Olive Solve - Enhanced Tetra3 Solver in Rust

A fast Rust implementation and optimization of the [cedar-solve](https://github.com/smroid/cedar-solve) centroid extraction and plate solving algorithms, with new features. 

## Unique Features

This project is not just a straight port of the upstream Python logic. It introduces a number of new features:

* A complete IMU implementation for tracking movement between camera plate solves.
* A full-featured FusedSolver that integrates between image plate-solves and IMU sensors.
* Additional background extraction modes that optimize between accuracy and efficiency.
* Efficient "virtual" cropping of images during extraction, enabling clients to process different parts of an image to avoid obstructions.
* Multi-pass centroid extraction with shared background calculations and global background level estimation.
* New solver options to reject below horizon pattern stars and final matches.
* Solver enhancement to return the best low-confidence match when the match threshold isn't met.

### Extractor

* **Optimized `u8` Pipelines**: Highly optimized processing pipelines tailored specifically for 8-bit grayscale images, minimizing memory overhead and bandwidth.
* **Fast Extractor Implementation**: A zero-allocation, pre-allocated pipeline tailored for embedded microprocessors. Uses fixed-point/integer arithmetic, hardware downsampling, and custom background subtraction modes to maximize single-threaded throughput without multithreading overhead or runtime memory allocations.
* **Hybrid Background Subtraction Modes**: Includes custom `Line Median` and `Block Median` background subtraction modes. These act as high-performance compromises between the fast (but less accurate) `Global Median` and the highly accurate (but computationally expensive) `Local Median` modes. *Note: `Line Median` is specifically designed to excel at handling cameras that exhibit horizontal banding noise.*
* **Virtual Crops**: Calculates the centroids for sections of an image in addition to the full image.
* **Multi-Variant Batching**: Supports evaluating multiple parameter combinations over a single image by reusing the initial background subtraction and noise matrices.
* **Background Level Output**: Calculates a normalized global background brightness value for downstream exposure/gain control.

### Solver

* **Database Support**: Supports both `tetra3` and `cedar-solve` database formats.
* **Performance**: Incredibly fast single-threaded performance - centroids generated from clean images typically solve in under 0.25ms on a Raspberry Pi Zero 2W.
* **Centroid Matcher**: Given a plate solution and a list of centroids, determines the valid set of centroids that match stars in the database.
* **Horizon Rejection**: When provided with observer latitude and LST, reject below-horizon stars and final matches.
* **Low-confidence Matches**: Optionally return the best potential match that doesn't meet the configured match threshold.

### Olive IMU

A robust integration framework for inertial measurement units.

* **Supported Sensors**: Integration with the Bosch BMI160, Ceva BNO055 and BNO085, and TDK InvenSense MPU series sensors is included. The ImuDevice trait can be implemented for other sensors.
* **Real-time SVD Alignment**: Implements continuous Singular Value Decomposition to mathematically derive the optimal transformation matrix,
  keeping the camera and IMU reference frames synchronized.
* **Continuous Bias Compensation**: Uses rolling variance windows and an exponential moving average to actively calculate and eliminate zero-
  rate gyroscope drift.
* **Timing Synchronization**: Employs queue-draining and back-dating timestamp strategies to map sensor-relative measurements to host wall-clock time, absorbing I2C jitter and preventing timeline drift.
* **Asynchronous Polling**: A dedicated OS thread handles blocking I2C transactions to maintain high polling rates (100 Hz+) without stalling the main `tokio` async runtime.
* **Sensor Telemetry**: Exposes real-time hardware telemetry (gyro, gravity vectors, and integrated quaternions if available) via the `FusedSolver`.

### Fused Solver

The `FusedSolver` provides an integrated architecture that bridges camera plate-solving and high-speed IMU telemetry.

* **Continuous Orientation Tracking**: By coupling sparse, high-accuracy plate solves with dense, real-time IMU gyroscope data, the solver maintains orientation between camera exposures and during large slews.
* **Batch Processing**: `get_centroids_from_image_variants` and `solve_from_centroid_batch` allow executing multiple extraction/solving variants over a single image efficiently.
* **Low-Confidence Fallback**: When exact matches are constrained, the solver includes fallback heuristics to verify and accept motion-correlated low-confidence matches.

## Repository Structure

This workspace includes the following crates:

* **`tetra3`**: The core algorithms. `solver.rs` is a Rust port of the [Tetra3](https://github.com/smroid/cedar-solve/blob/master/tetra3/tetra3.py) `solve_from_centroids` function. `extractor.rs` is a Rust port of the `get_centroids_from_image` function. `tetra3.rs` provides the standard interface corresponding to the Python project.
* **`tetra3-py`**: Python bindings for the optimized tetra3 Rust implementation.
* **`server`**: A gRPC server that exposes tetra3's algorithms as a service.
* **`olive-imu`**: A specialized IMU driver library providing real-time kinematic integration for telescope orientation.

The top-level crate exposes the FusedSolver integrated API, including Python bindings. The integrated API can be used with or without an IMU present.

## Getting Started

### Prerequisites
* [Rust / Cargo](https://rustup.rs/)
* Python 3

### Building
To build the workspace:

```bash
cargo build --release
```

#### Python Bindings (`olive-solve`)

To build the integrated `FusedSolver` library with Python bindings:

```bash
cargo build --release --features python
# Verify API functionality
python3 test_python_api.py
```

#### Embedded Platforms & 32-Bit Solver

For resource-constrained embedded targets (such as Raspberry Pi Zero 1 ARMv6 or Rockchip RV1103 ARMv7), enable the `force-32bit-solver` feature:

```bash
cargo build --release --features force-32bit-solver
```

Pre-configured cross-compilation targets (`arm-unknown-linux-gnueabihf`, `armv7-unknown-linux-gnueabihf`, and `aarch64-unknown-linux-gnu`) are provided in `.cargo/config.toml`, and cross-compilation wheel packaging is automated via `tetra3-py/build_cross_wheels.sh`.

### Running Services and Tools

#### gRPC Service (`server`)

The `server` crate provides a high-throughput gRPC interface exposing both star extraction and plate solving:

```bash
cargo run --release -p server -- --database-path <path/to/database.npz> --port 50051
```

#### IMU Sensor Diagnostics (`olive-imu`)

The `olive-imu` crate includes a CLI diagnostic utility for evaluating hardware sensor drift and zero-rate bias compensation:

```bash
cargo run --release -p olive-imu --example test_bias -- --imu bmi160 --duration 10
```

### Testing

A set of real-world test data is provided for validating the algorithms and the wrappers.

#### Validation Tests

From the project root, run:

```bash
cargo test --release -- --test-threads=1
```

Optionally add `--nocapture` to the end of the command above to print the full test output to `stdout`.

#### Tests for Python Bindings

From the `tetra3-py` root, run:

```bash
./test_python_wrapper.sh
```

To test the integrated `olive-solve` Python wrapper from the workspace root:

```bash
cargo build --release --features python
python3 test_python_api.py
```

#### Performance Tests

The solver tests provide a performance report at the end of the output:

```bash
cargo test --release test_solver_consistency_with_testdata -- --nocapture
```

To compare the extraction performance against the original `cedar-solve` implementation:

1. Clone [cedar-solve](https://github.com/smroid/cedar-solve)
2. Run `./setup.sh` in the `cedar-solve` root.
3. Source the Python activation script:
```bash
source ../cedar-solve/.cedar_venv/bin/activate
```
4. From the repo root run:
```bash
cargo test --release test_performance_vs_python -- --nocapture --test-threads=1 --ignored
```

To compare the extraction performance against `cedar-detect`, run:

```bash
cargo test --release test_performance_vs_cedar -- --nocapture --test-threads=1 --ignored
```

## FAQ

1\. Why not port the database generation function?

Database generation is a one-time operation that doesn't benefit from a port.

2\. How can I generate an appropriate database?

Refer to [cedar-solve](https://github.com/smroid/cedar-solve/blob/master/tetra3/tetra3.py) or [esa/tetra](https://github.com/esa/tetra3/blob/master/tetra3/tetra3.py) for database generation. Note that the 2 versions of tetra3 have slightly different database formats, but this Rust implementation is compatible with both.

3\. What kind of performance gain can I expect to see for the solver?

On a Raspberry Pi 5 with 4 GB RAM the Rust version is ~130x faster than Python. On a Raspberry Pi Zero 2W with 512 MB of RAM the Rust version achieves a similar performance gain. In both cases solves in the `olive-solve` pipeline take well under 1 ms.

4\. What kind of performance gain can I expect to see for the extractor?

Benchmarks on the Raspberry Pi 5 with 4 GB RAM show ~15x improvement over the extractor in `cedar-solve`.

5\. How does the extractor port compare to `cedar-detect`?

The standard `Extractor` port reproduces the exact algorithm and math from `cedar-solve` for strict 1:1 validation, which `cedar-detect` outperforms by >2x using an alternative algorithm. However, this repository also includes the custom `FastExtractor`, which incorporates fixed-point pipelines, integer downsampling, and tiled `BlockMedian` / `LineMedian` background subtraction. `FastExtractor` has similar performance to `cedar-detect` while maintaining excellent centroid fidelity and solve rates.

## License

This project is licensed under Apache 2.0 license.

See LICENSE.md for full details.

## Disclaimer

All product names, trademarks and registered trademarks are property of their respective owners. All company, product and service names used in this website are for identification purposes only. Use of these names, trademarks and brands does not imply endorsement.

`olive-solve` is not affiliated with, endorsed by, or sponsored by Clear Skies Astro or the European Space Agency.

Cedar™ is a trademark of Clear Skies Astro, registered in the U.S. and other countries.
