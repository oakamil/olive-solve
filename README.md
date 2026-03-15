# Tetra3 Solver in Rust

A fast, robust, and async-friendly Rust implementation of the [cedar-solve](https://github.com/smroid/cedar-solve) centroid extraction and plate solving algorithms. 

## Repository Structure

This workspace is divided into two primary crates:

* **`tetra3`**: The core algorithm. `solver.rs` is a Rust port of the [Tetra3](https://github.com/smroid/cedar-solve/blob/master/tetra3/tetra3.py) `solve_from_centroids` function. `extractor.rs` is a Rust port of the `get_centroids_from_image` function.
* **`cedar-solver`**: The integration layer for the [Cedar™](https://github.com/smroid/cedar) telescope control system.

## Getting Started

### Prerequisites
* [Rust / Cargo](https://rustup.rs/) (edition 2021)
* Python 3 (Optional, for running the Python test suite)
* [cedar-server](https://github.com/smroid/cedar-server) cloned into the same location as tetra-solve-rs

### Building
To build the workspace:

```
cargo build --release
```

### Testing

A set of real-world test data is provided for validating the algorithm.

#### Python Tetra3 Solver Validation

To run the Python tests, ensure that you have the following repos cloned into the same location as tetra-solve-rs:
* [cedar-solve](https://github.com/smroid/cedar-solve)
* [tetra3_server](https://github.com/smroid/tetra3_server)

In `cedar-solve` ensure that the `setup.sh` script is run. Then run the tests against the Python solver:

```
./run_python_tets.sh
```

#### Rust Port Solver Validation

Ensure that `tetra3-server` is cloned to the same location as `tetra-solve-rs`.

```
cargo test --release tetra3_solver -- --nocapture
```

#### Rust Port Extractor Validation

Ensure that you have the following repos cloned into the same location as tetra-solve-rs:
* [cedar-solve](https://github.com/smroid/cedar-solve)

In `cedar-solve` ensure that the `setup.sh` script is run and then the environment activation script is sourced from `cedar-solve/.cedar_venv/bin/activate`. Then run:

```
cargo test --release --test validate_extractor -- --nocapture --test-threads=1
```

The above runs the sanity tests, downsampling tests, and benchmark tests. To run the full validation tests (>15 minutes on a Raspberry Pi 5):

```
cargo test --release --test validate_extractor -- --nocapture --test-threads=1 --ignored
```

## FAQ

1\. Why port only the solving and extraction functions?

Database generation is a one-time operation that doesn't benefit from a port.

2\. What kind of performance gain can I expect to see for the solver?

On a Raspberry Pi 5 with 4 GB RAM the Rust version ~130x faster. On a Raspberry Pi Zero 2W with 512 MB of RAM the Rust version has a similar performance gain. In both cases solves in the `cedar-server` pipeline take well under 1 ms.

3\. What kind of performance gain can I expect to see for the extractor?

Benchmarks on the Raspberry Pi 5 with 4 GB RAM show ~15x improvement over the extractor in `cedar-solve`.

4\. How does the extractor port compare to `cedar-detect`?

`cedar-detect` is at least 2x as fast. The extractor port here uses the same algorithm as `cedar-solve` and produces the same results. `cedar-detect` provides a custom algorithm.

## License

This project is licensed under the PolyForm Noncommercial License 1.0.0.

See LICENSE.md for full details.

## Disclaimer

All product names, trademarks and registered trademarks are property of their respective owners. All company, product and service names used in this website are for identification purposes only. Use of these names, trademarks and brands does not imply endorsement.

`tetra-solve-rs` is not affiliated with, endorsed by, or sponsored by Clear Skies Astro.

Cedar™ is a trademark of Clear Skies Astro, registered in the U.S. and other countries.
