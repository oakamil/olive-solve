# tetra3-py

Python bindings for the heavily optimized Rust port of the Tetra3 star tracker and plate solver. 

This package provides a zero-copy, natively compiled Python interface to the Rust `tetra3` engine. By leveraging PyO3's Buffer Protocol, image data is passed directly from NumPy arrays into Rust without any memory duplication, ensuring maximum performance.

## Prerequisites

To compile the bindings from source, you will need:
* **Rust:** (Install via [rustup](https://rustup.rs/))
* **Python:** 3.8 or newer
* **Maturin:** The build system for PyO3 (`pip install maturin`)

## Installation & Development

It is highly recommended to build and install this package inside a Python virtual environment.

### 1. Local Development (Testing)
To compile the library and immediately install it into your active virtual environment, run:
```bash
maturin develop --release --features pyo3/extension-module
```

*Note: Always use `--release` when benchmarking or plate solving. Unoptimized debug builds are significantly slower.*

### 2. Building a Wheel for Distribution

To create a standalone Python package (`.whl` file) that you can distribute to other machines with the same OS/Architecture:

```bash
maturin build --release --features pyo3/extension-module
```

The compiled wheel will be placed in `target/wheels/`. You can then install it on target machines using standard pip:

```bash
pip install target/wheels/tetra3-*.whl
```

## Quick Start

The API is designed to feel native to Python scientific computing pipelines. The engine expects a 2D float32 NumPy array.

```python
import numpy as np
from PIL import Image
import tetra3

# 1. Initialize the Tetra3 Engine
# The database is lazy-loaded and won't hit the disk until the first solve
t3 = tetra3.Tetra3("/path/to/default_database.npz")

# 2. Load an image as an 8-bit grayscale and convert to a float32 NumPy array
img = Image.open("night_sky.jpg").convert('L')
img_arr = np.asarray(img, dtype=np.float32)

# 3. Solve! (Zero-copy ingestion)
result = t3.solve_from_image(
    img_arr, 
    fov_estimate=11.4, 
    downsample=2
)

# Print results
if result.get("ra") is not None:
    print(f"Match Found!")
    print(f"RA:  {result['ra']:.4f}")
    print(f"Dec: {result['dec']:.4f}")
    print(f"Solve Time: {result['t_solve_ms']:.2f} ms")
else:
    print("No match found.")
```

## API Reference

### `Tetra3(database_path: str)`

Creates a new Tetra3 solver instance.

### `get_centroids_from_image(image: np.ndarray, **kwargs) -> dict`

Extracts stars from the image.

* **Returns:** A dictionary containing a single N x 4 NumPy array under the key `'centroids'`. The columns correspond to `[y, x, sum, area]`.

### `solve_from_image(image: np.ndarray, **kwargs) -> dict`

Runs the complete extraction and plate solving pipeline.

* **Returns:** A dictionary containing the solution: `'ra'`, `'dec'`, `'roll'`, `'fov'`, `'t_extract_ms'`, `'t_solve_ms'`, and optionally `'rotation_matrix'`.

---

*Copyright (c) 2026 Omair Kamil. See the root LICENSE file for terms.*
