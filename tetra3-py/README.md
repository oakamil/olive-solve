# tetra3-py

Python bindings for the heavily optimized Rust port of the Tetra3 star tracker and plate solver. 

This package provides a zero-copy, natively compiled Python interface to the Rust `tetra3` engine. By leveraging PyO3's Buffer Protocol, image data is passed directly from NumPy arrays into Rust without any memory duplication, ensuring maximum performance.

## Prerequisites

- A working Rust toolchain (install via [rustup](https://rustup.rs/))
- Python 3.8+
- `maturin` for building the Python wheel (`pip install maturin`)

## Building & Installation

To build the wheel for your current environment, run:

```bash
maturin build --release
```

The compiled `.whl` file will be placed in the `target/wheels/` directory, which you can then install using `pip`:

```bash
pip install target/wheels/tetra3_py-*.whl
```

---

## API Reference

### `Tetra3(database_path)`
Creates a new Tetra3 instance. The underlying star database is lazy-loaded, meaning it won't hit the disk until the first plate-solving operation is executed.

* **`database_path`** *(str)*: Path to the `.npz` catalog database.

---

### `get_centroids_from_image(image, **kwargs)`
Extracts star centroids from a 2D NumPy array using zero-copy memory access.

**Arguments:**
* **`image`** *(numpy.ndarray)*: A 2D array of `float32` representing the image.
* **`**kwargs`**: Extraction options:
  * `sigma` *(float)*
  * `image_th` *(float)*
  * `downsample` *(int)*
  * `filtsize` *(int)*
  * `binary_open` *(bool)*
  * `centroid_window` *(int)*
  * `min_area` / `max_area` *(int)*
  * `min_sum` / `max_sum` *(float)*
  * `max_axis_ratio` *(float)*
  * `max_returned` *(int)*
  * `bg_sub_mode` *(str)*: `"local_median"`, `"local_mean"`, `"global_median"`, `"global_mean"`, or `"none"`
  * `sigma_mode` *(str)*: `"local_median_abs"`, `"local_root_square"`, `"global_median_abs"`, or `"global_root_square"`
  * `return_moments` *(bool)*: If `True`, returns geometric moment data.
  * `return_images` *(bool)*: If `True`, returns intermediate debug image arrays.

**Returns:**
* **Default (`return_moments=False`, `return_images=False`)**: A `numpy.ndarray` of shape `(N, 2)` containing `[y, x]` coordinates of the found centroids.
* **With `return_moments=True`**: A tuple of arrays: `(centroids, sums, areas, moments, ratios)` where `moments` is an `(N, 3)` array of xx, yy, and xy second moments.
* **With `return_images=True`**: A tuple containing `(core_results, dict_of_images)`. 
  * `core_results` is the standard data payload that would have been returned if `return_images` was `False` (i.e., either the simple `(N, 2)` numpy array of centroids, or the full tuple of moment arrays if `return_moments=True`). 
  * `dict_of_images` is a dictionary containing the keys `"cropped_and_downsampled"`, `"removed_background"`, and `"binary_mask"`.

---

### `get_centroids_from_image_fast(image, **kwargs)`
Extracts star centroids using the zero-allocation fast extraction path. Accepts either `uint8` or `float32` 2D NumPy arrays.

**Arguments:**
* **`image`** *(numpy.ndarray)*: A 2D array (`uint8` or `float32`).
* **`**kwargs`**: Fast extraction options:
  * `sigma` *(float)*: Multiplier for background noise threshold.
  * `downsample` *(int)*: Downsampling factor (`1`, `2`, or `4`).
  * `bg_sub_mode` *(str)*: `"global_mean"`, `"global_median"`, `"block_median"`, or `"line_median"`.
  * `filtsize` *(int)*: Block size when using `"block_median"`.
  * `binary_open` *(bool)*: Whether to apply binary morphological opening filter.
  * `min_area` / `max_area` *(int)*: Minimum/maximum component pixel area.
  * `virtual_crops` *(list)*: Optional list of crop dictionaries for multi-region extraction.

**Returns:**
* **`numpy.ndarray`**: `(N, 2)` array containing `[y, x]` centroid coordinates, or a tuple containing base centroids and virtual crop centroids if `virtual_crops` were specified.

---

### `solve_from_centroids(centroids, size, **kwargs)`
Runs plate solving from pre-extracted centroids.

**Arguments:**
* **`centroids`** *(numpy.ndarray)*: A 2D array of `float64` representing the `[y, x]` centroid positions.
* **`size`** *(tuple)*: A tuple of `(height, width)` as floats, representing the original image dimensions.
* **`**kwargs`**: Solver configuration options:
  * `fov_estimate` *(float)*
  * `fov_max_error` *(float)*
  * `match_radius` *(float)*
  * `match_threshold` *(float)*
  * `solve_timeout` *(int)*: Maximum time in ms.
  * `distortion` *(float)*
  * `match_max_error` *(float)*
  * `return_matches` *(bool)*
  * `return_catalog` *(bool)*
  * `return_rotation_matrix` *(bool)*
  * `target_pixel` *(numpy.ndarray)*: A 2D array representing specific `[y, x]` target coordinates to solve for.
  * `target_sky_coord` *(numpy.ndarray)*: A 2D array representing specific `[RA, Dec]` targets.

**Returns:**
* **`dict`**: A dictionary containing the solution. Key outputs include:
  * `'RA'`, `'Dec'`, `'Roll'`, `'FOV'`, `'distortion'`
  * `'RMSE'`, `'P90E'`, `'MAXE'`, `'Matches'`, `'Prob'`, `'is_mirrored'`
  * `'epoch_equinox'`, `'epoch_proper_motion'`
  * `'T_solve'` *(Execution time in ms)*
  * `'status'` *(String representing the match state: e.g., 'MATCH_FOUND', 'NO_MATCH')*
  * Target mapping outputs (if requested): `'RA_target'`, `'Dec_target'`, `'y_target'`, `'x_target'`
  * Star outputs (if requested): `'matched_centroids'`, `'matched_stars'`, `'matched_catID'`, `'catalog_stars'`, `'rotation_matrix'`

---

### `solve_from_image_fast(image, **kwargs)`
Executes the fast-path extraction and immediate plate solve in an uninterrupted native execution flow. Accepts `uint8` or `float32` image arrays.

**Arguments:**
* **`image`** *(numpy.ndarray)*: A 2D array (`uint8` or `float32`).
* **`**kwargs`**: Combined configuration keys for fast extraction and solving.

**Returns:**
* **`dict`**: Solution dictionary identical to `solve_from_centroids`, with `'T_extract'` recording fast extraction time in ms.

---

### `solve_from_image(image, **kwargs)`
Runs both standard extraction and plate solving in one uninterrupted, native pipeline.

**Arguments:**
* **`image`** *(numpy.ndarray)*: A 2D array of `float32`.
* **`**kwargs`**: Accepts combined configuration keys for both extraction and solving operations (as detailed in the functions above).

**Returns:**
* **`dict`**: The exact same solution dictionary as `solve_from_centroids`, with the addition of the `'T_extract'` key tracking extraction time in milliseconds.

---

*Copyright (c) 2026 Omair Kamil. See the root LICENSE file for terms.*
