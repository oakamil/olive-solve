#!/bin/bash
# Copyright (c) 2026 Omair Kamil
# See LICENSE file in root directory for license terms.
# Unified Benchmark Script for Tetra3 Fast Path
# This script builds the project in release mode and runs performance tests.

set -e

# 1. Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "--- Building Tetra3 in Release Mode ---"
cd "$PROJECT_ROOT"
cargo build --release -p tetra3 -p tetra3-py

# 2. Setup the Python environment in tetra3-py
echo "--- Setting up library symlink ---"
cd "$SCRIPT_DIR"
ln -sf "$PROJECT_ROOT/target/release/libtetra3_py.so" tetra3_py.so

# 3. Create a temporary python script for benchmarking
cat << 'EOF' > run_bench.py
import os
import sys
import time
import glob
import numpy as np
from PIL import Image

# Ensure we can import the local .so
sys.path.insert(0, os.getcwd())

try:
    import tetra3_py as tetra3
except ImportError as e:
    print(f"Error importing tetra3_py: {e}")
    sys.exit(1)

# Paths relative to project root (since we run from tetra3-py)
db_path = "../tetra3/tests/fixtures/default_database.npz"
img_dir = "../tetra3/tests/fixtures/sample_images"

if not os.path.exists(db_path):
    print(f"Database not found: {db_path}")
    sys.exit(1)

print(f"\n--- Tetra3 Performance Benchmark ---")
print(f"Loading database: {db_path}")
t3 = tetra3.Tetra3(db_path)

images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
if not images:
    print(f"No images found in {img_dir}")
    sys.exit(1)

# Warm-up
print("Warming up database and caches...")
img_warm = Image.open(images[0]).convert('L')
arr_f32 = np.asarray(img_warm, dtype=np.float32)
t3.solve_from_image(arr_f32) 

print("\n--- RESULTS (Extraction + Solve) ---")
header = f"{'Filename':<40} | {'Standard (ms)':<12} | {'Fast f32 (ms)':<12} | {'Speedup (x)':<10} | {'Fast u8 (ms)':<12} | {'Speedup (x)':<10}"
print(header)
print("-" * len(header))

for img_path in images:
    fname = os.path.basename(img_path)
    img = Image.open(img_path).convert('L')
    arr_f32 = np.asarray(img, dtype=np.float32)
    arr_u8 = np.asarray(img, dtype=np.uint8)

    ITERATIONS = 100

    # Standard Path
    t0 = time.perf_counter()
    for _ in range(ITERATIONS):
        t3.get_centroids_from_image(arr_f32, bg_sub_mode='global_median', sigma_mode='global_median_abs')
    t_std = ((time.perf_counter() - t0) / ITERATIONS) * 1000

    # Fast Path (f32)
    t0 = time.perf_counter()
    for _ in range(ITERATIONS):
        t3.get_centroids_from_image_fast(arr_f32, bg_sub_mode='global_median', sigma_mode='global_median_abs')
    t_f32 = ((time.perf_counter() - t0) / ITERATIONS) * 1000

    # Fast Path (u8)
    t0 = time.perf_counter()
    for _ in range(ITERATIONS):
        t3.get_centroids_from_image_fast(arr_u8, bg_sub_mode='global_median', sigma_mode='global_median_abs')
    t_u8 = ((time.perf_counter() - t0) / ITERATIONS) * 1000

    speedup_f32 = t_std / t_f32 if t_f32 > 0 else 0
    speedup_u8 = t_std / t_u8 if t_u8 > 0 else 0
    print(f"{fname:<40.40} | {t_std:<12.2f} | {t_f32:<12.2f} | {speedup_f32:<10.1f} | {t_u8:<12.2f} | {speedup_u8:<10.1f}")

# Cache Test
print("\n--- CACHE COLD VS WARM (u8) ---")
img_path = images[0]
arr_u8 = np.asarray(Image.open(img_path).convert('L'), dtype=np.uint8)
t3.get_centroids_from_image_fast(arr_u8, sigma=5.0) # Clear/change cache state
t0 = time.perf_counter()
t3.get_centroids_from_image_fast(arr_u8, sigma=2.0)
t_cold = (time.perf_counter() - t0) * 1000
t0 = time.perf_counter()
t3.get_centroids_from_image_fast(arr_u8, sigma=2.0)
t_warm = (time.perf_counter() - t0) * 1000
print(f"Cold call: {t_cold:.2f} ms")
print(f"Warm call: {t_warm:.2f} ms")

EOF

# 4. Run the benchmark
python3 run_bench.py

# 5. Cleanup temporary python script
rm run_bench.py
echo -e "\nBenchmark complete."
