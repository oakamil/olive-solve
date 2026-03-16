#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=============================================="
echo "  Tetra3 PyO3 Build & Test Automation Script  "
echo "=============================================="

# 1. Ensure we are in the right directory (tetra3-py)
if [ ! -f "Cargo.toml" ] || ! grep -q "name = \"tetra3-py\"" Cargo.toml; then
    echo "Error: This script must be run from inside the 'tetra3-py' directory."
    exit 1
fi

VENV_DIR=".env"

# 2. Create the Python virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating Python virtual environment in ./$VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/4] Python virtual environment already exists."
fi

# 3. Activate the environment
echo "[2/4] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# 4. Install build and test dependencies
echo "[3/4] Installing Python dependencies (maturin, numpy, pillow)..."
pip install --upgrade pip
pip install maturin numpy pillow

# 5. Build the Rust extension using maturin
echo "[4/4] Compiling Rust extension and linking to Python..."
# Build in release mode to ensure mathematical operations are optimized
maturin develop --release --features pyo3/extension-module

# 6. Run the Cargo integration test
echo "=============================================="
echo "  Running Rust/Python Integration Test        "
echo "=============================================="
# This runs the specific test we just wrote, ensuring the Python runtime
# uses the .env we just populated with the tetra3 module.
cargo test test_python_wrapper --release -- --ignored --nocapture

echo "=============================================="
echo "  Success!                                    "
echo "=============================================="
echo "To interact with the wrapper manually in Python, activate the environment:"
echo "    source $VENV_DIR/bin/activate"
echo "    python"
echo "    >>> import tetra3"
