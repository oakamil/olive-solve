#!/usr/bin/env bash
# Copyright (c) 2026 Omair Kamil
# See LICENSE file in root directory for license terms.
# Build script for creating a standalone binary distribution of the fused plate solver.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building Standalone Fused Plate Solver Binary ==="

VENV_PYTHON="./venv/bin/python"
VENV_PYINSTALLER="./venv/bin/pyinstaller"

# Ensure virtual environment exists with system site-packages so it can access picamera2, libcamera, numpy
if [ ! -d "venv" ] || ! grep -q "include-system-site-packages = true" venv/pyvenv.cfg 2>/dev/null; then
    echo "Creating python virtual environment with system site-packages..."
    python3 -m venv --system-site-packages venv
fi

if [ ! -f "$VENV_PYINSTALLER" ]; then
    echo "Installing pyinstaller into virtual environment..."
    ./venv/bin/pip install pyinstaller
fi

# Ensure numpy is installed in virtual environment
if ! "$VENV_PYTHON" -c "import numpy" &>/dev/null; then
    echo "Installing numpy into virtual environment..."
    ./venv/bin/pip install numpy
fi

# Install latest olive_solve wheel into virtual environment so PyInstaller can discover and collect it
WHEEL=$(ls -t dist/olive_solve-*.whl 2>/dev/null | head -n 1)
if [ -n "$WHEEL" ]; then
    echo "Installing $WHEEL into virtual environment..."
    ./venv/bin/pip install --force-reinstall --no-deps "$WHEEL"
fi

echo "Verifying environment dependencies..."
"$VENV_PYTHON" -c "import numpy; import olive_solve; import picamera2; print('Dependencies verified: numpy, olive_solve, picamera2 present.')"

DB_FIXTURE="tetra3/tests/fixtures/default_database.npz"

if [ ! -f "$DB_FIXTURE" ]; then
    echo "Warning: Database fixture $DB_FIXTURE not found in standard path."
fi

ENTRY_SCRIPT="${1:-examples/fused_plate_solve.py}"
BIN_NAME="$(basename "$ENTRY_SCRIPT" .py)"

echo "Packaging standalone executable '$BIN_NAME' from '$ENTRY_SCRIPT' with PyInstaller..."
$VENV_PYINSTALLER \
    --clean \
    --noupx \
    --noconfirm \
    --onefile \
    --name "$BIN_NAME" \
    --add-data "$DB_FIXTURE:." \
    --collect-all numpy \
    --collect-all olive_solve \
    --collect-all picamera2 \
    --collect-all libcamera \
    --exclude-module av \
    --exclude-module cv2 \
    --exclude-module PyQt5 \
    --exclude-module tkinter \
    --exclude-module matplotlib \
    --exclude-module pandas \
    --exclude-module IPython \
    --exclude-module scipy \
    --exclude-module jedi \
    --exclude-module openpyxl \
    --exclude-module prompt_toolkit \
    --exclude-module tables \
    --exclude-module PyQt6 \
    --exclude-module PySide2 \
    --exclude-module PySide6 \
    --exclude-module PIL.ImageTk \
    "$ENTRY_SCRIPT"

echo ""
echo "=== Build Complete ==="
echo "Standalone binary created at: dist/$BIN_NAME"
echo ""
echo "To test on this machine or copy to another aarch64 Linux machine:"
echo "  ./dist/$BIN_NAME --help"

