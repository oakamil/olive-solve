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

if [ ! -f "$VENV_PYINSTALLER" ]; then
    echo "Installing pyinstaller into virtual environment..."
    ./venv/bin/pip install pyinstaller
fi

DB_FIXTURE="tetra3/tests/fixtures/default_database.npz"

if [ ! -f "$DB_FIXTURE" ]; then
    echo "Warning: Database fixture $DB_FIXTURE not found in standard path."
fi

echo "Packaging standalone executable with PyInstaller..."
$VENV_PYINSTALLER \
    --clean \
    --noupx \
    --noconfirm \
    --onefile \
    --name fused_plate_solve \
    --add-data "$DB_FIXTURE:." \
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
    examples/fused_plate_solve.py

echo ""
echo "=== Build Complete ==="
echo "Standalone binary created at: dist/fused_plate_solve"
echo ""
echo "To test on this machine or copy to another aarch64 Linux machine:"
echo "  ./dist/fused_plate_solve --help"
echo "  ./dist/fused_plate_solve --exposure 100 --lat 37.35 --lon -121.96 --rate 10"
