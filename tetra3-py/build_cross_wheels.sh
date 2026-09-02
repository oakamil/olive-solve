#!/usr/bin/env bash
# ==============================================================================
# Cross-Compilation Wheel Builder for Tetra3 Python Bindings
# Host: Raspberry Pi 5 (aarch64)
# Targets:
#   - Pi Zero 2W / Pi 3 / Pi 4 / Pi 5 (aarch64, Cortex-A53 baseline)
#   - Pi Zero 1 (armv6l, ARM1176JZF-S, VFPv2, 32-bit solver)
#   - Rockchip RV1103 (armv7l, Cortex-A7, NEON + VFPv4, 32-bit solver)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WHEELS_DIR="$PROJECT_ROOT/target/wheels"
cd "$SCRIPT_DIR"

echo "========================================================"
echo "    Tetra3 Embedded Python Wheel Builder (on Pi 5)     "
echo "========================================================"

# Virtual Environment & Tooling Verification
if ! command -v maturin &>/dev/null; then
    VENV_DIR=".env"
    if [ ! -d "$VENV_DIR" ]; then
        echo "[Setup] No maturin found in PATH. Creating virtualenv in ./$VENV_DIR..."
        python3 -m venv "$VENV_DIR"
    fi
    echo "[Setup] Activating virtual environment $VENV_DIR..."
    source "$VENV_DIR/bin/activate"
    if ! command -v maturin &>/dev/null; then
        echo "[Setup] Installing maturin in virtualenv..."
        pip install --upgrade pip
        pip install maturin
    fi
fi

# Verify Cross Linker
check_cross_linker() {
    if ! command -v arm-linux-gnueabihf-gcc &>/dev/null; then
        echo ""
        echo "ERROR: Cross-linker 'arm-linux-gnueabihf-gcc' not found!"
        echo "Please install it on your Pi 5 by running:"
        echo "    sudo apt-get update && sudo apt-get install -y gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf"
        echo ""
        exit 1
    fi
}

# Ensure rustup target is installed
ensure_rust_target() {
    local target="$1"
    if ! rustup target list | grep -E "^${target} \(installed\)" &>/dev/null; then
        echo "[Setup] Adding Rust target: $target..."
        rustup target add "$target"
    fi
}

# Target 1: Universal 64-Bit (Pi Zero 2W / 3 / 4 / 5)
build_zero2w() {
    echo ""
    echo ">>> [1/3] Building Universal 64-Bit Wheel (Pi Zero 2W / 3 / 4 / 5)..."
    ensure_rust_target "aarch64-unknown-linux-gnu"
    maturin build --release --strip
}

# Target 2: Raspberry Pi Zero 1 (ARMv6, VFPv2, 32-bit solver)
build_zero1() {
    echo ""
    echo ">>> [2/3] Building Raspberry Pi Zero 1 Wheel (ARMv6, force-32bit-solver)..."
    check_cross_linker
    ensure_rust_target "arm-unknown-linux-gnueabihf"
    maturin build \
        --release \
        --target arm-unknown-linux-gnueabihf \
        --features force-32bit-solver \
        --strip
}

# Target 3: Rockchip RV1103 (Cortex-A7 + NEON, 32-bit solver)
build_rv1103() {
    echo ""
    echo ">>> [3/3] Building Rockchip RV1103 Wheel (Cortex-A7 + NEON, force-32bit-solver)..."
    check_cross_linker
    ensure_rust_target "armv7-unknown-linux-gnueabihf"
    maturin build \
        --release \
        --target armv7-unknown-linux-gnueabihf \
        --features force-32bit-solver \
        --strip
}

# Target 3b: Rockchip RV1103 musl (Buildroot)
build_rv1103_musl() {
    echo ""
    echo ">>> Building Rockchip RV1103 musl Wheel (Buildroot, force-32bit-solver)..."
    check_cross_linker
    ensure_rust_target "armv7-unknown-linux-musleabihf"
    maturin build \
        --release \
        --target armv7-unknown-linux-musleabihf \
        --features force-32bit-solver \
        --strip
}

TARGET="${1:-all}"

case "$TARGET" in
    zero2w|aarch64|pi3|pi4|pi5)
        build_zero2w
        ;;
    zero1|armv6|armv6l)
        build_zero1
        ;;
    rv1103|armv7|cortex-a7)
        build_rv1103
        ;;
    rv1103-musl)
        build_rv1103_musl
        ;;
    all)
        build_zero2w
        build_zero1
        build_rv1103
        ;;
    *)
        echo "Usage: $0 [zero2w | zero1 | rv1103 | rv1103-musl | all]"
        exit 1
        ;;
esac

echo ""
echo "========================================================"
echo " Build Complete! Generated wheels in $WHEELS_DIR:"
echo "========================================================"
ls -lh "$WHEELS_DIR"/*.whl

# Also mirror wheels to tetra3-py/dist/ for convenience
mkdir -p "$SCRIPT_DIR/dist"
cp -u "$WHEELS_DIR"/tetra3-*.whl "$SCRIPT_DIR/dist/" 2>/dev/null || cp "$WHEELS_DIR"/tetra3-*.whl "$SCRIPT_DIR/dist/"
echo ""
echo "Wheels mirrored to $SCRIPT_DIR/dist/"
