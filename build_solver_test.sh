#!/bin/bash
# Copyright (c) 2026 Omair Kamil
# See LICENSE file in root directory for license terms.

# Exit on any error
set -e

echo "Compiling validate_solver tests ..."

# Build the test binary but do not run it
cargo test --release -p tetra3 --test validate_solver --no-run

# Cargo puts the test binaries in target/release/deps/ with a hash suffix.
# We look for the most recently modified executable file starting with "validate_solver-".
BIN_PATH=$(ls -t target/release/deps/validate_solver-* 2>/dev/null | grep -v "\.d$" | grep -v "\.rmeta$" | head -n 1)

if [ -n "$BIN_PATH" ] && [ -x "$BIN_PATH" ]; then
    echo "✅ Successfully compiled standalone test binary: $BIN_PATH"

    # Create a temporary directory for packaging
    TMP_DIR=$(mktemp -d)

    # Copy binary
    cp "$BIN_PATH" "$TMP_DIR/validate_solver_bench"

    # Copy fixtures with correct relative path
    # The test looks for "tests/fixtures/..." so we mirror that structure
    mkdir -p "$TMP_DIR/tests/fixtures"
    cp tetra3/tests/fixtures/default_database.npz "$TMP_DIR/tests/fixtures/"
    cp tetra3/tests/fixtures/solver_fixtures.zip "$TMP_DIR/tests/fixtures/"

    # Package into zip
    ZIP_NAME="solver_test_package.zip"
    rm -f "$ZIP_NAME"
    cd "$TMP_DIR"
    zip -r "$ZIP_NAME" validate_solver_bench tests/fixtures
    cd - > /dev/null

    mv "$TMP_DIR/$ZIP_NAME" .
    rm -rf "$TMP_DIR"

    echo ""
    echo "📦 Packaged binary and fixtures into: ./$ZIP_NAME"
    echo ""
    echo "To run the benchmark on your target machine, copy $ZIP_NAME over, unzip it, and run:"
    echo "unzip $ZIP_NAME"
    echo "./validate_solver_bench test_location_filtering_performance --nocapture"
else
    echo "❌ Failed to find the compiled executable test binary."
    exit 1
fi
