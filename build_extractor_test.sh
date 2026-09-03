#!/bin/bash
# Copyright (c) 2026 Omair Kamil
# See LICENSE file in root directory for license terms.

# Exit on any error
set -e

echo "Compiling validate_extractor tests ..."

# Build the test binary but do not run it
cargo test --release -p tetra3 --test validate_extractor --no-run

# Cargo puts the test binaries in target/release/deps/ with a hash suffix.
# We look for the most recently modified executable file starting with "validate_extractor-".
BIN_PATH=$(ls -t target/release/deps/validate_extractor-* 2>/dev/null | grep -v "\.d$" | grep -v "\.rmeta$" | head -n 1)

if [ -n "$BIN_PATH" ] && [ -x "$BIN_PATH" ]; then
    # Copy it to the workspace root for easy access
    cp "$BIN_PATH" ./validate_extractor_bench
    echo ""
    echo "✅ Successfully compiled standalone test binary!"
    echo "Executable copied to: ./validate_extractor_bench"
    echo ""
    echo "To run the benchmark on your target machine, copy this file over and run:"
    echo "./validate_extractor_bench test_benchmark_bg_sub_modes --nocapture --ignored"
else
    echo "❌ Failed to find the compiled executable test binary."
    exit 1
fi
