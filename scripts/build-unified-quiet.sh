#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BUILD_LOG="$SCRIPT_DIR/../.install/unified-build.log"

# Run the actual build script with output redirection
if "$SCRIPT_DIR/build-unified.sh" > "$BUILD_LOG" 2>&1; then
    # If successful, just show the key messages
    if grep -q "Unified LLVM/MLIR/CIRCT build already complete" "$BUILD_LOG"; then
        echo "✓ Unified LLVM/MLIR/CIRCT build already complete"
    else
        echo "✓ Unified LLVM/MLIR/CIRCT build completed successfully"
        echo "  Build log: $BUILD_LOG"
        # Show Python path info
        tail -n 3 "$BUILD_LOG" | grep -E "PYTHONPATH|Python bindings" || true
    fi
else
    # If failed, show the error
    echo "✗ Unified build failed!"
    echo "  Error details:"
    tail -n 20 "$BUILD_LOG"
    echo "  Full log: $BUILD_LOG"
    exit 1
fi