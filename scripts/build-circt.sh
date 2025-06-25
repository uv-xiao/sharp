#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INSTALL_DIR="$PROJECT_ROOT/.install"
CIRCT_DIR="$PROJECT_ROOT/circt"
BUILD_DIR="$INSTALL_DIR/circt-build"
CIRCT_INSTALL_DIR="$INSTALL_DIR/circt"
LLVM_INSTALL_DIR="$INSTALL_DIR/llvm"

echo "Building CIRCT for Sharp..."

# Check if CIRCT submodule is initialized
if [ ! -d "$CIRCT_DIR/.git" ]; then
    echo "Error: CIRCT submodule not initialized. Please run 'pixi run init-submodules' first."
    exit 1
fi

# Create build and install directories
mkdir -p "$BUILD_DIR"
mkdir -p "$CIRCT_INSTALL_DIR"
cd "$BUILD_DIR"

# Use clang-20
export PATH="$INSTALL_DIR/clang20/bin:$PATH"
CLANG_PATH="$INSTALL_DIR/clang20/bin/clang"
CLANGXX_PATH="$INSTALL_DIR/clang20/bin/clang++"
LLD_PATH="$INSTALL_DIR/clang20/bin/lld"

# Verify clang-20 is available
if [ ! -x "$CLANG_PATH" ]; then
    echo "Error: clang-20 not found. Please run 'pixi run setup-clang20' first."
    exit 1
fi

echo "Using clang: $($CLANG_PATH --version | head -n1)"
echo "Using lld: $($LLD_PATH --version | head -n1 || echo 'lld not found, using default linker')"

# Configure CIRCT build
cmake "$CIRCT_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
    -DCMAKE_INSTALL_PREFIX="$CIRCT_INSTALL_DIR" \
    -DCMAKE_C_COMPILER="$CLANG_PATH" \
    -DCMAKE_CXX_COMPILER="$CLANGXX_PATH" \
    -DMLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir" \
    -DLLVM_DIR="$LLVM_INSTALL_DIR/lib/cmake/llvm" \
    -DLLVM_ENABLE_ASSERTIONS="${LLVM_ENABLE_ASSERTIONS:-ON}" \
    -DLLVM_USE_LINKER="$LLD_PATH" \
    -DCIRCT_BINDINGS_PYTHON_ENABLED=ON \
    -DPython3_EXECUTABLE=$(which python)

# Build and install CIRCT
ninja
ninja install

echo "CIRCT build complete!"
echo "Installed to: $CIRCT_INSTALL_DIR"