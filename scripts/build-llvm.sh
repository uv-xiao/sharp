#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INSTALL_DIR="$PROJECT_ROOT/.install"
CIRCT_DIR="$PROJECT_ROOT/circt"
LLVM_DIR="$CIRCT_DIR/llvm"
BUILD_DIR="$INSTALL_DIR/llvm-build"
LLVM_INSTALL_DIR="$INSTALL_DIR/llvm"

echo "Building LLVM/MLIR for Sharp..."

# Check if CIRCT submodule is initialized
if [ ! -d "$CIRCT_DIR/.git" ]; then
    echo "Error: CIRCT submodule not initialized. Please run 'pixi run init-submodules' first."
    exit 1
fi

# Check if LLVM submodule in CIRCT is initialized
if [ ! -d "$LLVM_DIR/.git" ]; then
    echo "Initializing CIRCT's LLVM submodule with shallow clone..."
    cd "$CIRCT_DIR"
    git submodule update --init --depth 1 llvm
fi

# Create build and install directories
mkdir -p "$BUILD_DIR"
mkdir -p "$LLVM_INSTALL_DIR"
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

# Configure LLVM/MLIR build
cmake "$LLVM_DIR/llvm" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
    -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_DIR" \
    -DCMAKE_C_COMPILER="$CLANG_PATH" \
    -DCMAKE_CXX_COMPILER="$CLANGXX_PATH" \
    -DLLVM_USE_LINKER="$LLD_PATH" \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS="${LLVM_ENABLE_ASSERTIONS:-ON}" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_INSTALL_UTILS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python)

# Build and install LLVM/MLIR
ninja
ninja install

echo "LLVM/MLIR build complete!"
echo "Installed to: $LLVM_INSTALL_DIR"