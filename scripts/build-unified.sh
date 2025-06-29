#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INSTALL_DIR="$PROJECT_ROOT/.install"
CIRCT_DIR="$PROJECT_ROOT/circt"
LLVM_DIR="$CIRCT_DIR/llvm"
BUILD_DIR="$INSTALL_DIR/unified-build"
UNIFIED_INSTALL_DIR="$INSTALL_DIR/unified"
MARKER_FILE="$UNIFIED_INSTALL_DIR/.build-complete"

# Check if build is already complete
if [ -f "$MARKER_FILE" ]; then
    # Check if gtest support is already enabled
    if grep -q "GTest support: Enabled" "$MARKER_FILE" 2>/dev/null; then
        echo "Unified LLVM/MLIR/CIRCT build already complete (with gtest support)."
        echo "To force rebuild, delete: $MARKER_FILE"
        exit 0
    else
        echo "Existing build found without gtest support. Rebuilding with gtest enabled..."
        rm -f "$MARKER_FILE"
        # Also clean the build directory to ensure clean rebuild
        echo "Cleaning build directory for fresh build with gtest..."
        rm -rf "$BUILD_DIR"
    fi
fi

echo "Building LLVM/MLIR/CIRCT in unified mode for Python bindings..."

# Check if CIRCT submodule is initialized
if [ ! -d "$CIRCT_DIR" ] || [ ! -f "$CIRCT_DIR/CMakeLists.txt" ]; then
    echo "Error: CIRCT submodule not initialized. Please run 'pixi run init-submodules' first."
    exit 1
fi

# Check if LLVM submodule in CIRCT is initialized
if [ ! -d "$LLVM_DIR" ] || [ ! -f "$LLVM_DIR/CMakeLists.txt" ]; then
    echo "Initializing CIRCT's LLVM submodule with shallow clone..."
    cd "$CIRCT_DIR"
    git submodule update --init --depth 1 llvm
fi

# Create build and install directories
mkdir -p "$BUILD_DIR"
mkdir -p "$UNIFIED_INSTALL_DIR"
cd "$BUILD_DIR"

# Use clang-20 from environment or fallback to explicit path
if [ -n "$CC" ] && [ -x "$CC" ]; then
    CLANG_PATH="$CC"
else
    CLANG_PATH="$INSTALL_DIR/clang20/bin/clang"
fi

if [ -n "$CXX" ] && [ -x "$CXX" ]; then
    CLANGXX_PATH="$CXX"
else
    CLANGXX_PATH="$INSTALL_DIR/clang20/bin/clang++"
fi

if [ -n "$LD" ] && [ -x "$LD" ]; then
    LLD_PATH="$LD"
else
    LLD_PATH="$INSTALL_DIR/clang20/bin/ld.lld"
fi

# Ensure the PATH includes clang20
export PATH="$INSTALL_DIR/clang20/bin:$PATH"

# Verify clang is available
if [ ! -x "$CLANG_PATH" ]; then
    echo "Error: clang not found. Please run 'pixi run setup-clang20' first."
    exit 1
fi

CLANG_VERSION=$($CLANG_PATH --version | head -n1)
echo "Using clang: $CLANG_VERSION"

# Verify it's at least clang-20
if [[ ! "$CLANG_VERSION" =~ "clang version (20|2[1-9]|[3-9][0-9])" ]]; then
    echo "Warning: Expected clang-20 or newer, but found: $CLANG_VERSION"
    echo "Run 'pixi run setup-clang20' to install the correct version."
fi

# Check if lld is available
if [ -x "$LLD_PATH" ] || [ -x "$INSTALL_DIR/clang20/bin/ld.lld" ]; then
    if [ -x "$INSTALL_DIR/clang20/bin/ld.lld" ]; then
        LLD_PATH="$INSTALL_DIR/clang20/bin/ld.lld"
    fi
    echo "Using lld: $($LLD_PATH --version | head -n1)"
    LINKER_FLAGS="-DLLVM_USE_LINKER=lld"
else
    echo "lld not found, using default linker"
    LINKER_FLAGS=""
fi

# Configure unified build
echo "Configuring unified LLVM/MLIR/CIRCT build..."
cmake "$LLVM_DIR/llvm" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
    -DCMAKE_INSTALL_PREFIX="$UNIFIED_INSTALL_DIR" \
    -DCMAKE_C_COMPILER="$CLANG_PATH" \
    -DCMAKE_CXX_COMPILER="$CLANGXX_PATH" \
    $LINKER_FLAGS \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_EXTERNAL_PROJECTS="circt" \
    -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR="$CIRCT_DIR" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS="${LLVM_ENABLE_ASSERTIONS:-ON}" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_ENABLE_OCAMLDOC=OFF \
    -DLLVM_OCAML_INSTALL_PATH=/tmp/llvm-ocaml \
    -DLLVM_BUILD_TESTS=ON \
    -DLLVM_INSTALL_GTEST=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DCIRCT_BINDINGS_PYTHON_ENABLED=ON \
    -DVERILATOR_DISABLE=ON \
    -DPython3_EXECUTABLE=$(which python)

# Build and install
echo "Building unified LLVM/MLIR/CIRCT..."
ninja

echo "Installing unified build..."
ninja install

# Create marker file with build info
cat > "$MARKER_FILE" << EOF
Build completed on: $(date)
Compiler: $($CLANG_PATH --version | head -n1)
Build type: ${CMAKE_BUILD_TYPE:-Release}
Python bindings: Enabled
GTest support: Enabled
EOF

echo "Unified build complete!"
echo "Installed to: $UNIFIED_INSTALL_DIR"

# Set up Python path
echo ""
echo "To use the Python bindings, add this to your environment:"
echo "export PYTHONPATH=\"$BUILD_DIR/tools/circt/python_packages/circt_core:$BUILD_DIR/tools/mlir/python_packages/mlir_core\""