#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INSTALL_DIR="$PROJECT_ROOT/.install"
CLANG_DIR="$INSTALL_DIR/clang20"

# Create install directory
mkdir -p "$INSTALL_DIR"

# Check if clang-20 is already installed
if [ -d "$CLANG_DIR" ] && [ -x "$CLANG_DIR/bin/clang" ]; then
    VERSION=$("$CLANG_DIR/bin/clang" --version | head -n1)
    if [[ "$VERSION" == *"clang version 20"* ]]; then
        echo "Clang-20 already installed: $VERSION"
        exit 0
    fi
fi

echo "Setting up clang-20 and lld-20..."

# Detect platform
OS=$(uname -s)
ARCH=$(uname -m)

if [ "$OS" = "Linux" ]; then
    if [ "$ARCH" = "x86_64" ]; then
        # For Ubuntu/Debian based systems, we can use the LLVM apt repository
        echo "Detected Linux x86_64"
        
        # Create temporary directory for download
        TEMP_DIR=$(mktemp -d)
        cd "$TEMP_DIR"
        
        # Download pre-built LLVM 20 from GitHub releases
        # Note: As of the time of writing, LLVM 20 might not be released yet.
        # You may need to build from source or use a nightly build.
        
        # Try to download pre-built binaries (no sudo required)
        echo "Attempting to download pre-built LLVM 20..."
            
        # Check GitHub releases for LLVM
        RELEASE_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-20.0.0/clang+llvm-20.0.0-x86_64-linux-gnu-ubuntu-22.04.tar.xz"
        
        # Note: The above URL is hypothetical. Check actual releases at:
        # https://github.com/llvm/llvm-project/releases
        
        # For development/nightly builds:
        echo "Checking for LLVM 20 pre-built releases..."
        NIGHTLY_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-20.0.0-rc1/clang+llvm-20.0.0-rc1-x86_64-linux-gnu.tar.xz"
        
        # Try to download
        if ! wget -q --spider "$NIGHTLY_URL" 2>/dev/null; then
            echo "LLVM 20 pre-built not found. Will build from source..."
            BUILD_FROM_SOURCE=1
        else
            wget "$NIGHTLY_URL" -O llvm20.tar.xz
            echo "Extracting..."
            tar -xf llvm20.tar.xz
            mv clang+llvm-*/* "$CLANG_DIR/"
        fi
    else
        echo "Unsupported architecture: $ARCH"
        BUILD_FROM_SOURCE=1
    fi
elif [ "$OS" = "Darwin" ]; then
    echo "Detected macOS"
    # For macOS, try to download pre-built binaries first
    echo "Checking for macOS pre-built binaries..."
    
    if [ "$ARCH" = "x86_64" ]; then
        MACOS_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-20.0.0/clang+llvm-20.0.0-x86_64-apple-darwin.tar.xz"
    elif [ "$ARCH" = "arm64" ]; then
        MACOS_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-20.0.0/clang+llvm-20.0.0-arm64-apple-darwin.tar.xz"
    fi
    
    if [ -n "$MACOS_URL" ] && wget -q --spider "$MACOS_URL" 2>/dev/null; then
        wget "$MACOS_URL" -O llvm20.tar.xz
        echo "Extracting..."
        tar -xf llvm20.tar.xz
        mv clang+llvm-*/* "$CLANG_DIR/"
    else
        echo "No pre-built binaries found for macOS. Will build from source..."
        BUILD_FROM_SOURCE=1
    fi
else
    echo "Unsupported OS: $OS"
    BUILD_FROM_SOURCE=1
fi

# Build from source if pre-built not available
if [ "$BUILD_FROM_SOURCE" = "1" ]; then
    echo "Building LLVM 20 from source..."
    cd "$PROJECT_ROOT"
    
    # Clone LLVM if not exists
    LLVM_SRC_DIR="$INSTALL_DIR/llvm-project-20"
    if [ ! -d "$LLVM_SRC_DIR" ]; then
        git clone --depth 1 --branch release/20.x https://github.com/llvm/llvm-project.git "$LLVM_SRC_DIR" || \
        git clone --depth 1 --branch main https://github.com/llvm/llvm-project.git "$LLVM_SRC_DIR"
    fi
    
    # Build minimal clang and lld
    BUILD_DIR="$LLVM_SRC_DIR/build"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    cmake ../llvm \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$CLANG_DIR" \
        -DLLVM_ENABLE_PROJECTS="clang;lld" \
        -DLLVM_TARGETS_TO_BUILD="host" \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        -DLLVM_ENABLE_ZLIB=OFF \
        -DLLVM_ENABLE_ZSTD=OFF \
        -DLLVM_ENABLE_TERMINFO=OFF \
        -DLLVM_ENABLE_LIBXML2=OFF \
        -DLLVM_BUILD_EXAMPLES=OFF \
        -DLLVM_BUILD_TESTS=OFF \
        -DLLVM_BUILD_BENCHMARKS=OFF \
        -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DLLVM_INCLUDE_BENCHMARKS=OFF \
        -DCLANG_BUILD_EXAMPLES=OFF \
        -DCLANG_INCLUDE_TESTS=OFF
    
    ninja install
fi

# Clean up
if [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
fi

# Verify installation
if [ -x "$CLANG_DIR/bin/clang" ]; then
    echo "Clang-20 successfully installed!"
    "$CLANG_DIR/bin/clang" --version
    "$CLANG_DIR/bin/lld" --version || true
else
    echo "Error: Failed to install clang-20"
    exit 1
fi

# Create symlink for lld if not present
if [ ! -x "$CLANG_DIR/bin/lld" ] && [ -x "$CLANG_DIR/bin/ld.lld" ]; then
    ln -sf ld.lld "$CLANG_DIR/bin/lld"
fi

echo "Setup complete!"