#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INSTALL_DIR="$PROJECT_ROOT/.install"
CLANG_DIR="$INSTALL_DIR/clang20"

# Create install directory
mkdir -p "$INSTALL_DIR"

# Check if clang-20 is already installed
if [ -d "$CLANG_DIR" ] && [ -x "$CLANG_DIR/bin/clang" ] && [ -x "$CLANG_DIR/bin/ld.lld" ]; then
    VERSION=$("$CLANG_DIR/bin/clang" --version | head -n1)
    # Check for clang version 20 or newer
    if [[ "$VERSION" =~ clang\ version\ (20|2[1-9]|[3-9][0-9]) ]]; then
        echo "Clang-20+ already installed: $VERSION"
        # Verify lld is also present
        if [ -x "$CLANG_DIR/bin/ld.lld" ] || [ -x "$CLANG_DIR/bin/lld" ]; then
            exit 0
        else
            echo "Warning: lld not found, will reinstall..."
        fi
    else
        echo "Found older clang version: $VERSION"
        echo "Will install clang-20..."
    fi
fi

echo "Setting up clang-20 and lld-20..."

# Detect platform
OS=$(uname -s)
ARCH=$(uname -m)

if [ "$OS" = "Linux" ]; then
    if [ "$ARCH" = "x86_64" ]; then
        echo "Detected Linux x86_64"
        
        # Create temporary directory for download
        TEMP_DIR=$(mktemp -d)
        cd "$TEMP_DIR"
        
        # Download pre-built LLVM 20.1.7 from GitHub releases
        echo "Downloading pre-built LLVM 20.1.7..."
        RELEASE_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-20.1.7/LLVM-20.1.7-Linux-X64.tar.xz"
        
        if ! wget -q --spider "$RELEASE_URL" 2>/dev/null; then
            echo "Error: LLVM 20.1.7 release not found at: $RELEASE_URL"
            echo "Please check https://github.com/llvm/llvm-project/releases for available versions."
            exit 1
        fi
        
        echo "Downloading from: $RELEASE_URL"
        wget "$RELEASE_URL" -O llvm20.tar.xz --progress=bar:force 2>&1
        
        echo "Extracting..."
        mkdir -p "$CLANG_DIR"
        tar -xf llvm20.tar.xz --strip-components=1 -C "$CLANG_DIR"
        
        echo "Download and extraction complete."
    else
        echo "Unsupported architecture: $ARCH"
        echo "LLVM 20.1.7 pre-built binaries are only available for x86_64."
        echo "For $ARCH, you'll need to build from source."
        BUILD_FROM_SOURCE=1
    fi
elif [ "$OS" = "Darwin" ]; then
    echo "Detected macOS"
    # For macOS, check if pre-built binaries are available
    echo "Checking for macOS pre-built binaries..."
    
    if [ "$ARCH" = "x86_64" ]; then
        MACOS_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-20.1.7/LLVM-20.1.7-macOS-X64.tar.xz"
    elif [ "$ARCH" = "arm64" ]; then
        MACOS_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-20.1.7/LLVM-20.1.7-macOS-ARM64.tar.xz"
    fi
    
    if [ -n "$MACOS_URL" ] && wget -q --spider "$MACOS_URL" 2>/dev/null; then
        echo "Downloading from: $MACOS_URL"
        wget "$MACOS_URL" -O llvm20.tar.xz --progress=bar:force 2>&1
        echo "Extracting..."
        mkdir -p "$CLANG_DIR"
        tar -xf llvm20.tar.xz --strip-components=1 -C "$CLANG_DIR"
    else
        echo "No pre-built binaries found for macOS $ARCH."
        echo "Will build from source..."
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

# Clean up temporary directory if it exists
if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
    cd "$PROJECT_ROOT"
    rm -rf "$TEMP_DIR"
fi

# Verify installation
if [ -x "$CLANG_DIR/bin/clang" ]; then
    echo "Clang-20 successfully installed!"
    "$CLANG_DIR/bin/clang" --version
    
    # Check for lld or ld.lld
    if [ -x "$CLANG_DIR/bin/ld.lld" ]; then
        "$CLANG_DIR/bin/ld.lld" --version || true
    elif [ -x "$CLANG_DIR/bin/lld" ]; then
        "$CLANG_DIR/bin/lld" --version || true
    else
        echo "Warning: lld not found in installation"
    fi
else
    echo "Error: Failed to install clang-20"
    exit 1
fi

# Create symlink for lld if not present
if [ ! -x "$CLANG_DIR/bin/lld" ] && [ -x "$CLANG_DIR/bin/ld.lld" ]; then
    ln -sf ld.lld "$CLANG_DIR/bin/lld"
fi

echo "Setup complete!"