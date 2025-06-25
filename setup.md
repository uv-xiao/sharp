# Sharp Project Setup Guide

This document provides instructions for setting up and building the Sharp project, an MLIR-based infrastructure project.

## Prerequisites

- CMake >= 3.20.0
- Python >= 3.8 (for Python bindings and lit tests)
- Git
- Ninja build system

Note: The project uses a standalone Clang-20 and LLD-20 for building, which will be automatically set up by pixi. No sudo privileges are required for any part of the build process.

## Dependencies

Sharp depends on:
- LLVM/MLIR (included as submodule in CIRCT)
- CIRCT (included as git submodule)

## Setup Instructions

### 1. Install Pixi (Package Manager)

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 2. Clone the Repository

```bash
git clone <your-repo-url> sharp
cd sharp
```

Note: We do NOT use `--recursive` here because CIRCT's LLVM submodule is very large. The submodules will be initialized properly by pixi with shallow clones.

### 3. Install Dependencies via Pixi

```bash
pixi install
```

This will automatically:
- Initialize git submodules (CIRCT and its LLVM dependency)
- Download and set up Clang-20 and LLD-20 in `.install/clang20/`
- Build LLVM/MLIR from CIRCT's submodule and install to `.install/llvm/`
- Build CIRCT and install to `.install/circt/`
- Set up the development environment

### 4. Build Sharp

#### Option A: Using Pixi (Recommended)

```bash
pixi run build
```

#### Option B: Manual CMake Build

```bash
mkdir build
cd build
cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMLIR_DIR=$PWD/../.install/llvm/lib/cmake/mlir \
  -DLLVM_DIR=$PWD/../.install/llvm/lib/cmake/llvm \
  -DCIRCT_DIR=$PWD/../.install/circt/lib/cmake/circt \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=$PWD/../.install/clang20/bin/clang \
  -DCMAKE_CXX_COMPILER=$PWD/../.install/clang20/bin/clang++
ninja
```

### 5. Run Tests

```bash
pixi run test
```

Or manually:

```bash
ninja check-sharp
```

## Development Commands

### Build Commands
- `pixi run build` - Build the project
- `pixi run build-release` - Build in release mode
- `pixi run clean` - Clean build artifacts

### Test Commands
- `pixi run test` - Run all tests
- `pixi run test-lit` - Run lit tests only
- `pixi run test-unit` - Run unit tests only

### Development Tools
- `pixi run format` - Format C++ code
- `pixi run tidy` - Run clang-tidy
- `pixi run docs` - Generate documentation

## Project Structure

```
sharp/
├── CMakeLists.txt          # Root build configuration
├── .gitmodules            # Git submodules configuration
├── circt/                 # CIRCT submodule
│   └── llvm/             # LLVM submodule (inside CIRCT)
├── .install/              # All build artifacts and installations
│   ├── clang20/          # Clang-20 installation
│   ├── llvm/             # LLVM/MLIR installation
│   ├── llvm-build/       # LLVM build directory
│   ├── circt/            # CIRCT installation
│   └── circt-build/      # CIRCT build directory
├── include/sharp/          # Public headers
│   ├── Dialect/           # Dialect definitions
│   ├── Analysis/          # Analysis passes
│   ├── Conversion/        # Conversion passes
│   ├── Support/           # Utilities
│   └── Transforms/        # Transformations
├── lib/                   # Implementation files
│   ├── Dialect/          # Dialect implementations
│   ├── Analysis/         # Analysis implementations
│   ├── Conversion/       # Conversion implementations
│   ├── Support/          # Utility implementations
│   └── Transforms/       # Transformation implementations
├── tools/                # Executable tools
│   └── sharp-opt/       # Sharp optimizer tool
├── test/                # FileCheck integration tests
├── unittests/           # C++ unit tests
└── docs/               # Documentation
```

## Environment Variables

The pixi environment automatically sets:
- `MLIR_DIR` - MLIR installation directory
- `LLVM_DIR` - LLVM installation directory
- `CIRCT_DIR` - CIRCT installation directory

## Troubleshooting

### Build Errors

1. **CMake cannot find MLIR/LLVM**
   - Ensure you've run `pixi install`
   - Check that environment variables are set: `pixi shell`

2. **Compilation errors**
   - Verify C++ compiler version: `clang++ --version` or `g++ --version`
   - Ensure C++17 support

3. **Test failures**
   - Run `pixi run build` before testing
   - Check test output in `build/test/`

4. **Large disk usage from LLVM**
   - If you accidentally cloned with `--recursive`, the LLVM submodule will be very large
   - To fix: `cd circt && git submodule deinit -f llvm && git submodule update --init --depth 1 llvm`

### Common Issues

- **Out of memory during build**: Reduce parallel jobs with `ninja -j4`
- **Python not found**: Install Python 3.8+ and ensure it's in PATH
- **CIRCT headers not found**: Rebuild with `pixi run build-deps`

## Contributing

Before submitting code:
1. Run `pixi run format` to format code
2. Run `pixi run test` to ensure tests pass
3. Run `pixi run tidy` for static analysis

## Additional Resources

- [MLIR Documentation](https://mlir.llvm.org/)
- [CIRCT Documentation](https://circt.llvm.org/)
- [Sharp Documentation](./docs/README.md)