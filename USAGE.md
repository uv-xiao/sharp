# Sharp Project Usage Guide

This document provides comprehensive instructions for setting up, building, and using the Sharp project - an MLIR-based infrastructure for bridging hardware, architecture, and programming.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Build System](#build-system)
- [Development Commands](#development-commands)
- [Project Structure](#project-structure)
- [Current Status](#current-status)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- CMake >= 3.20.0
- Python >= 3.8 (for Python bindings and lit tests)
- Git
- Ninja build system

Note: The project uses a standalone Clang-20 and LLD-20 for building, which will be automatically set up by pixi. No sudo privileges are required for any part of the build process.

## Quick Start

```bash
# 1. Install Pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Clone the repository (without --recursive)
git clone <your-repo-url> sharp
cd sharp

# 3. Install dependencies and build
pixi install
pixi run build

# 4. Run sharp-opt
./build/bin/sharp-opt <input.mlir>

# 5. Set up Python bindings (optional)
export PYTHONPATH="$PWD/.install/unified-build/tools/circt/python_packages/circt_core:$PWD/.install/unified-build/tools/mlir/python_packages/mlir_core"
```

## Installation

### Step 1: Install Pixi Package Manager

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Step 2: Clone the Repository

```bash
git clone <your-repo-url> sharp
cd sharp
```

**Important**: Do NOT use `--recursive` here because CIRCT's LLVM submodule is very large. The submodules will be initialized properly by pixi with shallow clones.

### Step 3: Install Dependencies

```bash
pixi install
```

This command automatically:
- Initializes git submodules (CIRCT and its LLVM dependency) with shallow clones
- Downloads and sets up Clang-20 and LLD-20 in `.install/clang20/`
- Builds LLVM/MLIR/CIRCT with Python bindings in `.install/unified/`
- Sets up the development environment

## Build System

### Overview

The Sharp project build system has been optimized to:
1. Guarantee all builds use clang-20 and lld
2. Reduce unnecessary output and warnings during builds
3. Avoid redundant rebuilds using marker files
4. Provide both quiet and verbose build modes

### Key Features

#### 1. Clang-20 and LLD Enforcement
- All builds use clang-20 via pixi environment activation
- Environment variables (`CC`, `CXX`, `LD`, `AR`, `RANLIB`) are set automatically
- PATH includes clang-20 binaries
- CMake variables `LLVM_DIR`, `MLIR_DIR`, `CIRCT_DIR` are set for convenience

#### 2. Build Output Reduction
- **Unified Build**: Uses a marker file (`.install/unified/.build-complete`) to skip rebuilds
- **Quiet Mode**: Build output is redirected to log files with only key messages shown
- **Smart Scripts**: All setup scripts check if work is already done before proceeding
- Build logs saved to `.install/unified-build.log`

### Build Commands

#### Standard Build (Quiet)
```bash
pixi run build
```
Output is minimal, showing only:
- ✓ Build status messages
- Important warnings/errors
- Build completion status

#### Verbose Build (for debugging)
```bash
pixi run build-unified-verbose
```

#### Force Rebuild
```bash
rm .install/unified/.build-complete
pixi run build
```

#### Manual CMake Build (Alternative)
```bash
mkdir build
cd build
cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMLIR_DIR=$PWD/../.install/unified/lib/cmake/mlir \
  -DLLVM_DIR=$PWD/../.install/unified/lib/cmake/llvm \
  -DCIRCT_DIR=$PWD/../.install/unified/lib/cmake/circt \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=$PWD/../.install/clang20/bin/clang \
  -DCMAKE_CXX_COMPILER=$PWD/../.install/clang20/bin/clang++
ninja
```

## Development Commands

### Build Commands
- `pixi run build` - Build the project (quiet mode)
- `pixi run build-unified-verbose` - Build with full output
- `pixi run build-release` - Build in release mode
- `pixi run clean` - Clean build artifacts
- `pixi run configure` - Reconfigure CMake

### Test Commands
- `pixi run test` - Run all tests
- `pixi run test-lit` - Run lit tests only
- `pixi run test-unit` - Run unit tests (requires LLVM built with gtest)

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
│   ├── unified/          # Unified LLVM/MLIR/CIRCT installation
│   └── unified-build/    # Build directory with Python packages
├── include/sharp/         # Public headers
│   ├── Dialect/          # Dialect definitions
│   │   └── Core/        # Sharp Core dialect
│   ├── Analysis/         # Analysis passes
│   ├── Conversion/       # Conversion passes
│   ├── Support/          # Utilities
│   └── Transforms/       # Transformations
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

## Python Bindings

The Sharp project includes Python bindings that follow the CIRCT pattern. The bindings allow you to work with Sharp dialects from Python.

### Structure

The Python bindings are organized as follows:
```
sharp/lib/Bindings/Python/
├── CMakeLists.txt           # Build configuration
├── SharpModule.cpp          # Main C++ extension module
├── __init__.py              # Python package initialization
├── support.py               # Support utilities
├── construction.py          # Pythonic construction API
└── dialects/                # Dialect-specific bindings
    ├── SharpOps.td          # TableGen for Python bindings
    └── sharp.py             # Sharp dialect Python module
```

### Pythonic Construction API

Sharp provides a high-level Pythonic API for constructing hardware modules, similar to CIRCT's PyCDE. This allows hardware designers to use familiar Python syntax and patterns while generating Sharp Txn dialect IR.

#### Quick Start

```python
from sharp.construction import module, ModuleBuilder, i32, i8, ConflictRelation

@module
def Counter():
    builder = ModuleBuilder("Counter")
    
    @builder.value_method(return_type=i32)
    def getValue(b):
        return b.constant(42)
        
    @builder.action_method(return_type=i32)
    def increment(b, current: i32):
        one = b.constant(1)
        return current + one
        
    @builder.rule
    def autoIncrement(b):
        # Rules fire automatically
        pass
        
    # Add conflict relationships
    builder.add_conflict("increment", "autoIncrement", ConflictRelation.C)
    
    return builder

# Generate MLIR
mlir_module = Counter.build()
print(mlir_module)
```

#### Key Features

- **Decorator-based module definition** using `@module`
- **Type-safe hardware types** (i8, i16, i32, i64, etc.)
- **Operator overloading** for arithmetic and logic operations
- **Automatic conflict matrix management**
- **Integration with existing Python tools and workflows**

#### Hardware Types

```python
from sharp.construction import i1, i8, i16, i32, i64, i128, i256, BoolType, IntType

# Predefined types
bool_type = i1        # 1-bit boolean
byte_type = i8        # 8-bit integer  
word_type = i32       # 32-bit integer

# Custom types
custom_type = IntType(width=24)  # 24-bit integer
```

#### Operator Overloading

```python
@builder.value_method(return_type=i32)
def compute_example(b, a: i32, flag: i1):
    # Arithmetic operations
    result = a + 10
    result = result * 2
    result = result - 5
    
    # Bitwise operations
    masked = result & 0xFF
    combined = masked | 0x100
    
    # Shifts
    left_shifted = result << 2
    right_shifted = result >> 1
    
    # Comparisons
    is_equal = result == 42
    is_greater = result > 100
    
    # Select based on condition
    final = b.select(flag, result, b.constant(0))
    
    return final
```

#### Conflict Management

```python
# Define relationships between actions
builder.add_conflict("action1", "action2", ConflictRelation.C)   # Conflict
builder.add_conflict("rule1", "action1", ConflictRelation.SB)   # rule1 before action1
builder.add_conflict("method1", "method2", ConflictRelation.SA) # method1 after method2
builder.add_conflict("rule2", "rule3", ConflictRelation.CF)     # Conflict-free
```

#### Setup and Usage

1. **Build Sharp with Python bindings**:
   ```bash
   pixi run build
   ```

2. **Set up Python path**:
   ```bash
   export PYTHONPATH="$PWD/.install/unified-build/tools/circt/python_packages/circt_core:$PWD/.install/unified-build/tools/mlir/python_packages/mlir_core:$PWD/build/lib/Bindings/Python"
   ```

3. **Use the Pythonic API**:
   ```python
   from sharp.construction import module, ModuleBuilder, i32
   
   # Define your hardware modules using decorators
   @module
   def MyModule():
       builder = ModuleBuilder("MyModule")
       # ... define methods and rules
       return builder
   
   # Generate MLIR
   mlir_code = MyModule.build()
   ```

For comprehensive examples and documentation, see `docs/pythonic_frontend.md`.

### Low-Level MLIR Bindings

For direct MLIR manipulation, you can use the low-level bindings:

```python
import sys
# Add paths for MLIR and Sharp Python packages
sys.path.insert(0, '.install/unified/python_packages/mlir_core')
sys.path.insert(0, 'build/python_packages/sharp_core')

import mlir
from mlir import ir
import sharp

# Create context and register Sharp dialects
with ir.Context() as ctx:
    # Register Sharp dialects with the context
    sharp.register_dialects(ctx)
    
    # Access the Sharp dialect
    sharp_dialect = ctx.dialects["sharp"]
    
    # Parse Sharp operations
    module = ir.Module.parse("""
        func.func @test_constant() -> i32 {
            %0 = "sharp.constant"() {value = 42 : i32} : () -> i32
            return %0 : i32
        }
    """, ctx)
    
    print(module)
```

### Testing

Test scripts are provided:
```bash
# Test the Pythonic construction API
pixi run python test/python/construction_test.py

# Test low-level MLIR bindings
pixi run python test_sharp.py
```

### Current Status

The Python bindings have been implemented following the CIRCT pattern:
- ✅ C++ extension module using nanobind
- ✅ TableGen-based operation bindings
- ✅ Dialect registration mechanism
- ✅ Complete Pythonic construction API with decorators and operator overloading
- ✅ Type-safe hardware types and automatic MLIR conversion
- ✅ Conflict matrix management
- ⚠️ Runtime issue under investigation (possible ABI compatibility issue)

The Pythonic construction API is fully functional and provides a modern Python interface for hardware design. The low-level bindings are structurally complete but there is a runtime crash when registering dialects that needs to be resolved. This appears to be related to ABI compatibility between the unified LLVM/MLIR/CIRCT build and the Sharp Python extension.

## Current Status

### ✅ Working Components

1. **Build System**
   - Unified LLVM/MLIR/CIRCT build with Python bindings
   - Standalone clang-20 and lld installation
   - Smart rebuild detection
   - Both quiet and verbose build modes

2. **sharp-opt Tool**
   - Successfully built and executable
   - Can process standard MLIR dialects
   - Sharp dialect is registered

3. **Infrastructure**
   - Proper project structure following CIRCT patterns
   - CMake build configuration
   - Test infrastructure setup

### ⚠️ Components Needing Attention

1. **Sharp Dialect**
   - The dialect is recognized but the custom parser needs adjustment
   - The constant operation parser expects different syntax

2. **Python Bindings**
   - Structurally complete following CIRCT's pattern
   - Runtime crash when registering dialects (likely ABI compatibility issue)
   - See the Python Bindings section above for details

## Environment Variables

The pixi environment automatically sets:
- `MLIR_DIR` - MLIR installation directory
- `LLVM_DIR` - LLVM installation directory
- `CIRCT_DIR` - CIRCT installation directory
- `CC`, `CXX` - Clang-20 compilers
- `LD` - LLD linker

## Troubleshooting

### Build Errors

1. **CMake cannot find MLIR/LLVM**
   - Ensure you've run `pixi install`
   - Check that environment variables are set: `pixi shell`

2. **Compilation errors**
   - Verify compiler version: Should be clang-20
   - Ensure C++17 support
   - Check build logs in `.install/unified-build.log`

3. **Test failures**
   - Run `pixi run build` before testing
   - Check test output in `build/test/`

4. **sharp-opt not found**
   - Ensure build completed successfully
   - Binary should be at `./build/bin/sharp-opt`

### Common Issues

- **Out of memory during build**: Reduce parallel jobs with `ninja -j4`
- **Python not found**: Install Python 3.8+ and ensure it's in PATH
- **CIRCT headers not found**: Force rebuild with `rm .install/unified/.build-complete && pixi run build`
- **Large disk usage**: If LLVM was cloned with full history, re-clone with shallow depth
- **Unit tests fail with missing gtest**: The unified LLVM build doesn't include gtest libraries. Unit tests are optional. Use lit tests instead with `pixi run test-lit`

### Disk Space Management

The `.install` directory contains:
- `clang20/` - Standalone clang-20 compiler and lld (~500MB)
- `unified/` - Unified LLVM/MLIR/CIRCT installation (~2GB)
- `unified-build/` - Build directory with Python packages (~10GB)

To free space, you can remove `unified-build/` after installation is complete.

## Next Steps

1. Fix the Sharp constant operation parser to handle the syntax correctly
2. Add more operations to the Sharp dialect
3. Create comprehensive test suite
4. Implement dialect conversions and transformations
5. Develop hardware-specific optimizations

## Contributing

Before submitting code:
1. Run `pixi run format` to format code
2. Run `pixi run test` to ensure tests pass
3. Run `pixi run tidy` for static analysis

## Additional Resources

- [MLIR Documentation](https://mlir.llvm.org/)
- [CIRCT Documentation](https://circt.llvm.org/)
- [Sharp Documentation](./docs/README.md)

## Benefits of the Current Setup

1. **Faster Builds**: Skips unnecessary LLVM/MLIR/CIRCT rebuilds
2. **Cleaner Output**: Only shows relevant information
3. **Consistent Toolchain**: All compilation uses clang-20 and lld
4. **Better Developer Experience**: Less noise, more signal
5. **No Root Access Required**: Everything installs locally
6. **Reproducible Builds**: Fixed compiler versions ensure consistency