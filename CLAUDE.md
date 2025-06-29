# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sharp is an MLIR-based infrastructure that bridges hardware, architecture, and programming. It provides custom MLIR dialects for hardware description and integrates with CIRCT for RTL generation.

## Essential Commands

### Build Commands
```bash
pixi run build              # Standard quiet build
pixi run build-unified-verbose  # Verbose build with full output
pixi run build-release      # Release mode build
pixi run clean             # Clean build artifacts
```

### Testing Commands
```bash
pixi run test              # Run all tests
pixi run test-lit          # Run lit tests only
pixi run test-unit         # Run unit tests only
pixi run test-python       # Test Python bindings

# Run a single test file
./build/bin/llvm-lit test/Dialect/Txn/basic.mlir -v
```

### Development Commands
```bash
pixi run format            # Format C++ code with clang-format
pixi run tidy             # Run clang-tidy static analysis
pixi run shell            # Enter development shell with proper environment

# Run sharp-opt (the main compiler tool)
./build/bin/sharp-opt <input.mlir>
pixi run run-opt
```

## Architecture Overview

### Dialect Structure
Sharp implements custom MLIR dialects following CIRCT patterns:

1. **Sharp Core Dialect** (`include/sharp/Dialect/Core/`)
   - Basic infrastructure and common operations
   - Currently implements constant operations

2. **Sharp Txn Dialect** (`include/sharp/Dialect/Txn/`)
   - Transaction-based hardware description
   - Key concepts: Modules, Methods (value/action), Rules, Scheduling
   - Inspired by Fjfj language for concurrent hardware verification
   - See `docs/txn.md` for detailed semantics

### Key Implementation Patterns

**Adding New Operations:**
1. Define the operation in `include/sharp/Dialect/*/IR/*Ops.td`
2. Implement verifiers/builders in `lib/Dialect/*/IR/*Ops.cpp`
3. Add tests in `test/Dialect/*/`
4. Update Python bindings if needed in `lib/Bindings/Python/`

**Conversion Passes:**
- Conversions go in `lib/Conversion/`
- Follow the pattern: source dialect → target dialect
- Register in `lib/Conversion/PassDetail.h`

**Analysis Passes:**
- Implement in `lib/Analysis/`
- Register in `include/sharp/Analysis/Passes.td`

### Build System Details

- Uses CMake with Ninja generator
- Managed via Pixi package manager
- All dependencies (LLVM/MLIR/CIRCT) built from source using Clang-20
- Build artifacts in `.install/unified/`
- Sharp-specific builds in `build/`

### Testing Infrastructure

- Uses LLVM lit with FileCheck
- Integration tests in `test/` mirror the source structure
- Test files use `.mlir` extension with RUN lines
- Example test pattern:
  ```mlir
  // RUN: sharp-opt %s | FileCheck %s
  ```

### Current Development Focus

Per `PLAN.md`, active work includes:
- Conflict matrix (CM) implementation for scheduling
- Timing attributes for rules/methods
- Translation from txn dialect to FIRRTL/Verilog
- Multi-cycle rule support

## Known Issues

- Python bindings have a runtime issue under investigation
- Sharp Txn to FIRRTL conversion is in planning phase
- Multi-cycle operations not yet supported