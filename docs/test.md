# Sharp Test Suite Documentation

This document provides a comprehensive overview of Sharp's test suite, explaining the purpose of each test category and identifying which tests are essential for validating core functionality.

## Test Organization

Tests are organized into the following categories:

### 1. Dialect Tests (`test/Dialect/Txn/`)
Core tests for the Transaction (Txn) dialect operations and attributes.

**Essential Tests:**
- `basic.mlir` - Basic module and method operations
- `conflict-matrix.mlir` - Conflict matrix on schedule operations
- `primitives.mlir` - Primitive operations (Register, Wire, etc.)
- `parametric-primitives.mlir` - Parametric primitive instantiation
- `method-attributes.mlir` - Method timing and ready/enable attributes
- `schedule-simple.mlir` - Schedule operation syntax

**Can be removed/consolidated:**
- `simple-attributes.mlir` - Redundant with method-attributes.mlir
- `attributes.mlir` - Redundant basic attribute tests
- `fifo.mlir` - Placeholder for future FIFO primitive

### 2. Analysis Tests (`test/Analysis/`)
Tests for various analysis and optimization passes.

**Essential Tests:**
- `conflict-matrix-inference.mlir` - Infers missing conflict relationships
- `action-scheduling.mlir` - Automatic schedule completion
- `action-scheduling-heuristic.mlir` - Heuristic scheduling for large modules
- `combinational-loop-detection.mlir` - Detects combinational loops
- `pre-synthesis-check.mlir` - Checks for synthesizable constructs
- `reachability-analysis.mlir` - Computes reachability conditions
- `method-attribute-validation.mlir` - Validates signal name uniqueness

**Can be removed:**
- `dominance-issue-example.mlir` - Debugging artifact
- `pre-synthesis-check-ops.mlir` - Redundant with main test

### 3. Conversion Tests (`test/Conversion/TxnToFIRRTL/`)
Tests for Txn to FIRRTL conversion pass.

**Essential Tests:**
- `basic-module.mlir` - Basic module conversion
- `complex-conflicts.mlir` - Complex conflict matrix handling
- `will-fire-conflicts.mlir` - Will-fire signal generation
- `method-guards.mlir` - Method guard conditions
- `nested-modules.mlir` - Module hierarchy
- `parametric-instance-test.mlir` - Parametric instantiation
- `vector-types.mlir` - Vector type support
- `primitive-usage.mlir` - Primitive method calls
- `control-flow.mlir` - If/else control flow

**Can be consolidated:**
- Multiple conflict-inside tests can be merged
- Multiple will-fire tests can be merged

### 4. Simulation Tests (`test/Simulation/`)
Tests for simulation infrastructure.

**Currently Failing (need fixes):**
- All simulation tests are currently failing due to incomplete implementation

**Essential Tests to Fix:**
- `txn-simulate.mlir` - Basic TxnSimulatePass test
- `counter-sim.mlir` - Counter simulation example
- `multi-cycle.mlir` - Multi-cycle operation simulation
- `arcilator-simple.mlir` - Arcilator integration
- `hybrid-simple.mlir` - Hybrid simulation

### 5. Miscellaneous Tests (`test/Misc/`)
Tests for dialect coexistence and integration.

**Can be removed:**
- Both tests are minimal examples that don't add value

## Test Coverage Gaps

The following features lack comprehensive tests:

1. **TxnToFunc Conversion** - No tests for JIT compilation pipeline
2. **Concurrent Simulation** - No tests for DAM methodology implementation
3. **Hybrid Bridge** - Tests exist but need proper implementation
4. **PySharp** - No integration tests for Python frontend
5. **Verilog Export** - Limited end-to-end export tests
6. **Spec Primitives** - No tests for SpecFIFO, SpecMemory
7. **State Operations** - txn.state operation not implemented/tested

## Recommended Test Suite

### Core Dialect Tests (10 tests)
1. Module and method operations
2. Conflict matrix specification
3. Schedule operations
4. Primitive operations
5. Control flow (if/else)
6. Method attributes (timing, ready/enable)
7. Rule operations
8. Instance operations
9. Call operations
10. Yield/return operations

### Analysis Pass Tests (7 tests)
1. Conflict matrix inference
2. Action scheduling (optimal)
3. Action scheduling (heuristic)
4. Combinational loop detection
5. Pre-synthesis checking
6. Reachability analysis
7. Method attribute validation

### Conversion Tests (10 tests)
1. Basic module to FIRRTL
2. Complex conflicts
3. Method implementation
4. Control flow conversion
5. Primitive usage
6. Module hierarchy
7. Parametric types
8. Vector types
9. Will-fire generation
10. Verilog export pipeline

### Simulation Tests (8 tests)
1. Transaction-level simulation
2. Multi-cycle operations
3. Concurrent simulation
4. JIT compilation
5. RTL simulation (arcilator)
6. Hybrid TL-RTL bridge
7. Event-driven simulation
8. Performance benchmarks

### Integration Tests (5 tests)
1. End-to-end counter example
2. FIFO implementation
3. Pipeline example
4. State machine
5. PySharp integration

## Test Execution

Run all tests:
```bash
pixi run test
```

Run specific category:
```bash
lit test/Dialect/Txn -v
```

Run single test:
```bash
sharp-opt test/Dialect/Txn/basic.mlir
```

## Test Format

Tests use LLVM's FileCheck format:
```mlir
// RUN: sharp-opt %s | FileCheck %s

// CHECK-LABEL: txn.module @Example
txn.module @Example {
  // CHECK: txn.value_method @getValue
  txn.value_method @getValue() -> i32 {
    %c0 = arith.constant 0 : i32
    txn.return %c0 : i32
  }
}
```

## Adding New Tests

1. Create .mlir file in appropriate category
2. Add RUN line with appropriate passes
3. Add CHECK directives for expected output
4. Ensure test is focused on single feature
5. Include both positive and negative test cases
6. Document any special requirements

## Python Binding Tests

Python binding tests are located in `test/python/` and test both Sharp and PySharp functionality.

### Running Python Tests

```bash
# Run all Python tests via lit
pixi run test-lit test/python/

# Run specific Python test
pixi run python test/python/basic-import.py
```

### Python Test Structure

1. **Basic imports** (`basic-import.py`): Tests Sharp binding imports and dialect registration
2. **PySharp imports** (`pysharp-import.py`): Tests PySharp frontend imports
3. **Module creation** (`pysharp-module.py`): Tests PySharp module decorators

### Writing Python Tests

Python tests use the lit infrastructure with RUN lines:
```python
# RUN: %python %s

import sharp
from sharp import ir

# Test code here
```

The test environment automatically sets PYTHONPATH to include:
- `build/python_packages` (for Sharp bindings)
- `build/python_packages/pysharp` (for PySharp frontend)