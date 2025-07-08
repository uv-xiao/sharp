# Sharp Test Documentation

## Test Organization

### Dialect Tests (`test/Dialect/Txn/`) - 24 tests
Core tests for Transaction dialect operations:
- **Essential**: basic.mlir, conflict-matrix.mlir, primitives.mlir, parametric-primitives.mlir, method-attributes.mlir, schedule-simple.mlir, multi-cycle.mlir, launch-basic.mlir
- **Removed redundant**: multi-cycle-static-launch.mlir, multi-cycle-dynamic-launch.mlir, multi-cycle-combined.mlir, primitive-usage.mlir, launch-simple.mlir

### Analysis Tests (`test/Analysis/`) - 14 tests
Analysis and optimization passes:
- **Essential**: conflict-matrix-inference.mlir, action-scheduling.mlir, combinational-loop-detection.mlir, pre-synthesis-check.mlir, reachability-analysis.mlir
- **Removed redundant**: action-scheduling-updated.mlir

### Conversion Tests (`test/Conversion/`) - 35 tests
Txn to FIRRTL/Func conversion:
- **TxnToFIRRTL**: basic.mlir, basic-module.mlir, complex-conflicts.mlir, will-fire-conflicts.mlir, verilog-export-checks.mlir
- **TxnToFunc**: basic-conversion.mlir, will-fire-simple.mlir, will-fire-abort.mlir, will-fire-guards.mlir
- **Removed redundant**: simple-verilog-export.mlir, simple-lowering.mlir, will-fire-abort-debug.mlir

### Simulation Tests (`test/Simulation/`) - 13 tests
Multiple simulation modes:
- **TL Simulation**: txn-simulate.mlir, counter-sim.mlir
- **RTL**: arcilator-basic.mlir, arcilator-simple.mlir  
- **JIT**: jit-basic.mlir, jit-simple.mlir
- **Concurrent**: concurrent-dam.mlir, concurrent-simple.mlir
- **Hybrid**: hybrid-bridge.mlir, hybrid-simple.mlir

## Test Execution

```bash
# Run all tests
pixi run test

# Run specific category
cd build && ../circt/llvm/llvm/utils/lit/lit.py test/Dialect/Txn/ -v

# Run single test
./build/bin/sharp-opt test/Dialect/Txn/basic.mlir
```

## Test Format

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

## Test Status

**Total: 93 tests**
- Passed: 81 (87.10%)
- Expectedly Failed: 1 (1.08%)
- Failed: 11 (11.83%)

**Consolidation Completed:**
- Removed 9 redundant test files
- Maintained comprehensive coverage
- Improved organization and clarity

## Coverage Gaps

1. TxnToFunc JIT compilation pipeline
2. PySharp runtime loading issues (7 failing tests)
3. Multi-cycle FIRRTL conversion
4. Some guard/control flow patterns

## Python Tests

Located in `test/python/`:
```bash
# Run Python tests
cd build && ../circt/llvm/llvm/utils/lit/lit.py test/python/ -v

# Direct execution
PYTHONPATH=build/python_packages python test/python/basic-import.py
```