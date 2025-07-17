# Sharp Test Infrastructure

## Overview
The Sharp test suite uses LLVM's lit testing framework to validate all components. Tests are organized into comprehensive files that exercise multiple related features together.

## Test Organization

### Directory Structure
```
test/
├── Dialect/           # Core dialect tests
│   ├── Core/         # Core dialect operations
│   └── Txn/          # Transaction dialect tests (8 tests)
├── Analysis/         # Analysis pass tests (6 tests)
├── Conversion/       # Conversion pass tests
│   ├── TxnToFIRRTL/  # Txn to FIRRTL conversion (10 tests)
│   └── TxnToFunc/    # Txn to Func conversion (8 tests)
├── Simulation/       # Simulation tests (8 tests)
└── python/           # Python binding tests
```

### Test Categories (~40 comprehensive tests)

#### Core Dialect Tests
- **core-comprehensive.mlir**: All basic Txn operations in realistic scenarios
- **conflict-matrix-advanced.mlir**: Complex conflict scenarios with inference
- **control-flow-edge-cases.mlir**: Nested control flow, aborts, complex guards
- **primitives-all.mlir**: All primitive types with parametric instantiation
- **multi-cycle-comprehensive.mlir**: Launch operations with all timing modes
- **reachability-complex.mlir**: Deep nesting with conditional aborts

#### Analysis Tests
- **analysis-integration.mlir**: Multiple analysis passes working together
- **conflict-inference-advanced.mlir**: Transitive conflicts
- **pre-synthesis-check.mlir**: Non-synthesizable constructs

#### Conversion Tests
- **txn-to-firrtl-complete.mlir**: All TxnToFIRRTL features
- **will-fire-all-modes.mlir**: Static and dynamic modes
- **abort-propagation-full.mlir**: Abort through multiple levels
- **txn-to-func-transactional.mlir**: Full transactional execution

#### Simulation Tests
- **three-phase-execution.mlir**: Value/Execute/Commit phases
- **concurrent-dam.mlir**: Multi-module concurrent simulation
- **jit-simulation.mlir**: JIT compilation and execution
- **hybrid-tl-rtl.mlir**: Mixed simulation modes

## Running Tests

### All Tests
```bash
pixi run test              # Run all tests
pixi run test-lit          # Run lit tests only
```

### Individual Tests
```bash
# From build directory
cd build && ../circt/llvm/llvm/utils/lit/lit.py test/Dialect/Txn/core-comprehensive.mlir -v

# Run all tests in a directory
cd build && ../circt/llvm/llvm/utils/lit/lit.py test/Analysis/ -v

# Filter tests by pattern
cd build && ../circt/llvm/llvm/utils/lit/lit.py test/ -v --filter "abort"
```

### Debug Test Failures
```bash
# Run sharp-opt directly
./build/bin/sharp-opt test/Dialect/Txn/conflict-matrix-advanced.mlir --sharp-infer-conflict-matrix

# With error diagnostics
./build/bin/sharp-opt test/Analysis/pre-synthesis-errors.mlir --verify-diagnostics
```

## Test Patterns

### Basic Test Structure
```mlir
// RUN: sharp-opt %s | FileCheck %s
// RUN: not sharp-opt %s --verify-diagnostics 2>&1 | FileCheck %s --check-prefix=ERROR

// CHECK-LABEL: txn.module @TestModule
txn.module @TestModule {
  // Test implementation
}
```

### Multiple Check Prefixes
```mlir
// RUN: sharp-opt %s --convert-txn-to-firrtl | FileCheck %s
// RUN: sharp-opt %s --convert-txn-to-firrtl=mode=static | FileCheck %s --check-prefix=STATIC
```

### Testing Analysis Passes
```mlir
// RUN: sharp-opt %s --sharp-infer-conflict-matrix | FileCheck %s

// CHECK: conflict_matrix = {
// CHECK-DAG: "a,b" = 2 : i32
```

## Writing Comprehensive Tests

### Guidelines
1. Each test validates multiple related features
2. Include edge cases and error conditions
3. Use realistic, meaningful scenarios
4. Document what each section tests
5. Include both positive and negative cases

### Example Pattern
```mlir
txn.module @ProcessorPipeline {
  // Multiple primitive types
  %pc = txn.instance @pc of @Register<i32> : index
  
  // Complex control flow with aborts
  txn.action_method @execute(%enable: i1) {
    // Multiple abort paths
    // Nested conditionals
  }
  
  // Complete conflict matrix
  txn.schedule [@execute] {
    conflict_matrix = { /* all relationships */ }
  }
}
```

## Test Quality
- **Coverage**: Every feature has comprehensive tests
- **Complexity**: Tests exercise non-trivial scenarios
- **Maintainability**: Self-documenting and modular
- **Performance**: Reasonable execution time
- **Reliability**: Catches regressions effectively