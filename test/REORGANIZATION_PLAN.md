# Test Suite Reorganization Plan

## Current State (86 tests)
- Many trivial tests that just check basic syntax
- Redundant coverage (e.g., multiple primitive tests doing the same thing)
- Missing comprehensive integration tests
- Insufficient edge case coverage

## Proposed Structure (35-40 comprehensive tests)

### 1. Core Dialect Tests (8 tests)
- `core-comprehensive.mlir` - All basic operations in one file
- `conflict-matrix-advanced.mlir` - Complex conflict scenarios with all relation types
- `control-flow-edge-cases.mlir` - Nested ifs, aborts, complex guards
- `primitives-all.mlir` - All primitive types with parametric instantiation
- `multi-cycle-comprehensive.mlir` - Launch operations with all timing modes
- `reachability-complex.mlir` - Deep nesting with conditional aborts
- `schedule-violations.mlir` - Invalid schedules (negative test)
- `type-system-edge-cases.mlir` - Complex type interactions

### 2. Analysis Tests (6 tests)
- `analysis-integration.mlir` - Multiple passes working together
- `conflict-inference-advanced.mlir` - Transitive conflicts, method inheritance
- `loop-detection-nested.mlir` - Complex combinational loops
- `reachability-with-aborts.mlir` - Abort propagation through calls
- `pre-synthesis-errors.mlir` - Non-synthesizable constructs
- `scheduling-complex.mlir` - Large modules with many conflicts

### 3. Conversion Tests (10 tests)
- `txn-to-firrtl-complete.mlir` - All features in one module
- `will-fire-all-modes.mlir` - Static, dynamic, most-dynamic comparison
- `abort-propagation-full.mlir` - Abort through multiple call levels
- `conflict-inside-advanced.mlir` - Complex reachability conditions
- `txn-to-func-transactional.mlir` - Full transactional execution
- `primitive-lowering-all.mlir` - All primitive types to FIRRTL
- `guard-evaluation-complex.mlir` - Nested guards with side effects
- `instance-method-calls.mlir` - Complex instance hierarchies
- `conversion-errors.mlir` - Invalid conversions (negative test)
- `verilog-export-integration.mlir` - Full pipeline test

### 4. Simulation Tests (8 tests)
- `three-phase-execution.mlir` - Value/Execute/Commit phases
- `concurrent-dam-complex.mlir` - Multi-module with time sync
- `jit-performance.mlir` - JIT with complex computations
- `hybrid-tl-rtl.mlir` - Mixed simulation modes
- `multi-cycle-simulation.mlir` - Launch operations in simulation
- `abort-recovery.mlir` - Abort handling in simulation
- `event-ordering.mlir` - Complex event dependencies
- `simulation-errors.mlir` - Runtime errors (negative test)

### 5. Integration Tests (3-5 tests)
- `full-processor.mlir` - Simple processor with all features
- `fifo-network.mlir` - Producer-consumer network
- `cache-controller.mlir` - Complex state machine
- `cryptographic-core.mlir` - High-performance pipeline
- `error-recovery-system.mlir` - Comprehensive error handling

## Test Quality Guidelines
1. Each test should validate multiple related features
2. Include both positive and negative test cases
3. Test edge cases and error conditions
4. Use realistic, non-trivial examples
5. Include performance/stress tests where applicable
6. Document what each test is validating

## Migration Strategy
1. Merge similar tests into comprehensive ones
2. Remove purely syntactic tests
3. Add missing edge case coverage
4. Create integration tests that exercise full pipelines
5. Ensure each remaining test provides unique value