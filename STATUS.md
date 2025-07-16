# Sharp Development Status

## Overview
Sharp is a transaction-based hardware description language with MLIR-based implementation, conflict matrix support, and FIRRTL/Verilog generation capabilities.

## Completed Features

### Core Infrastructure
- Sharp Txn dialect with modules, methods, rules, and scheduling
- Conflict matrix support (SB=0, SA=1, C=2, CF=3)
- Primitive infrastructure (Register, Wire, FIFO, Memory, SpecFIFO, SpecMemory)
- Testing: Reorganized to ~64 tests with ~40 comprehensive tests (currently 51/71 passing = 71.83%)

### Analysis Passes
- Conflict matrix inference
- Pre-synthesis validation
- Reachability analysis  
- Schedule validation

### Code Generation
- **Txn-to-FIRRTL**: Complete two-pass conversion with will-fire logic and conflict checking
  - Phase 1: LowerTxnBodyToFIRRTLPass - Converts all types to FIRRTL types
  - Phase 2: TranslateTxnToFIRRTLPass - Generates FIRRTL circuit structure
  - Comprehensive type conversion without UnrealizedConversionCastOp
  - Fixed use-after-free errors and architectural issues
- **Txn-to-Func**: Transactional execution with abort propagation
- **Verilog Export**: Via CIRCT pipeline (Txn → FIRRTL → HW → Verilog)

### Simulation Infrastructure
- **TL Simulation**: Event-driven with conflict resolution
- **RTL Simulation**: Via CIRCT's arcilator
- **JIT Compilation**: Direct execution via LLVM
- **Concurrent Simulation**: DAM methodology for parallel execution
- **Hybrid TL/RTL**: Bridge infrastructure for mixed simulation

### Python Frontend (PySharp)
- PyCDE-style module construction
- Type system with operator overloading  
- Conflict matrix management
- MLIR generation

### Execution Model
- Three-phase execution: Value → Execution → Commit
- Schedules contain only rules/actions (not value methods)
- Full abort propagation in transactional model
- Multi-cycle support via launch operations

### Documentation & Examples
- 8-chapter tutorial covering all features
- Comprehensive test suite
- API documentation for all passes

## Current Limitations
- Python bindings require runtime fixes
- Empty action bodies generate empty FIRRTL blocks
- Multi-cycle synthesis not yet supported
- Some spec primitives are placeholders

## Next Steps

### Recent Progress

1. **✅ Improved TxnToFIRRTL Will-Fire Logic** 
   - Implemented comprehensive `reach_abort` calculation
   - Updated ReachabilityAnalysis to handle abort operations
   - Added proper abort reachability tracking in all will-fire modes
   - Integrated ReachabilityAnalysis results into conversion context
   - Added explicit `conflict_inside` logic with reachability conditions

2. **✅ Implemented Three-Phase Execution Model**
   - Added Value Phase, Execution Phase, and Commit Phase separation
   - Created value method cache infrastructure for once-per-cycle calculation
   - Modified step() to execute all three phases properly
   - Added executeValuePhase(), executeEventPhase(), and executeCommitPhase() methods

3. **✅ Test Suite Reorganization**
   - Reduced from 86+ simple tests to ~40 comprehensive tests
   - Created tests that validate multiple features in realistic scenarios
   - Removed redundant and trivial test cases
   - Improved test documentation and organization

### Documentation vs Implementation Resolution

4. **✅ Resolved TxnToFIRRTL Issues**
   - Implemented `reach_abort` calculation in all will-fire modes
   - Added proper `conflict_inside` logic with reachability conditions
   - Integrated ReachabilityAnalysis pass results properly
   - All critical mismatches from `docs/txn_to_firrtl.md` resolved

5. **✅ Resolved Execution Model Issues**  
   - Implemented three-phase execution model (Value → Execution → Commit)
   - Added value method caching for once-per-cycle calculation
   - Proper phase separation in Simulator.cpp
   - Critical mismatches from `docs/execution_model.md` resolved

### Remaining High Priority Tasks

1. **Fix Test Suite Issues** (In Progress)
   - ✅ Fixed FIRRTL output format issues in some tests
   - ✅ Added basic launch operation support in TxnToFIRRTL
   - ✅ Fixed priority attribute syntax errors in rules
   - ✅ Fixed duplicate else clause issues
   - ✅ Fixed missing else branches in txn.if statements
   - ✅ Fixed malformed else branches (extra `} else {` blocks)
   - ✅ Fixed conflict matrix syntax errors (trailing commas)
   - ✅ Fixed control flow issues in reachability tests
   - ✅ Fixed more syntax errors (txn.if/else, trailing commas, token types)
   - ✅ Fixed test violations (actions calling actions)
   - ✅ Fixed dynamic mode dominance error (architectural issue)
   - ✅ Fixed txn.if missing terminator issues
   - ✅ Added FIFO primitive support to TxnToFIRRTL conversion
   - ✅ Added Memory primitive support to TxnToFIRRTL conversion
   - ✅ Fixed primitive FIRRTL module creation crashes (now fails gracefully)
   - ✅ Added proper error handling for primitive instantiation failures
   - Currently 19 failing tests (71.83% passing)
   - Remaining issues:
     - Memory/FIFO primitives return nullptr (need actual FIRRTL implementation)
     - Launch operations need complete implementation
     - Type mismatch issues (signedness in arithmetic operations)
     - Control flow syntax errors in complex tests
     - TxnToFunc conversion issues
     - Python binding tests: 7 tests failing (runtime issues)
2. **Complete Dynamic Mode Enhancements**
   - ✅ Fixed dominance error when updating will-fire nodes (SSA violation)
   - ✅ Implemented two-pass approach for will-fire signal generation
   - Add full recursive call tracking as described in docs
   - Implement proper conflict detection for indirect calls
   - Current implementation has reach_abort but lacks full capability
3. **Complete Multi-Cycle Operations** 
   - Implement launch operation handling in FIRRTL conversion
   - Add launch operation support in simulation (`lib/Simulation/`)
   - Add abort propagation for multi-cycle actions
   - Integrate launch until/after conditions
4. **Complete DAM implementation** with proper time synchronization (`lib/Simulation/Concurrent/`)

## Future Work

### Analysis and Validation
- **Combinational Loop Detection**: Robust cycle detection through combinational paths in Wire networks and value method dependencies

### Secondary Issues (Medium Priority)

5. **Fix Python binding runtime issues** (7 failing tests)
6. **Add state operation support**
7. **Enhance IDE integration and tooling support**