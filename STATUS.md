# Sharp Development Status

## Overview
Sharp is a transaction-based hardware description language with MLIR-based implementation, conflict matrix support, and FIRRTL/Verilog generation capabilities.

## Completed Features

### Core Infrastructure
- Sharp Txn dialect with modules, methods, rules, and scheduling
- Conflict matrix support (SB=0, SA=1, C=2, CF=3)
- Primitive infrastructure (Register, Wire, FIFO, Memory, SpecFIFO, SpecMemory)
- Testing: 81/93 tests passing (87.10%)

### Analysis Passes
- Conflict matrix inference
- Pre-synthesis validation
- Reachability analysis  
- Combinational loop detection
- Schedule validation

### Code Generation
- **Txn-to-FIRRTL**: Complete with will-fire logic and conflict checking
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

### Remaining High Priority Tasks

1. **Complete Multi-Cycle Operations** 
   - See `docs/execution_model.md` for more details.
   - Implement launch operation handling in simulation (`lib/Simulation/`)
   - Add abort propagation for multi-cycle actions
   - Integrate launch until/after conditions with simulation infrastructure

2. **Enhance Analysis Pass Integration**
   - Fix ReachabilityAnalysis condition integration in TxnToFIRRTL conversion
   - Ensure ConflictMatrixInference results are properly used in will-fire logic
   - Add primitive-level conflict matrix integration

### Secondary Issues (Medium Priority)

5. **Fix Python binding runtime issues** (7 failing tests)
6. **Complete DAM implementation** with proper time synchronization (`lib/Simulation/Concurrent/`)
7. **Add state operation support**
8. **Enhance IDE integration and tooling support**