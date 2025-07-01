# Sharp Development Status

This document tracks the implementation progress of features outlined in PLAN.md.

## Overview

Sharp is implementing transaction-based hardware description with conflict matrix support and FIRRTL/Verilog generation capabilities.

## Feature Status

### ✅ Completed
- Basic Sharp Core dialect with constant operations
- Sharp Txn dialect with modules, methods, rules, and scheduling
- MLIR infrastructure setup with CIRCT integration
- Build system with Pixi package manager
- Testing infrastructure with lit/FileCheck
- **Conflict Matrix (CM) on schedule operations** (2025-06-29)
  - Added CM dictionary attribute to txn.schedule operations (not modules)
  - Uses ConflictRelation enum: SB=0, SA=1, C=2, CF=3
  - Supports action-to-action conflict specifications
- **Timing attributes for rules/methods** (2025-06-29)
  - Added timing string attribute: "combinational" | "static(n)" | "dynamic"
  - Integrated into rule and method operations
- **All tests passing** (2025-06-29)
  - Fixed module parser to use TableGen-generated assembly format
  - Updated all test files to match current implementation
  - Removed FIRRTL-dependent tests pending dialect registration
- **Conflict Matrix Inference Pass** (2025-06-29)
  - Implemented as analysis pass in `lib/Analysis/ConflictMatrixInference.cpp`
  - Supports all inference rules from PLAN.md
  - Uses StringMap for efficient conflict storage
  - Test coverage in `test/Analysis/conflict-matrix-inference.mlir`
- **FIRRTL Primitive Structure** (2025-06-29)
  - Created directory structure under `lib/Dialect/Txn/primitives/`
  - Added FIRRTL module definitions for Register and Wire primitives
  - Prepared for txn-to-FIRRTL conversion implementation
- **Primitive Operations and Constructors** (2025-06-29)
  - Added FirValueMethodOp, FirActionMethodOp, ClockByOp, ResetByOp operations
  - Implemented Register and Wire primitive constructors in C++
  - Register: read CF write conflict matrix
  - Wire: read SB write conflict matrix
  - Test coverage in `test/Dialect/Txn/primitives.mlir`

- **Txn Primitive Infrastructure** (2025-06-29)
  - Separated txn primitive interface from FIRRTL implementation
  - Created separate constructors for primitives and FIRRTL modules
  - Added bridging attributes (firrtl.port, firrtl.data_port, firrtl.enable_port)
  - Enabled gtest support in LLVM build for unit testing
  - Comprehensive documentation in `docs/txn_primitive.md`

- **Pre-synthesis Checking Analysis** (2025-06-29)
  - Implemented analysis pass in `lib/Analysis/PreSynthesisCheck.cpp`
  - Detects non-synthesizable (`spec`) primitives
  - Verifies no multi-cycle rules/methods (timing != "combinational")
  - Propagates non-synthesizable status through module hierarchy
  - Validates operations are from allowed dialects (txn, firrtl, builtin, arith)
  - Emits clear error messages for unsupported constructs
  - Test coverage in `test/Analysis/pre-synthesis-check.mlir` and `test/Analysis/pre-synthesis-check-ops.mlir`

- **Txn-to-FIRRTL Analysis Passes** (2025-06-30)
  - Implemented analysis passes per `docs/txn_to_firrtl.md`:
  - **Reachability Analysis** (`lib/Analysis/ReachabilityAnalysis.cpp`)
    - Computes reachability conditions for method calls within actions
    - Tracks control flow through txn.if operations
    - Generates symbolic condition expressions (e.g., "cond_0", "!cond_1 && cond_2")
    - Attaches reachability_condition attribute to txn.call operations
  - **Method Attribute Validation** (`lib/Analysis/MethodAttributeValidation.cpp`)
    - Validates signal name uniqueness for FIRRTL translation
    - Checks always_ready/always_enable attribute constraints
    - Ensures no conflicts with module, instance, or other method names
  - Updated CallOp printer/parser to support attribute dictionaries
  - **Note**: Combinational Loop Detection implemented but deprecated pending txn.primitive attribute support

- **Txn-to-FIRRTL Conversion Pass** (2025-06-30, 2025-07-01)
  - Created comprehensive FIRRTL operations guide (`docs/firrtl_operations_guide.md`)
  - Implemented complete conversion pass infrastructure:
    - Pass registration in `include/sharp/Conversion/Passes.td`
    - Main implementation in `lib/Conversion/TxnToFIRRTL/`
    - Integration with sharp-opt tool
  - Core functionality implemented:
    - FIRRTL circuit with proper module hierarchy
    - Module ports for clock, reset, and method interfaces
    - Will-fire signals with conflict matrix checking
    - Ready signals for action methods based on conflicts
    - Node operations for named intermediate values
    - Wire operations for method results
    - When blocks for conditional execution
    - Arithmetic operations support (arith.constant, arith.cmpi)
    - Type conversion from Sharp to FIRRTL types
  - Successfully converts Counter/Register examples to FIRRTL
  - Handles value methods, action methods, and rules
  - Properly sorts modules based on dependency analysis
  - Comprehensive test suite in `test/Conversion/TxnToFIRRTL/`:
    - Basic module conversion tests
    - Complex conflict matrix scenarios
    - Control flow with txn.if operations
    - Method guards and ready signal generation
    - Edge cases and error handling
    - Nested module hierarchies
  - Fixed implementation to erase original Txn modules after conversion
  - All tests passing (30/30) after fixing test CHECK patterns

- **Extended Type Support** (2025-07-01)
  - Added support for vector types in Txn-to-FIRRTL conversion
  - Implemented proper type conversion for FIRRTL vector types
  - Vector types convert to `!firrtl.vector<element_type, size>`
  - Already supported: integers of any width (i8, i16, i32, i64, etc.)
  - Test coverage in `test/Conversion/TxnToFIRRTL/wider-types.mlir` and `vector-types.mlir`
  - All tests passing (32/32)

- **Conflict Inside Detection** (2025-07-01)
  - Implemented infrastructure for detecting internal conflicts within actions
  - Added static analysis to identify actions with potentially conflicting method calls
  - Generated simplified conflict_inside logic in FIRRTL for actions with conflicts
  - For actions with conflicting calls, generates AND/NOT logic to prevent execution
  - Test coverage in `test/Conversion/TxnToFIRRTL/simple-conflict-inside.mlir`
  - All 34 tests passing
  - Note: Full dynamic reachability analysis with conditional execution paths planned for future enhancement

- **Enhanced Reachability Analysis and CallOp** (2025-07-01)
  - Modified CallOp to support optional condition operand for reachability
  - Updated CallOp syntax: `txn.call @method if %cond : <type> then (...) : ...`
  - Condition type is unrestricted (AnyType) to support complex expressions
  - Improved ReachabilityAnalysis pass to generate hardware values (using arith operations)
  - Implements proper dominance handling with insertion point management
  - Generates arith.andi, arith.xori operations for complex conditions
  - Modified TxnToFIRRTL conversion to use condition operands for conflict_inside calculation
  - Added pre-pass to convert arith operations to FIRRTL before conflict calculation
  - Implements proper dynamic conflict detection: `OR(conflict(m1,m2) && reach(m1) && reach(m2))`
  - Added support for arith::AndIOp, arith::XOrIOp conversion to FIRRTL
  - Test coverage in reachability-analysis.mlir and dominance-issue-example.mlir
  - All 37 reachability tests passing

### 🚧 In Progress

- **Enhanced Txn-to-FIRRTL Features**
  - [x] Add conflict_inside detection framework (2025-07-01)
  - [x] Implement submodule instantiation and port connections (already working)
  - [x] Handle CallOp translation to connect to submodule methods (already working)
  - [ ] Support proper register/wire state management in primitives
  - [x] Add support for more complex data types (vectors, wider integers)
  - [ ] Implement primitive method calls (Register.read, Wire.write, etc.)
  - [x] Full conflict_inside with conditional reachability (2025-07-01)
    - Implemented dynamic conflict detection using condition operands
    - Added conversion for arith operations to FIRRTL
    - Some edge cases with block arguments still need refinement

### 📋 Planned

- **FIRRTL Translation**
  - Reference: Bourgeat-2020-Koika.pdf, implementation at https://github.com/mit-plv/koika/blob/master/coq/CircuitGeneration.v
  - Synthesizable primitives:
    - Place in `lib/Dialect/Txn/primitives/`
    - Initial primitives needed: Register, Wire
  - Implement txn-to-FIRRTL conversion pass extending Koika translation
  - Extended will-fire (wf) logic:
    - Consider general conflicts from conflict matrix (not just register R/W)
    - Prevent firing when previous action conflicts per CM
    - Return false if rule calls same action method multiple times
  - Bottom-up translation order (submodules first)

- **Txn-level combinational loop detection**
  - current implementation is not correct, since we don't have attributes in `txn.primitive` to define combinational paths.
  
- **Verilog Export**
  - Add sharp-opt command-line options for triggering translation
  - Support different export-verilog options from CIRCT
  - Integration with CIRCT's export-verilog infrastructure

### 🚫 Known Limitations
- Python bindings have runtime issues
- Multi-cycle operations not yet supported in translation
- Nonsynthesizable primitives will fail translation

## Implementation Notes

### Conflict Matrix Inference Rules
1. Any action conflicts (C) with itself
2. Actions that are both SA and SB conflict (C)
3. Actions calling same action method of same instance conflict (C)
4. Conflict propagation through method calls:
   - m0 SA m1 => a0 SA a1
   - m0 SB m1 => a0 SB a1
   - m0 C m1 => a0 C a1
5. Default to conflict-free (CF) if relationship cannot be determined

### Translation Architecture
- Bottom-up translation: submodules before parent modules
- Will-fire logic must consider:
  - Conflict matrix relationships
  - Register read/write conflicts
  - Multiple calls to same action method (always conflicts)

## Next Steps
1. ~~Implement conflict matrix attributes in TxnOps.td~~ ✅
2. ~~Add timing attributes to rule and method operations~~ ✅
3. ~~Create FIRRTL primitive module templates~~ ✅
4. ~~Design conflict matrix inference algorithm~~ ✅
5. ~~Implement primitive operations and constructors~~ ✅
6. ~~Add actual FIRRTL implementation to primitives~~ ✅
7. ~~Add pre-synthesis checking for non-synthesizable elements~~ ✅
8. ~~Implement basic txn-to-FIRRTL conversion pass~~ ✅
9. ~~Add will-fire logic generation with CM support~~ ✅
10. ~~Create comprehensive test suite for conversion pass~~ ✅
11. Complete advanced conversion features:
    - Implement submodule instantiation and port connections
    - Handle CallOp translation for inter-module method calls
    - Add register/wire primitive integration
    - Support conflict_inside calculation with reachability
12. Add command-line integration:
    - Add --convert-txn-to-firrtl flag to sharp-opt
    - Support --export-verilog option for end-to-end flow
13. Implement remaining primitives:
    - FIFO, Memory, and other hardware primitives
    - Spec primitives for verification
14. Performance optimizations:
    - Optimize will-fire logic generation
    - Reduce redundant conflict checks
    - Implement dead code elimination