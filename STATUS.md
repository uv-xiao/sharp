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

### 🚧 In Progress

- **Txn-to-FIRRTL Conversion Pass**
  - [x] Design conversion architecture following Koika approach (docs/txn_to_firrtl.md)
  - [x] Implement required analysis passes (reachability, loop detection, validation)
  - [ ] Implement basic module structure translation
  - [ ] Add will-fire logic generation with CM support

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
8. Implement basic txn-to-FIRRTL conversion pass
9. Add will-fire logic generation with CM support