# Sharp Development Status

This document tracks the implementation progress of features outlined in PLAN.md.

## Overview

Sharp is a transaction-based hardware description language with conflict matrix support and FIRRTL/Verilog generation capabilities. The project implements custom MLIR dialects for hardware description using transaction-level modeling inspired by Bluespec and Koika.

## Feature Status

### ✅ Completed

#### Core Infrastructure
- Basic Sharp Core dialect with constant operations
- Sharp Txn dialect with modules, methods, rules, and scheduling
- MLIR infrastructure setup with CIRCT integration
- Build system with Pixi package manager
- Testing infrastructure with lit/FileCheck (48/48 tests passing)

#### Txn Dialect Features (2025-06-29)
- **Conflict Matrix (CM) on schedule operations**
  - Added CM dictionary attribute to txn.schedule operations
  - Uses ConflictRelation enum: SB=0, SA=1, C=2, CF=3
  - Supports action-to-action conflict specifications
- **Timing attributes for rules/methods**
  - Added timing string attribute: "combinational" | "static(n)" | "dynamic"
  - Integrated into rule and method operations
- **Primitive Operations and Infrastructure**
  - Added FirValueMethodOp, FirActionMethodOp, ClockByOp, ResetByOp operations
  - Implemented Register and Wire primitive constructors
  - Separated txn primitive interface from FIRRTL implementation
  - Added bridging attributes (firrtl.port, firrtl.data_port, firrtl.enable_port)

#### Analysis Passes (2025-06-29 to 2025-06-30)
- **Conflict Matrix Inference Pass**
  - Implemented as analysis pass in `lib/Analysis/ConflictMatrixInference.cpp`
  - Supports all inference rules from PLAN.md
  - Uses StringMap for efficient conflict storage
- **Pre-synthesis Checking Analysis**
  - Detects non-synthesizable (`spec`) primitives
  - Verifies no multi-cycle rules/methods
  - Validates operations are from allowed dialects
- **Reachability Analysis**
  - Computes reachability conditions for method calls within actions
  - Tracks control flow through txn.if operations
  - Generates hardware values using arith operations
- **Method Attribute Validation**
  - Validates signal name uniqueness for FIRRTL translation
  - Checks always_ready/always_enable attribute constraints

#### Txn-to-FIRRTL Conversion (2025-06-30 to 2025-07-01)
- **Complete Conversion Pass Implementation**
  - FIRRTL circuit generation with proper module hierarchy
  - Module ports for clock, reset, and method interfaces
  - Will-fire signals with conflict matrix checking
  - Ready signals for action methods based on conflicts
  - Submodule instantiation and port connections
  - Method call translation to FIRRTL connections
  - Type conversion supporting integers of any width and vectors
  
- **Advanced Features**
  - **Conflict Inside Detection**: Detects and prevents internal conflicts within actions
  - **Static and Dynamic Will-Fire Modes**: Two modes for conflict resolution logic
  - **Enhanced CallOp**: Support for conditional method calls with reachability
  - **Block Argument Handling**: Proper conversion of method arguments to FIRRTL ports
  
- **Automatic Primitive Construction with Parametric Typing** (2025-07-01)
  - Primitives (Register, Wire) created on-demand when referenced
  - Proper parametric typing support: `@instance of @Module<type1, type2>`
  - Generates unique FIRRTL modules for each type instantiation
  - Fixed circuit naming to identify true top-level modules
  - Complete test coverage with all 45 tests passing

#### Action Scheduling Algorithm (2025-07-02)
- **Complete Implementation of Automatic Schedule Completion**
  - Analysis pass in `lib/Analysis/ActionScheduling.cpp`
  - Completes partial schedules while preserving specified orderings
  - Minimizes conflicts where `action1 SB action2 && action1 > action2`
  - Two-phase approach: optimal algorithm for ≤10 actions, heuristic for larger modules
  - Detects and reports cyclic dependencies
  - Full test coverage with 48 tests passing (including 3 new test files)

#### Verilog Export (2025-07-02)
- **Complete Integration with CIRCT's Export Infrastructure**
  - Added custom pipelines: `--txn-export-verilog` and `--lower-to-hw`
  - Full conversion pipeline: Txn → FIRRTL → HW → Verilog
  - Proper handling of clock/reset signals and method interfaces
  - Support for both single-file and split-file export modes
  - Created comprehensive documentation in `docs/verilog_export.md`
  - Added tests demonstrating Verilog generation capabilities
  - Current limitation: Empty action bodies generate empty FIRRTL when blocks

#### Pythonic Construction Frontend (2025-07-02)
- **Complete Python API for Hardware Construction**
  - Decorator-based module definition with `@module` and `ModuleBuilder`
  - Type-safe hardware types (i8, i16, i32, i64, etc.) with automatic MLIR conversion
  - Operator overloading for arithmetic, logic, and comparison operations
  - Method decorators: `@value_method`, `@action_method`, `@rule`
  - Conflict matrix management with `ConflictRelation` enum
  - Builder API with fine-grained control over operations
  - Comprehensive documentation in `docs/pythonic_frontend.md`
  - Integration with MLIR Python bindings infrastructure
  - Example modules demonstrating typical hardware design patterns

#### Txn-level Combinational Loop Detection (2025-07-02)
- **Complete Loop Detection Analysis Pass**
  - Implemented `--sharp-detect-combinational-loops` pass
  - Builds dependency graph of combinational signal paths
  - Uses depth-first search to detect cycles in the graph
  - Distinguishes combinational vs sequential primitives (Wire vs Register)
  - Reports detailed error messages with complete cycle paths
  - Supports custom primitive combinational path attributes
  - Algorithm based on signal flow analysis from TxnToFIRRTL conversion
  - Comprehensive test suite with positive and negative test cases
  - Documentation in `docs/combinational_loop_detection.md`
  - Integration with Sharp's analysis pass pipeline

### 📋 Planned

- **Simulation at Arbitrary Level**
  - Transactional: event-driven, both spec/hw modules/primitives
    - spec actions can be multi-cycle: they have a sequence of behavior (trigger other actions) to be finished in a specific sequence of time steps.
    - learn from EQueue (references/Li-2022-EQueue.pdf) and DAM (references/Zhang-2024-DAM.pdf, https://github.com/stanford-ppl/DAM-RS)
    - try to be high-performance (concurrent simulation)
  - RTL: use CIRCT's `arcilator`
  - Testbench: we can specify several instances to server as both a testbench generator and a result checker, and others to be tested against them.
  - Hybrid: can we specify several instances to be simulated at transaction-level, and others at RTL-level? They should be able to communicate with each other.

- **Additional Primitives**
  - FIFO, Memory, and other common hardware primitives
  - Spec primitives for formal verification  
  
- **Performance Optimizations**
  - Optimize will-fire logic generation
  - Reduce redundant conflict checks
  - Implement dead code elimination

### 🚫 Known Limitations
- Python bindings have runtime issues
- Multi-cycle operations not yet supported in translation
- Nonsynthesizable primitives will fail translation

## Next Steps

1. **Implement Additional Primitives**
   - Create FIFO primitive with enqueue/dequeue methods
   - Add Memory primitive with read/write ports
   - Design spec primitives for verification
   
2. **Enhanced Analysis**
   - Implement combinational loop detection
   - Add performance analysis passes
   - Create resource utilization estimates
   
3. **Fix Empty When Block Issue**
   - Update TxnToFIRRTL conversion to avoid empty when regions
   - Ensure all action bodies generate valid FIRRTL
   
4. **Tooling and Integration**
   - Fix Python bindings for programmatic access
   - Create VSCode/IDE language support
   - Add debugging and visualization tools