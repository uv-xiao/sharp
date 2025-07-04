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

#### Pythonic Construction Frontend - PySharp (2025-07-03)
- **Python Bindings Infrastructure**
  - Removed Sharp Core dialect as it was not useful
  - Updated SharpModule.cpp to register MLIR dialects (SCF, SMT, Index, Arith)
  - Updated SharpModule.cpp to register CIRCT dialects (FIRRTL, Comb, HWArith, Seq, SV)
  - Note: HW dialect registration removed to avoid conflicts with CIRCT's builtin dialect
  - Linked against required CAPI libraries for dialect access
  
- **PySharp Frontend Module**
  - Created comprehensive EDSL in `lib/Bindings/Python/pysharp.py`
  - Type system: IntType, BoolType, FIRRTLType (uint, sint, clock, reset)
  - Predefined types: i1, i8, i16, i32, i64, i128, i256
  - ConflictRelation enum matching Txn dialect (SB=0, SA=1, C=2, CF=3)
  - Value class with operator overloading for arithmetic/logic operations
  - ModuleBuilder API for constructing hardware modules
  - Support for states, value methods, action methods, and rules
  - Module decorator for class-based hardware description
  - Helper functions: constant, read, write, if_then_else
  
- **Current Status**
  - PySharp frontend module successfully implemented and tested
  - Standalone functionality verified without MLIR dependency
  - Python bindings can access MLIR/CIRCT dialects (with HW exception)
  - Native extension has runtime loading issues that need investigation

### 🚧 In Progress

#### Simulation at Arbitrary Level (2025-07-03)
- **Transaction-Level Simulation Core**
  - Implemented event-driven simulation engine with dependency tracking
  - Created Event, SimModule, and Simulator classes in C++
  - Support for multi-cycle operations through continuation events
  - Conflict matrix support with SB/SA/C/CF relations
  - Performance tracking and debugging infrastructure
  - Created comprehensive design document in `docs/simulation.md`
  - Unit tests for basic simulation functionality
  
- **Simulation Operations**
  - Defined SimulationOps.td with configuration and spec primitives
  - Support for multi-cycle spec operations
  - Random value generation for testbenches
  - Performance measurement regions
  
- **Test Infrastructure**
  - Created simulation test examples in MLIR format
  - Counter module with conflict handling
  - Pipeline example with multi-cycle processing
  - Hybrid simulation configuration examples

### 📋 Planned

- **Additional Primitives**
  - FIFO, Memory, and other common hardware primitives
  - Spec primitives for formal verification  
  
- **Performance Optimizations**
  - Optimize will-fire logic generation
  - Reduce redundant conflict checks
  - Implement dead code elimination

### 🚫 Known Limitations
- Python bindings native extension has runtime loading issues (ImportError)
- HW dialect cannot be registered from Sharp due to conflicts with CIRCT's builtin dialect
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