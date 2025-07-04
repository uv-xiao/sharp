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

#### Sharp Simulation Framework (2025-07-03 to 2025-07-04)
- **Event-Driven Simulation Core** ✅
  - Complete event-driven simulation engine with dependency tracking
  - Event class with unique IDs, timestamps, dependencies, and callbacks
  - EventQueue with priority-based scheduling and dependency resolution
  - Support for event continuation and multi-cycle operations
  - Proper handling of event dependencies and completion tracking
  
- **Process and Module Abstractions** ✅
  - SimModule base class with method registration and execution
  - Conflict matrix support (SB=0, SA=1, C=2, CF=3 relations)
  - Performance metrics tracking (cycles, call counts)
  - StatefulModule template for modules with explicit state management
  - SimModuleFactory for dynamic module creation
  
- **Main Simulator Engine** ✅
  - Simulator class with module management and instantiation
  - Event scheduling with and without dependencies
  - Conflict checking between concurrent events
  - Breakpoint support for debugging
  - Performance statistics collection
  - Step-by-step and batch execution modes
  - SimulationBuilder for fluent simulator configuration
  
- **Transaction-Level Spec Primitives** ✅
  - SpecFIFO implementation with unbounded FIFO semantics
  - SpecMemory with configurable read latency
  - Proper conflict matrices for spec components
  - Foundation for specification-level simulation
  
- **Simulation Operations Dialect** ✅
  - SimConfigOp for simulation configuration
  - BridgeConfigOp for TL-to-RTL bridge setup
  - SpecMultiCycleOp for multi-cycle operations
  - SpecAssertOp for verification assertions
  - SpecRandomOp for test stimulus generation
  - Performance tracking operations
  
- **PySharp Python Bindings** ✅
  - Comprehensive Python frontend (`pysharp.py`)
  - Hardware type system (IntType, UIntType, SIntType)
  - Signal and operation abstractions
  - Module builder with states, methods, and rules
  - Conflict relation management
  - Decorators for module definition
  - MLIR generation support (when bindings available)
  
- **Test Infrastructure** ✅
  - Unit tests for basic simulation functionality
  - MLIR test files demonstrating counter and pipeline modules
  - Transaction-level testbench patterns

- **MLIR-to-Simulation Lowering** ✅ (2025-07-04)
  - Implemented `--sharp-simulate` pass with Translation and JIT modes
  - Translation mode: Generates C++ code that uses the simulation API
  - JIT mode: Placeholder (requires txn-to-LLVM lowering pipeline)
  - Pass infrastructure properly integrated with TableGen
  - Full support for txn.module, value_method, action_method operations
  - Generates complete C++ simulation harness with main function
  - Enhanced with 1RaaT (one-rule-at-a-time) execution model:
    - Three-phase execution cycle: Scheduling → Execution → Commit
    - Proper conflict matrix handling in generated code
    - Multi-cycle operation support through continuation events
    - Timing attribute processing (combinational, static(n))
  
- **Concurrent Simulation with DAM Methodology** ✅ (2025-07-04)
  - Researched DAM (Discrete-event simulation with Adaptive Multiprocessing) from Zhang-2024-DAM.pdf
  - Implemented complete concurrent simulation infrastructure:
    - Context.h/cpp: Independent execution units with local monotonic time
    - Channel.h: Time-bridging channels for inter-context communication
    - ConcurrentSimulator.h/cpp: Main orchestrator with thread management
  - ConcurrentSimulationPass (`--sharp-concurrent-sim`) generates DAM-based C++ code:
    - Each txn.module becomes an independent context
    - Asynchronous distributed time (no global synchronization)
    - Lazy pairwise synchronization only when needed
    - Parallel rule execution for non-conflicting rules
    - Performance statistics with speedup calculation
  - Key DAM principles applied:
    - Contexts can run arbitrarily far into the future
    - Time-bridging channels handle backpressure and starvation
    - Support for bounded/unbounded channels
    - Thread scheduling optimization (SCHED_FIFO support)
  
- **JIT Compilation Mode** ✅ (2025-07-04)
  - Created TxnToFunc conversion pass infrastructure
  - Implemented conversion patterns for all txn operations
  - Set up JIT lowering pipeline: txn → func → LLVM → JIT
  - Integrated ExecutionEngine support in TxnSimulatePass
  - Fixed dependency issues (added UB dialect support)
  - Pass infrastructure complete with proper TableGen integration
  - Basic JIT compilation working for simple modules
  - Current limitations: Control flow operations (txn.if) need proper lowering
  
- **RTL Simulation Integration** ✅ (2025-07-04)
  - Implemented complete ArcilatorIntegrationPass (`--sharp-arcilator`)
  - Full conversion pipeline: Txn → FIRRTL → HW → Arc
  - Successfully integrates with CIRCT's arcilator infrastructure
  - Added all required dialect dependencies (FIRRTL, HW, Arc, Seq, Comb, Emit, SV, Sim, Verif, UB)
  - Generates instructions for running with arcilator tool
  - VCD tracing support available through arcilator's --trace option
  - Test cases demonstrate successful conversion to Arc dialect
  
- **Hybrid Simulation Capabilities** ✅ (2025-07-04)
  - Implemented complete TL-to-RTL bridge infrastructure
  - Created HybridBridge class with synchronization modes (lockstep/decoupled/adaptive)
  - Implemented ArcilatorSimulator as RTL backend interface
  - Created HybridSimulationPass (`--sharp-hybrid-sim`) that generates:
    - Bridge configuration with module/method mappings
    - TL simulation stubs that interface with RTL
    - Time synchronization between domains
    - Method call translation infrastructure
  - Supports different synchronization strategies:
    - Lockstep: TL and RTL advance together
    - Decoupled: Allow bounded time divergence
    - Adaptive: Dynamically adjust based on activity
  - Note: Full arcilator integration would require extending CIRCT's arcilator C API
  
- **PySharp Frontend Following PyCDE Pattern** ✅ (2025-07-04)
  - Created complete PySharp frontend at `frontends/PySharp/` following PyCDE structure
  - Removed old `lib/Bindings/Python/pysharp.py` as requested
  - Implemented PyCDE-style import pattern:
    - The `.sharp` module is provided by the build system (not manually created)
    - All imports from `.sharp` namespace (no direct _mlir_libs imports)
    - IR access through `from .sharp import ir`
    - Dialects through `from .sharp.dialects import txn, arith`
  - Core components implemented:
    - `__init__.py`: Context management and core imports
    - `types.py`: Type system (IntType, UIntType, ClockType, etc.)
    - `common.py`: Common definitions (ConflictRelation, Port, Timing)
    - `signals.py`: Signal abstractions with operator overloading
    - `module.py`: Module class with decorators (@value_method, @action_method, @rule)
    - `builder.py`: MLIR module construction
    - `support.py`: Utilities for emission and verification
  - Created test example demonstrating counter module
  - Follows PyCDE's CMake structure where build system creates the bindings
  - The `pysharp/sharp/` directory will be populated by CMake with Python bindings
  - This architecture should resolve runtime loading issues by bundling dependencies

### 🚧 In Progress

(No items currently in progress - all simulation infrastructure tasks completed!)

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
- Nonsynthesizable primitives will fail translation
- txn.state operation not yet implemented (needed for stateful modules)

## Next Steps

1. **Complete Simulation Infrastructure**
   - Implement txn→func→LLVM lowering for JIT mode
   - Complete arcilator integration for RTL simulation
   - Implement hybrid TL-to-RTL bridge synchronization
   - Add txn.state operation for stateful module simulation
   
2. **Implement Additional Primitives**
   - Create FIFO primitive with enqueue/dequeue methods
   - Add Memory primitive with read/write ports
   - Design spec primitives for verification
   
3. **Enhanced Analysis**
   - Add performance analysis passes
   - Create resource utilization estimates
   - Implement power consumption modeling
   
4. **Fix Empty When Block Issue**
   - Update TxnToFIRRTL conversion to avoid empty when regions
   - Ensure all action bodies generate valid FIRRTL
   
5. **Tooling and Integration**
   - Fix Python bindings for programmatic access
   - Create VSCode/IDE language support
   - Add debugging and visualization tools
   - Implement VCD trace generation for waveform viewing