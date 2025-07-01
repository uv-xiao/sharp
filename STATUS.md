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
- Testing infrastructure with lit/FileCheck (45/45 tests passing)

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

### 📋 Planned

- **Additional Primitives**
  - FIFO, Memory, and other common hardware primitives
  - Spec primitives for formal verification
  
- **Verilog Export**
  - Integration with CIRCT's export-verilog infrastructure
  - Command-line options for different export modes
  
- **Txn-level Combinational Loop Detection**
  - Requires attributes in txn.primitive to define combinational paths
  
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
   
2. **Verilog Export Pipeline**
   - Add --export-verilog flag to sharp-opt
   - Integrate with CIRCT's Verilog emission
   - Test end-to-end hardware generation
   
3. **Enhanced Analysis**
   - Implement combinational loop detection
   - Add performance analysis passes
   - Create resource utilization estimates
   
4. **Tooling and Integration**
   - Fix Python bindings for programmatic access
   - Create VSCode/IDE language support
   - Add debugging and visualization tools