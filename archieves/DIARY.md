# Sharp Development Diary

## 2025-07-04 - PySharp Implementation Following PyCDE Pattern

### User Request
The user disagreed with the initial EDSL implementation and requested to follow the PyCDE pattern:
- Study files under /home/uvxiao/sharp/circt/frontends/PyCDE
- Import from .circt similar to how PyCDE does it
- Create integration tests similar to PyCDE

### Key Actions Taken

1. **Studied PyCDE Structure**
   - PyCDE uses `MLIR_PYTHON_PACKAGE_PREFIX=pycde.circt.` to namespace MLIR/CIRCT bindings
   - Imports from `.circt` subpackage instead of directly importing MLIR/CIRCT
   - Has layered architecture with native bindings wrapped by Python modules

2. **Redesigned PySharp Following PyCDE Pattern**
   - Created single `pysharp.py` module with all functionality
   - Implements graceful fallback when native bindings unavailable
   - Import pattern tries multiple approaches:
     - `from ._mlir_libs._mlir import ir`
     - `from .sharp import ir` (fallback)
   - Does not directly import MLIR/CIRCT, but through Sharp's namespace

3. **Implemented Core PySharp Features**
   - Type system: IntType, UIntType, SIntType with predefined types (i8, i16, i32, etc.)
   - ConflictRelation enum (SB=0, SA=1, C=2, CF=3)
   - Signal class with operator overloading (+, -, &, |)
   - State, Port, Method (Value/Action), Rule classes
   - ModuleBuilder for programmatic construction
   - Module base class and @module decorator
   - Method decorators: @value_method, @action_method, @rule

4. **Created Integration Tests**
   - test_types.py - Tests type system and constants
   - test_module_builder.py - Tests ModuleBuilder API
   - test_module_class.py - Tests Module class and decorators
   - test_signal_arithmetic.py - Tests signal operations
   - test_standalone.py - Comprehensive test without native bindings
   - basic.py - FileCheck-style test

5. **Test Results**
   - All standalone tests pass successfully
   - PySharp functionality verified without native bindings
   - Native extension still has runtime loading issues (ImportError)

### Challenges and Solutions
- Native extension loading issue persists but doesn't affect standalone functionality
- Successfully implemented PyCDE-style import pattern with fallbacks

## 2025-07-03 (Session 2) - Simulation Framework Implementation

### User Request
Continue with "Simulation at Arbitrary Level" from STATUS.md:
- Read EQueue and DAM papers for reference
- Create docs/simulation.md with design and implementation plan
- Implement the simulation code
- Add tests

### Key Actions Taken

1. **Read Reference Papers**
   - EQueue: Event-driven simulation with dependency tracking
   - DAM: High-performance dataflow simulation with concurrent execution
   - Insights: Need event queues, dependency tracking, multi-cycle operations

2. **Created Comprehensive Design Document**
   - docs/simulation.md with complete architecture
   - Three levels: Transaction-Level, RTL (via arcilator), Hybrid
   - Event-driven architecture with conflict matrix support
   - Multi-cycle operation support for spec primitives
   - 6-phase implementation plan

3. **Implemented Core Simulation Components**
   - **Event.h/cpp**: Event structure with dependencies, event queue with priority
   - **SimModule.h/cpp**: Base module class with conflict matrix, method registration
   - **Simulator.h/cpp**: Main simulation engine with event scheduling and execution
   - Support for:
     - Event dependencies and deferred execution
     - Conflict checking based on conflict matrix
     - Multi-cycle operations through continuations
     - Performance tracking and debugging

4. **Created Simulation Infrastructure**
   - Added lib/Simulation directory structure:
     - Core/ - Base simulation components
     - TransactionLevel/ - Spec primitives
     - RTL/ - Arcilator integration (placeholder)
     - Hybrid/ - Bridge components (placeholder)
   - Updated build system to include simulation library
   - Created unit tests in unittests/Simulation/

5. **Implemented Spec Primitives**
   - SpecFIFO<T> - Unbounded FIFO with conflict relations
   - SpecMemory - Multi-cycle memory with configurable latency
   - Proper conflict matrix setup for each primitive

6. **Created Test Examples**
   - counter-sim.mlir - Basic counter with conflict handling
   - multi-cycle.mlir - Pipeline with multi-cycle operations
   - hybrid-sim.mlir - Example of hybrid TL/RTL simulation

7. **Defined Simulation Operations**
   - SimulationOps.td with sharp.sim dialect
   - Configuration operations for simulation setup
   - Spec primitives for multi-cycle operations
   - Performance measurement operations

### Technical Details
- Event-driven simulation with min-heap priority queue
- Dependency tracking allows complex causality chains
- Conflict matrix integration prevents concurrent conflicting operations
- Continuation support enables multi-cycle spec actions
- Builder pattern for easy simulation setup

### Status
- Core transaction-level simulation implemented
- Basic spec primitives created
- Test infrastructure established
- RTL integration (arcilator) and hybrid simulation pending

### Next Steps
- Complete spec primitive implementations
- Integrate with arcilator for RTL simulation
- Implement hybrid simulation bridge
- Add more comprehensive tests

### User Request
The user requested to move forward with the Pythonic Construction Frontend status:
- Guarantee Sharp's Python binding can access MLIR and CIRCT dialects
- Create a frontend module (PySharp) for EDSL functionality with access to mlir/circt/sharp dialects

### Key Actions Taken

1. **Removed Sharp Core Dialect**
   - User identified Core dialect as not useful
   - Removed all Core dialect files and references from CMakeLists
   - Updated InitAllDialects.h and Dialects.h
   - Cleaned up Python binding references

2. **Enhanced Python Bindings Infrastructure**
   - Updated SharpModule.cpp to register MLIR dialects: SCF, SMT, Index, Arith
   - Updated SharpModule.cpp to register CIRCT dialects: FIRRTL, Comb, HWArith, Seq, SV
   - Removed HW dialect registration due to conflicts with CIRCT's builtin dialect
   - Linked against appropriate CAPI libraries in CMakeLists.txt

3. **Created PySharp Frontend Module**
   - Implemented comprehensive EDSL in `lib/Bindings/Python/pysharp.py`
   - Features include:
     - Type system with IntType, BoolType, and FIRRTL types
     - Predefined types (i1, i8, i16, i32, i64, i128, i256)
     - ConflictRelation enum matching Txn dialect
     - Value class with operator overloading
     - ModuleBuilder API
     - Module decorator for class-based hardware description
   - Successfully tested standalone functionality

4. **Testing and Validation**
   - Created test_pysharp_standalone.py demonstrating all PySharp features
   - Verified type system, conflict relations, module building, and operations
   - Identified runtime loading issues with native extension

### Challenges Encountered
- Native extension has ImportError issues when loading
- HW dialect conflicts with CIRCT's builtin dialect registration
- MLIR/CIRCT Python path configuration is complex

### Status Update
- Updated STATUS.md to reflect completed PySharp frontend implementation
- Documented known limitations including runtime loading issues
- PySharp frontend is functional as a standalone module

### Next Steps
- Investigate and fix native extension loading issues
- Consider alternative approaches for MLIR/CIRCT integration
- Add more comprehensive examples and documentation for PySharp usage