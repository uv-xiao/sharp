# Sharp Simulation Framework

## Overview

Sharp provides multiple simulation modes:
- **Transaction-level (TL)**: Fast functional simulation
- **RTL**: Cycle-accurate via CIRCT's arcilator  
- **JIT**: Direct execution via LLVM
- **Concurrent**: DAM methodology for parallel execution
- **Hybrid**: Mixed TL/RTL with synchronization

## Usage

### Transaction-Level Simulation
```bash
sharp-opt input.mlir --sharp-simulate -o sim.cpp
clang++ -std=c++17 sim.cpp -o sim
./sim --cycles 100 --verbose
```

### RTL Simulation via Arcilator
```bash
sharp-opt input.mlir --sharp-arcilator -o output.arc.mlir
arcilator output.arc.mlir --trace trace.vcd
```

### JIT Compilation
```bash
sharp-opt input.mlir --sharp-simulate=jit
```

### Concurrent Simulation
```bash
sharp-opt input.mlir --sharp-concurrent-sim -o concurrent.cpp
clang++ -std=c++17 -pthread concurrent.cpp -o concurrent_sim
./concurrent_sim --threads 4
```

## Transactional Execution Model

Sharp follows a three-phase execution model:

1. **Value Phase**: All value methods computed once
2. **Execution Phase**: Rules fire based on schedule and conflicts  
3. **Commit Phase**: State updates become visible

Key principles:
- Actions execute atomically with abort propagation
- Conflict matrix enforces mutual exclusion
- Deterministic scheduling within cycles


## Known Limitations

1. **Primitive Method Calls**: Instance method calls need wrapper generation
2. **Control Flow**: Complex txn.if/yield patterns may fail conversion
3. **Python Bindings**: Runtime loading issues need fixes
4. **Multi-cycle Synthesis**: Launch operations not yet synthesizable