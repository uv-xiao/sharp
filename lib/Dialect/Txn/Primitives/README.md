# Sharp Primitives

This directory contains implementations of hardware primitives with software semantics for simulation.

## Primitive Types

### Hardware Primitives (`type = "hw"`)
- **Register**: State-holding element with clock/reset
- **Wire**: Combinational connection
- **FIFO**: First-in-first-out queue
- **Memory**: Random access memory

### Specification Primitives (`type = "spec"`)  
- **SpecFIFO**: Unbounded FIFO for specification
- **SpecMemory**: Memory with configurable latency
- **SpecRandom**: Random value generator

## Software Semantics

Every primitive must provide software semantics for all its methods to enable transaction-level simulation. This includes:

1. **State representation**: How the primitive's state is modeled in software
2. **Method implementations**: Software behavior for each method
3. **Conflict relations**: How methods interact

## Implementation Pattern

Each primitive follows this pattern:

```mlir
txn.primitive @PrimitiveName<T> type = "hw" interface = !txn.module<"PrimitiveName"> {
  // Method declarations
  txn.fir_value_method @read() : () -> T
  txn.fir_action_method @write() : (T) -> ()
  
  // Hardware attributes
  txn.clock_by @clk
  txn.reset_by @rst
  
  // Conflict matrix
  txn.schedule [@read, @write] {
    conflict_matrix = { ... }
  }
}
```

The software semantics are implemented in the simulation passes.