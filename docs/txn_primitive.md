# Sharp Txn Primitives

## Overview

Sharp Txn primitives are fundamental building blocks providing transaction-level interfaces that bridge behavioral descriptions with hardware implementations in FIRRTL.

## Architecture

### Structure
```mlir
txn.primitive @PrimitiveName type = "hw" interface = index {
  // Method declarations with FIRRTL port mappings
  txn.fir_value_method @read() {firrtl.port = "read_data"} : () -> i32
  txn.fir_action_method @write() {firrtl.data_port = "write_data", 
                                 firrtl.enable_port = "write_enable"} : (i32) -> ()
  
  // Clocking and scheduling
  txn.clock_by @clk
  txn.reset_by @rst
  txn.schedule [@read, @write] {conflict_matrix = {...}}
} {firrtl.impl = "PrimitiveName_impl"}
```

### Method-Port Mapping
- **Value Methods**: `firrtl.port` specifies output port
- **Action Methods**: `firrtl.data_port` and `firrtl.enable_port` specify inputs

## Built-in Primitives

### Register
Stateful storage element holding values across clock cycles.

**Methods:**
- `read()` → value (CF with all)
- `write(value)` → void (C with write)

### Wire
Combinational connection between components.

**Methods:**
- `read()` → value (action method, SA with write)
- `write(value)` → void (SB with read, C with write)

### FIFO
Bounded queue with enqueue/dequeue operations.

**Methods:**
- `enqueue(value)` → void (C with dequeue)
- `dequeue()` → value (C with enqueue)
- `isEmpty()` → bool (CF)
- `isFull()` → bool (CF)

### Memory
Address-based storage for specification.

**Methods:**
- `read(addr)` → value (C with write on same addr)
- `write(addr, data)` → void (C with read/write on same addr)
- `clear()` → void (C with all)

### SpecFIFO
Unbounded queue for verification.

**Methods:**
- `enqueue(value)` → void (always succeeds)
- `dequeue()` → value
- `isEmpty()`, `size()`, `peek()` → status (CF)

### SpecMemory
Memory with configurable read latency.

**Methods:**
- `read(addr)` → value (dynamic timing)
- `write(addr, data)` → void
- `setLatency(cycles)`, `getLatency()`, `clear()`

## Usage Example

```mlir
txn.module @Counter {
  %count = txn.instance @count of @Register<i32> : index
  
  txn.action_method @increment() {
    %val = txn.call @count::@read() : () -> i32
    %one = arith.constant 1 : i32
    %next = arith.addi %val, %one : i32
    txn.call @count::@write(%next) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@increment]
}
```

## Conflict Relations
- **CF**: Conflict-Free - can execute in any order
- **SB/SA**: Sequenced Before/After - ordering constraint
- **C**: Conflict - cannot execute simultaneously

## Implementation Notes
- Primitives require `schedule` as last operation
- Type conversion from MLIR to FIRRTL handled automatically
- Spec primitives marked with `spec` attribute
- Software semantics included for simulation