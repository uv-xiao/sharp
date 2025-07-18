# Sharp Transaction (txn) Dialect

## Overview

The `txn` dialect provides transactional hardware description with atomic execution and conflict resolution. Inspired by Bluespec and Koika, it enables sequential reasoning about concurrent hardware, multi-level simulation, and efficient RTL generation.

## Core Concepts

- **Module**: instance + rules/methods + schedule
- **Primitive**: special built-in modules with software semantics and optional hardware implementation
- **Action**: Rule or action method - the schedulable units that modify state
- **Rule**: Guarded atomic action executing autonomously
- **Action Method**: State-modifying method callable from parent modules  
- **Value Method**: Pure function reading state without side effects
- **Schedule**: Execution order of actions (rules and action methods) and conflicts



## Core Operations

### Module Structure
```mlir
txn.module @Counter {
  %reg = txn.instance @count of @Register<i32> : index
  
  txn.rule @increment {
    %val = txn.call @count::@read() : () -> i32
    %one = arith.constant 1 : i32
    %next = arith.addi %val, %one : i32
    txn.call @count::@write(%next) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@increment] {
    conflict_matrix = {}
  }
}
```

### Method Types

> **IMPORTANT**: All rules and value methods must end with `txn.return` (with optional arguments for value methods). Action methods may also end with `txn.return` or may abort with `txn.abort`.

**Value Methods** - Pure, read-only (must end with `txn.return`):
```mlir
txn.value_method @getValue() -> i32 {
  %val = txn.call @reg::@read() : () -> i32
  txn.return %val : i32
}
```

**Action Methods** - Can modify state or abort (must end with `txn.return`):
```mlir
txn.action_method @setValue(%arg: i32) {
  txn.call @reg::@write(%arg) : (i32) -> ()
  txn.return
}
```

**Rules** - Spontaneous actions (must end with `txn.return`):
```mlir
txn.rule @autoIncrement {
  %cond = arith.cmpi slt, %counter, %limit : i32
  txn.if %cond {
    txn.call @increment() : () -> ()
    txn.return
  } else {
    txn.abort
  }
}
```

### Control Flow

```mlir
// Conditional execution
txn.if %condition {
  // then region
  txn.yield
} else {
  // else region
  txn.yield
}

// Early termination
txn.abort  // Aborts the entire transaction

// Multi-cycle operations
txn.future {
  %done = txn.launch after 3 {
    // Executes 3 cycles later
    txn.yield
  }
}
```

## Conflict Relations

Schedules specify execution order and conflicts:

- **SB (0)**: Sequenced Before - A must execute before B
- **SA (1)**: Sequenced After - A must execute after B  
- **C (2)**: Conflict - A and B cannot execute in same cycle
- **CF (3)**: Conflict-Free - A and B can execute in any order

```mlir
txn.schedule [@rule1, @rule2] {
  conflict_matrix = {
    "rule1,rule2" = 2 : i32  // Conflict
  }
}
```

## Execution Model

See [execution_model.md](execution_model.md) for more details.

## Primitives

Built-in hardware primitives with software semantics:

- **Register<T>**: State storage with read/write
- **Wire<T>**: Combinational connection  
- **FIFO<T>**: Queue with enqueue/dequeue
- **Memory<T>**: Addressable storage
- **SpecFIFO<T>**: Unbounded queue for specification
- **SpecMemory<T>**: Memory with configurable latency