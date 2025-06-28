# Sharp Transaction (txn) Dialect

## Overview

The `sharp.txn` dialect provides a transactional hardware description layer inspired by the Fjfj language from "Making Concurrent Hardware Verification Sequential" (Bourgeat et al., 2025). This dialect enables sequential reasoning about concurrent hardware while maintaining the ability to synthesize efficient concurrent implementations.

## Key Concepts

### Modules and Instances

Modules are the primary structuring mechanism, containing:
- **Instances**: Instances of other modules that can be called
- **Primitives**: Low-level hardware components (synthesizable or specification)
- **Value Methods**: Pure, read-only methods that observe state
- **Action Methods**: Methods that may modify state, abort, or return values
- **Rules**: Spontaneous transitions that execute
- **Schedule**: Lists the methods/rules that can be executed (terminator)

### Transaction Semantics

- **Atomicity**: All effects of a transaction happen together or not at all
- **Abort Propagation**: Failed preconditions cause the entire transaction to abort
- **Sequential Schedule**: Actions within a transaction execute in sequence
- **One-Action Restriction**: At most one action method per primitive per cycle

## Type System

### Action Type
```mlir
!txn.action<T>
```
Represents an action that either produces a value of type `T` or aborts.

### Value Type
```mlir
!txn.value<T>
```
Represents a pure computation producing type `T`.

### Module Type
```mlir
!txn.module<"ModuleName">
```
References a module's interface.

## Operations

### Module Definition

```mlir
txn.module @FIFO {
  // Instances of other modules
  %storage = txn.instance @storage of @Storage : !txn.module<"Storage">
  
  // Methods and rules...
  
  // Schedule - lists methods/rules that can be executed (terminator)
  txn.schedule [@enqueue, @dequeue, @processRule]
}
```

### Instance Declaration

```mlir
%inst = txn.instance @instanceName of @ModuleName : !txn.module<"ModuleName">
```

### Primitive Definition

```mlir
txn.primitive @buffer type = "hw" interface = !txn.module<"BufferInterface"> {
  // Primitive implementation (synthesizable or specification)
}
```

### Value Method

```mlir
txn.value_method @isEmpty() -> i1 {
  // Pure computation returning a value
  %empty = txn.call @storage.isEmpty() : () -> i1
  func.return %empty : i1
}
```

### Action Method

```mlir
txn.action_method @enqueue(%data: i32) -> i32 {
  // Method that may modify state, abort, or return values
  %full = txn.call @storage.isFull() : () -> i1
  txn.if %full {
    txn.abort
  } else {
    txn.call @storage.push(%data) : (i32) -> ()
  }
  func.return %data : i32
}
```

### Rule

```mlir
txn.rule @processReady {
  // Spontaneous transition
  %ready = txn.call @isReady() : () -> i1
  txn.if %ready {
    txn.call @process() : () -> ()
  } else {
    // Do nothing
  }
}
```

### Method Calls

```mlir
// Call method on current module
%value = txn.call @getValue() : () -> i32

// Call method on instance (nested symbol reference)
%result = txn.call @storage.push(%data) : (i32) -> ()
```

### Conditional Execution

```mlir
// If-then-else with results
%result = txn.if %condition -> i32 {
  %value = arith.constant 10 : i32
  scf.yield %value : i32
} else {
  %value = arith.constant 20 : i32
  scf.yield %value : i32
}

// If with no results - use txn.yield to terminate
txn.if %condition {
  "some.operation"() : () -> ()
  txn.yield
} else {
  txn.yield  // Empty else region  
}

// Guarded execution (using if + abort)
txn.if %not_ready {
  txn.abort  // abort is a terminator
} else {
  // Continue execution
  txn.yield
}
```

### Abort

```mlir
txn.abort
```
Causes the current transaction to fail and rollback.

## Integration with Standard Dialects

The TXN dialect integrates seamlessly with standard MLIR dialects:
- **func dialect**: Can be used for utility functions alongside TXN modules
- **scf dialect**: Regions can use `scf.yield` for yielding values
- **arith dialect**: Arithmetic operations for computations
- **CIRCT dialects**: HW, Comb, Seq for hardware implementation

### Terminators in TXN
- **txn.return**: Returns values from TXN methods (value_method, action_method)
- **txn.yield**: Yields values from regions (e.g., in txn.if)

## Example: FIFO Buffer

```mlir
txn.module @FIFO {
  // Instance of storage module
  %storage = txn.instance @storage of @Storage : !txn.module<"Storage">
  
  txn.value_method @isEmpty() -> i1 {
    %empty = txn.call @storage.isEmpty() : () -> i1
    func.return %empty : i1
  }
  
  txn.value_method @isFull() -> i1 {
    %full = txn.call @storage.isFull() : () -> i1
    func.return %full : i1
  }
  
  txn.action_method @enqueue(%data: i32) {
    %full = txn.call @isFull() : () -> i1
    txn.if %full {
      txn.abort
    } else {
      txn.call @storage.write(%data) : (i32) -> ()
      txn.yield
    }
    txn.return
  }
  
  txn.action_method @dequeue() -> i32 {
    %empty = txn.call @isEmpty() : () -> i1
    txn.if %empty {
      txn.abort
    } else {
      %data = txn.call @storage.read() : () -> i32
      txn.return %data : i32
    }
  }
  
  txn.rule @autoProcess {
    %ready = txn.call @isProcessReady() : () -> i1
    txn.if %ready {
      %data = txn.call @dequeue() : () -> i32
      txn.call @process(%data) : (i32) -> ()
      txn.yield
    } else {
      // Do nothing
      txn.yield
    }
  }
  
  // Schedule lists the methods/rules that can be executed
  txn.schedule [@enqueue, @dequeue, @autoProcess]
}

// Storage module (primitive)
txn.module @Storage {
  txn.primitive @mem type = "hw" interface = !txn.module<"StorageInterface"> {
    // HW implementation would go here
  }
  
  txn.value_method @isEmpty() -> i1 {
    %true = arith.constant true
    func.return %true : i1
  }
  
  txn.value_method @isFull() -> i1 {
    %false = arith.constant false
    func.return %false : i1
  }
  
  txn.action_method @write(%data: i32) {
    // Implementation
    txn.return
  }
  
  txn.value_method @read() -> i32 {
    %zero = arith.constant 0 : i32
    func.return %zero : i32
  }
  
  txn.schedule [@isEmpty, @isFull, @write, @read]
}
```

## Design Rationale

1. **Reuse Standard Operations**: Instead of duplicating return/yield operations, we reuse standard terminators from func/scf/cf dialects
2. **Auto-generated Assembly**: Most operations use TableGen's assemblyFormat for consistency
3. **Simple Rules**: Removed guard conditions from rules - use if/abort pattern instead
4. **Module Instances**: Enables hierarchical composition and reuse
5. **Explicit Scheduling**: Schedule lists methods/rules that can be executed
6. **Flexible Control Flow**: If regions can be empty for more natural patterns

## Implementation Status

The TXN dialect has been successfully implemented with the following features:

- ✅ Module and primitive definitions with auto-generated parsers where possible
- ✅ Module instances with proper symbol references
- ✅ Value and action methods using txn.return terminator
- ✅ Schedule as module terminator listing executable methods/rules as symbol references
- ✅ Instance method calls with nested symbol references
- ✅ Transaction semantics with abort propagation
- ✅ Conditional execution (txn.if) with results and txn.yield
- ✅ Rules for spontaneous transitions (without guards)
- ✅ Custom terminators: txn.return for methods, txn.yield for regions
- ✅ Type system with action, value, and module types
- ✅ Removed duplicate operations (no separate return/yield for each context)
- ✅ Auto-generated assembly formats for most operations

## Future Extensions

- Pattern matching support
- Advanced scheduling hints
- Verification attributes and contracts
- Performance optimization directives
- Lowering passes to hardware dialects
- Static analysis for one-action restriction