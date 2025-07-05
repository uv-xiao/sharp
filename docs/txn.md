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

## Attributes

### Conflict Relations

The dialect defines conflict relation attributes for scheduling:
- `#txn.SB` (0): Sequenced Before - first action must execute before second
- `#txn.SA` (1): Sequenced After - first action must execute after second  
- `#txn.C` (2): Conflict - actions cannot execute in same cycle
- `#txn.CF` (3): Conflict-Free - actions can execute in any order

### Timing Attributes

Methods and rules can specify timing constraints:
- `"combinational"`: Executes within the same cycle
- `"static(n)"`: Fixed n-cycle latency
- `"dynamic"`: Variable latency determined at runtime

### Method Attributes

For FIRRTL generation and interface control:
- `prefix`: Custom prefix for generated signals
- `result`: Result signal name
- `enable`: Enable signal name  
- `ready`: Ready signal name
- `always_ready`: Method is always ready (value methods)
- `always_enable`: Method fires every cycle when ready

## Operations

### Module Definition

```mlir
txn.module @FIFO {
  // Instances of other modules
  %storage = txn.instance @storage of @Storage : !txn.module<"Storage">
  
  // Methods and rules...
  
  // Schedule - lists methods/rules with optional conflict matrix
  txn.schedule [@enqueue, @dequeue, @processRule] {
    // Conflict matrix entries (optional)
    conflict_matrix = #txn.conflict_dict<{
      "enqueue_dequeue" = #txn.SB,   // enqueue Sequenced Before dequeue
      "dequeue_processRule" = #txn.C  // dequeue Conflicts with processRule
    }>
  }
}
```

### Instance Declaration

```mlir
%inst = txn.instance @instanceName of @ModuleName : !txn.module<"ModuleName">
```

### Primitive Operations

#### Primitive Instance Declaration
```mlir
// Instances of parametric primitives with type arguments
%reg32 = txn.instance @myReg of @Register<i32> : !txn.module<"Register">
%wire8 = txn.instance @myWire of @Wire<i8> : !txn.module<"Wire">
%fifo = txn.instance @fifo of @SpecFIFO<i32> : !txn.module<"SpecFIFO">  // Spec primitive
```

#### Calling Primitive Methods
```mlir
// Read from register using value method
%value = txn.call @myReg.read() : () -> i32

// Write to register using action method
txn.call @myReg.write(%newValue) : (i32) -> ()
```

### Value Method

```mlir
txn.value_method @isEmpty() -> i1 
    attributes {always_ready = true, prefix = "is_empty"} {
  // Pure computation returning a value
  %empty = txn.call @storage.isEmpty() : () -> i1
  txn.return %empty : i1
}
```

### Action Method

```mlir
txn.action_method @enqueue(%data: i32) -> i32 
    attributes {timing = "combinational", enable = "enq_en", ready = "enq_rdy"} {
  // Method that may modify state, abort, or return values
  %full = txn.call @storage.isFull() : () -> i1
  txn.if %full {
    txn.abort
  } else {
    txn.call @storage.push(%data) : (i32) -> ()
  }
  txn.return %data : i32
}
```

### Rule

```mlir
txn.rule @processReady attributes {timing = "static(2)"} {
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

// Conditional call with reachability condition
%cond = arith.constant true : i1
%result = txn.call @process() if %cond : () -> ()
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

## Additional Operations

### FIRRTL Bridge Operations

```mlir
// FIRRTL-specific value method (used in primitive definitions)
txn.fir_value_method @getValue() -> i32 
    attributes {firrtl.port = "get_value"} : () -> i32

// FIRRTL-specific action method (used in primitive definitions)
txn.fir_action_method @setValue(%v: i32) 
    attributes {firrtl.enable_port = "set_en", firrtl.data_port = "set_data"} : (i32) -> ()
```

### Clock and Reset Configuration

```mlir
// Specify clock for a primitive
txn.clock_by %reg, %clk : !txn.clock

// Specify reset for a primitive  
txn.reset_by %reg, %rst : !txn.reset
```

### State Operation (Planned)

```mlir
// Declare module state (not yet implemented)
txn.state @counter : i32
```

## Integration with Standard Dialects

The TXN dialect integrates seamlessly with standard MLIR dialects:
- **func dialect**: Can be used for utility functions alongside TXN modules
- **scf dialect**: Regions can use `scf.yield` for yielding values
- **arith dialect**: Arithmetic operations for computations
- **CIRCT dialects**: HW, Comb, Seq, SV for hardware implementation
- **SMT dialect**: For formal verification constraints
- **Index dialect**: For parameterized designs

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
  
  // Schedule with optional conflict matrix
  txn.schedule [@enqueue, @dequeue, @autoProcess] {
    conflict_matrix = #txn.conflict_dict<{
      "enqueue_dequeue" = #txn.C,     // Cannot execute together
      "enqueue_autoProcess" = #txn.CF  // Can execute in any order
    }>
  }
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

- ✅ Module and primitive definitions with auto-generated parsers
- ✅ Module instances with proper symbol references
- ✅ Value and action methods with timing and interface attributes
- ✅ Schedule with conflict matrix dictionary attribute
- ✅ Primitive operations (Register, Wire, SpecFIFO, SpecMemory)
- ✅ Read/Write operations for primitives
- ✅ FIRRTL bridge operations (fir_value_method, fir_action_method)
- ✅ Clock and reset configuration operations
- ✅ Instance method calls with nested symbol references
- ✅ Conditional method calls with reachability
- ✅ Transaction semantics with abort propagation
- ✅ Conditional execution (txn.if) with results and txn.yield
- ✅ Rules with timing attributes
- ✅ Custom terminators: txn.return for methods, txn.yield for regions
- ✅ Type system with action, value, and module types
- ✅ Conflict relation attributes (SB, SA, C, CF)
- ✅ Auto-generated assembly formats for most operations
- ⏳ State operation (txn.state) - not yet implemented

## Future Extensions

- Pattern matching support
- Advanced scheduling hints
- Verification attributes and contracts
- Performance optimization directives
- Lowering passes to hardware dialects
- Static analysis for one-action restriction