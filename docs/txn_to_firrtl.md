# Sharp Txn to FIRRTL Conversion

## Overview

This document describes the implemented algorithm for converting Sharp Txn modules to FIRRTL. The conversion extends the Koika approach to support transaction-based hardware description with methods, conflict matrices, and parametric primitives.

## Key Features

### Transaction Model Extensions
- **Methods**: Reusable action/value methods with conflict resolution
- **Conflict Matrix**: Explicit specification of action relationships (SB, SA, C, CF)
- **Module Hierarchy**: Support for instantiating submodules and primitives
- **Parametric Types**: Register<T>, Wire<T> with automatic instantiation

### Method Attributes
Methods support attributes to control FIRRTL signal generation:
- `prefix`: Custom prefix for signal names (default: method name)
- `result`: Postfix for data signals (default: "OUT")
- `ready`/`enable`: Postfixes for handshake signals (action methods only)
- `always_ready`/`always_enable`: Optimize away handshake signals when possible

## Translation Process

### 1. Module Processing Order
Modules are processed bottom-up based on dependencies:
1. Build dependency graph from instance relationships
2. Topologically sort (primitives and leaves first)
3. Identify top-level module (not instantiated by others)

### 2. FIRRTL Module Generation
Each Txn module generates a FIRRTL module with:
- Clock and reset ports
- Method interface ports (data, enable, ready)
- Instance ports for submodules
- Internal logic for rules and methods

### 3. Will-Fire Logic

The will-fire (WF) logic determines when actions can execute:

#### Static Mode (Conservative)
```
wf[action] = enabled[action] && !conflicts_with_earlier[action] && !conflict_inside[action]
```

#### Dynamic Mode (Precise)
```
wf[action] = enabled[action] && AND{for every m in action, NOT(reach(m) && conflict_with_earlier(m))}
```

### 4. Conflict Resolution

#### Conflict Detection
- Actions conflict based on the methods they call
- If action A calls method M1 and action B calls method M2, and M1 conflicts with M2, then A conflicts with B
- The conflict matrix specifies relationships: SB (sequential before), SA (sequential after), C (conflict), CF (conflict-free)

#### Conflict Inside
Detects conflicts within a single action:
- Analyzes all method calls within an action
- Computes reachability conditions considering control flow
- Generates hardware to prevent execution when internal conflicts would occur

### 5. Primitive Support

#### Parametric Primitives
- Primitives are instantiated on-demand when referenced
- Type parameters create unique FIRRTL modules: `Register_i32_impl`, `Wire_i8_impl`
- Automatic construction during conversion if primitive not found

#### Built-in Primitives
- **Register<T>**: State element with read/write methods
- **Wire<T>**: Combinational connection with read/write methods

## Implementation Details

### Pass Structure
1. **Pre-synthesis Check**: Validates synthesizable constructs
2. **Conflict Matrix Inference**: Completes partial conflict specifications
3. **Reachability Analysis**: Computes conditions for method calls
4. **Method Attribute Validation**: Ensures valid signal names
5. **TxnToFIRRTL Conversion**: Main translation pass

### Key Data Structures
```cpp
struct ConversionContext {
  DenseMap<Value, Value> txnToFirrtl;           // Value mapping
  DenseMap<StringRef, Value> willFireSignals;   // WF signals
  DenseMap<StringRef, SmallVector<StringRef>> methodCallers;
  DenseMap<Operation*, Value> reachabilityConditions;
  DenseMap<StringRef, DenseMap<StringRef, Value>> instancePorts;
};
```

## Example

### Input Txn
```mlir
txn.module @Counter {
  %count = txn.instance @count of @Register<i32> : !txn.module<"Register">
  
  txn.action_method @increment() {
    %val = txn.call @count::@read() : () -> i32
    %inc = arith.addi %val, %c1 : i32
    txn.call @count::@write(%inc) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@increment] {
    conflict_matrix = {}
  }
}
```

### Generated FIRRTL Structure
- Module with clock, reset, increment_EN, increment_RDY ports
- Instance of Register_i32_impl primitive
- Will-fire logic for increment method
- Connections to register read/write ports

## Current Status

### Completed
- ✅ Full TxnToFIRRTL conversion pass
- ✅ Parametric primitive support
- ✅ Conflict inside detection
- ✅ Static and dynamic will-fire modes
- ✅ Reachability analysis with conditional calls
- ✅ 45/45 tests passing

### Limitations
- Multi-cycle operations not yet supported
- Combinational loop detection requires primitive attributes
- Non-synthesizable primitives will fail translation

## Usage

```bash
# Convert Txn to FIRRTL
sharp-opt --convert-txn-to-firrtl input.mlir

# Use static will-fire mode (default is dynamic)
sharp-opt --convert-txn-to-firrtl="will-fire-mode=static" input.mlir
```