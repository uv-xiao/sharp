# Combinational Loop Detection

## Overview

Sharp automatically detects combinational loops - circular signal dependencies through purely combinational paths that cause undefined behavior or synthesis failures.

## Usage

```bash
sharp-opt --sharp-detect-combinational-loops input.mlir
```

## How It Works

1. **Build dependency graph** of signal flow
2. **Identify combinational paths** through value methods, Wire primitives, arithmetic operations
3. **Detect cycles** using depth-first search
4. **Report detailed paths** showing the complete loop

## Examples

### ✗ Invalid: Combinational Loop
```mlir
txn.module @BadModule {
  txn.value_method @getValue() -> i32 {
    %val = txn.call @compute() : () -> i32
    txn.return %val : i32
  }
  
  txn.value_method @compute() -> i32 {
    %val = txn.call @getValue() : () -> i32  // Creates loop!
    %c1 = arith.constant 1 : i32
    %result = arith.addi %val, %c1 : i32
    txn.return %result : i32
  }
}
```
**Error:** `Combinational loop detected: getValue -> compute -> getValue`

### ✓ Valid: Sequential Design
```mlir
txn.module @GoodModule {
  %reg = txn.instance @state of @Register<i32>
  
  txn.value_method @getValue() -> i32 {
    %val = txn.call @state::@read() : () -> i32
    txn.return %val : i32
  }
  
  txn.action_method @increment() {
    %val = txn.call @getValue() : () -> i32  // OK - Register breaks loop
    %c1 = arith.constant 1 : i32
    %result = arith.addi %val, %c1 : i32
    txn.call @state::@write(%result) : (i32) -> ()
    txn.return
  }
}
```

## Primitive Classification

- **Combinational**: Wire (read immediately reflects write)
- **Sequential**: Register (read delayed by clock cycle)

Custom primitives specify paths:
```mlir
txn.primitive @CustomPrimitive {
  // Methods...
} {combinational_paths = {read = true, write = false}}
```

## Integration

Run before FIRRTL conversion:
```bash
sharp-opt --sharp-detect-combinational-loops --convert-txn-to-firrtl input.mlir
```