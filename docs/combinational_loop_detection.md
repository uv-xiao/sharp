# Combinational Loop Detection

Sharp provides automatic detection of combinational loops in transaction-level modules that could create invalid hardware.

## Overview

Combinational loops occur when signal dependencies form cycles through purely combinational paths. These can cause undefined behavior, oscillations, or synthesis failures in hardware designs.

## How It Works

The combinational loop detection pass:

1. **Builds a dependency graph** of signals flowing through the design
2. **Identifies combinational paths** through:
   - Value method calls (always combinational)
   - Combinational primitives (e.g., Wire)
   - Arithmetic and logic operations
3. **Detects cycles** using depth-first search
4. **Reports detailed error messages** showing the complete cycle path

## Usage

```bash
sharp-opt --sharp-detect-combinational-loops input.mlir
```

## Examples

### ✗ Simple Combinational Loop

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
  
  txn.schedule [@getValue, @compute] {
    conflict_matrix = {}
  }
}
```

**Error:** `Combinational loop detected: BadModule::getValue -> BadModule::compute -> BadModule::getValue`

### ✓ Valid Sequential Design

```mlir
txn.module @GoodModule {
  %reg = txn.instance @state of @Register<i32> : !txn.module<"Register">
  
  txn.value_method @getValue() -> i32 {
    %val = txn.call @state::@read() : () -> i32
    txn.return %val : i32
  }
  
  txn.action_method @increment() -> () {
    %val = txn.call @getValue() : () -> i32  // OK - Register breaks the loop
    %c1 = arith.constant 1 : i32
    %result = arith.addi %val, %c1 : i32
    txn.call @state::@write(%result) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@getValue, @increment] {
    conflict_matrix = {}
  }
}
```

This is valid because the Register primitive provides a sequential (clocked) dependency, breaking the combinational loop.

## Primitive Classification

The pass distinguishes between combinational and sequential primitives:

- **Combinational**: Wire - read immediately reflects write
- **Sequential**: Register - read is delayed by a clock cycle

Custom primitives can specify their combinational paths using attributes:

```mlir
txn.primitive @CustomPrimitive type = "hw" interface = !txn.module<"CustomPrimitive"> {
  txn.fir_value_method @read() : () -> i32
  txn.fir_action_method @write() : (i32) -> ()
} {combinational_paths = {read = true, write = false}}
```

## Analysis Algorithm

1. **Value Method Analysis**: All value methods are considered combinational - their outputs immediately depend on their inputs.

2. **Call Chain Tracking**: The pass tracks dependencies through method calls:
   - If method A calls method B, then A depends on B
   - Cycles in this dependency graph indicate combinational loops

3. **Primitive Handling**: 
   - Wire primitives create combinational dependencies between read and write
   - Register primitives do NOT create combinational dependencies (they're sequential)

4. **Cycle Detection**: Uses depth-first search to find strongly connected components in the dependency graph.

## Limitations

- Currently only detects loops between different methods/operations
- Intra-method loops (within a single method body) are not detected
- Custom primitive combinational paths require explicit attributes

## Integration

The pass is typically run before FIRRTL conversion to catch design errors early:

```bash
sharp-opt --sharp-detect-combinational-loops --convert-txn-to-firrtl input.mlir
```

Combinational loops will prevent successful hardware synthesis, so this analysis is essential for validating designs before proceeding to implementation.