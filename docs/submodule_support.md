# Submodule Support in Sharp

## Overview

This document describes the implementation requirements for supporting submodules in the Sharp Txn-to-FIRRTL conversion. Currently, the conversion fails when encountering instance method calls like `@instance::@method` due to missing port generation and method call routing.

## Current Problem

When converting Txn modules with submodule instances, the TxnToFIRRTL pass fails with errors like:

```
error: Could not find output port for instance method: reg1::read
```

This occurs because the pass cannot resolve instance method calls to the appropriate FIRRTL ports on the corresponding `firrtl.instance` operations.

## Required Implementation

### 1. Instance Call Detection

The `TxnToFIRRTL` conversion logic needs to distinguish between:
- **Local method calls**: `txn.call @method()` - calls to methods in the same module
- **Instance method calls**: `txn.call @instance::@method()` - calls to methods on submodule instances

### 2. FIRRTL Instance Resolution

For instance method calls, the converter must:
1. Locate the corresponding `txn.instance` operation that declares the instance
2. Find the associated `firrtl.instance` operation created for that instance
3. Generate or locate the appropriate port names for the method being called

### 3. Port Name Generation

For each method type, the converter should generate consistent port names:

**Value Methods** (`txn.value_method @read() -> i32`):
- Input ports: `read_en` (enable signal)
- Output ports: `read_result` (return value)

**Action Methods** (`txn.action_method @write(%val: i32)`):
- Input ports: `write_en` (enable signal), `write_val` (parameter value)
- Output ports: None for void return type

### 4. Connection Generation

The converter must generate `firrtl.connect` operations to:
1. Connect rule enable signals to instance method enable ports
2. Connect method parameters to instance input ports
3. Connect instance output ports to where method results are used

## Implementation Location

The main changes should be made in:
- `lib/Conversion/TxnToFIRRTL/TxnToFIRRTLPass.cpp`
- Specifically in the `txn.call` conversion logic

## Example Conversion

**Before (Txn)**:
```mlir
%reg1 = txn.instance @reg1 of @Register<i32> : !txn.module<"Register">
txn.rule @copy_rule {
  %val = txn.call @reg1::@read() : () -> i32
  txn.call @reg2::@write(%val) : (i32) -> ()
  txn.return
}
```

**After (FIRRTL)**:
```mlir
%reg1 = firrtl.instance reg1 @Register_i32
firrtl.connect %reg1.read_en, %rule_enable
%val = %reg1.read_result
firrtl.connect %reg2.write_en, %rule_enable
firrtl.connect %reg2.write_val, %val
```

## Testing

The implementation should be validated using:
- `test/Conversion/TxnToFIRRTL/submodule-instantiation.mlir`
- `test/Conversion/TxnToFIRRTL/nested-modules.mlir`
- `test/Conversion/TxnToFIRRTL/txn-to-firrtl-complete.mlir`

## Status

- **Current**: Not implemented - causes conversion failures
- **Priority**: High - blocks advanced modular designs
- **Dependencies**: None - can be implemented with current infrastructure