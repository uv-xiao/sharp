# LLVM Lowering Completion

## Overview

This document describes the required implementation to complete the Txn dialect to LLVM lowering for simulation support. Currently, the `--sharp-simulate` pass fails because some `txn` operations survive the conversion chain and cannot be legalized to LLVM.

## Current Problem

The simulation fails with errors like:

```
error: failed to legalize operation 'txn.yield' that was explicitly marked illegal
```

This indicates that the conversion path from Txn → Func → LLVM is incomplete.

## Root Cause Analysis

The conversion follows this path:
1. **Txn → Func**: `--convert-txn-to-func` converts txn operations to standard dialects
2. **Func → LLVM**: Standard MLIR lowering converts func/scf/arith to LLVM

The failure occurs because the `TxnToFunc` conversion doesn't handle all cases of `txn.yield` correctly.

## Specific Issue: YieldToSCFYieldPattern

The `YieldToSCFYieldPattern` in `lib/Conversion/TxnToFunc/TxnToFuncPass.cpp` has incomplete logic:

```cpp
// Current problematic logic
} else if (auto funcOp = op->getParentOfType<mlir::func::FuncOp>()) {
  // Only handles rule functions
  if (funcOp.getName().contains("_rule_")) {
    // Convert to func.return
  } else {
    return mlir::failure(); // FAILS for action methods
  }
}
```

**Problem**: Action methods converted by `ActionMethodToFuncPattern` don't have `_rule_` in their names, so their `txn.yield` operations are not converted.

## Required Implementation

### 1. Fix YieldToSCFYieldPattern

Replace name-based detection with signature-based detection:

```cpp
// In lib/Conversion/TxnToFunc/TxnToFuncPass.cpp
mlir::LogicalResult YieldToSCFYieldPattern::matchAndRewrite(
    sharp::txn::YieldOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  
  if (auto ifOp = op->getParentOfType<mlir::scf::IfOp>()) {
    // Handle yields inside scf.if operations
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op);
    return mlir::success();
    
  } else if (auto funcOp = op->getParentOfType<mlir::func::FuncOp>()) {
    // Check if function returns single i1 (abort flag)
    // This correctly identifies both rule and action method functions
    if (funcOp.getNumResults() == 1 && 
        funcOp.getResultTypes()[0].isInteger(1)) {
      
      // txn.yield means success (not aborted)
      auto falseVal = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false));
      rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, falseVal.getResult());
      return mlir::success();
      
    } else {
      // Invalid context - yield in value method
      return op.emitError("txn.yield not allowed in value methods");
    }
    
  } else {
    return op.emitError("txn.yield in invalid context");
  }
}
```

### 2. Alternative: Function Attribute Approach

Another approach is to use function attributes:

```cpp
// In ActionMethodToFuncPattern, add attribute:
funcOp->setAttr("sharp.converted_from", rewriter.getStringAttr("action_method"));

// In RuleToFuncPattern, add attribute:
funcOp->setAttr("sharp.converted_from", rewriter.getStringAttr("rule"));

// In YieldToSCFYieldPattern, check attribute:
if (auto convertedFrom = funcOp->getAttrOfType<mlir::StringAttr>("sharp.converted_from")) {
  if (convertedFrom.getValue() == "action_method" || 
      convertedFrom.getValue() == "rule") {
    // Convert to func.return false
  }
}
```

## Testing

Validate the fix with:
- `test/Simulation/three-phase-execution.mlir`

Expected behavior:
- All `txn.yield` operations should be converted appropriately
- No `txn` operations should survive to LLVM lowering
- Simulation should execute successfully

## Implementation Priority

The signature-based approach is preferred because:
1. **More Robust**: Doesn't rely on naming conventions
2. **Type Safe**: Uses actual function signatures
3. **Maintainable**: Clear semantic meaning

## Files to Modify

1. `lib/Conversion/TxnToFunc/TxnToFuncPass.cpp`
   - Update `YieldToSCFYieldPattern::matchAndRewrite`
   - Replace name checking with signature checking

## Status

- **Current**: Not implemented - yields in action methods not converted
- **Priority**: High - blocks simulation functionality  
- **Dependencies**: None - fix is self-contained

## Notes

This fix is critical for the simulation pipeline. Once `txn.yield` operations are properly converted to `func.return`, the standard MLIR lowering pipeline can handle the rest of the conversion to LLVM.