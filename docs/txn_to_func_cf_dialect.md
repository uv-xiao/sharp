# TxnToFunc CF Dialect Support

## Overview

This document describes the required implementation for supporting mixed termination patterns in `txn.if` operations during TxnToFunc conversion. The current implementation uses the SCF (Structured Control Flow) dialect, which cannot handle cases where one branch uses `txn.yield` and another uses `txn.abort`.

## Current Problem

The TxnToFunc conversion fails with errors like:

```
error: 'scf.yield' op must be the last operation in the parent block
```

This occurs when `txn.if` operations have mixed termination patterns:
```mlir
txn.if %cond {
  // some operations
  txn.yield  // → converts to scf.yield
} else {
  txn.abort  // → converts to func.return, breaking scf.if structure
}
```

## Root Cause

The SCF dialect enforces structured control flow where regions must have single entry/exit points. When `txn.abort` (converted to `func.return`) appears inside an `scf.if` region, it creates an unstructured exit that violates SCF constraints.

## Solution: Switch to CF Dialect

The Control Flow (CF) dialect is designed for unstructured, branch-based control flow and can handle early returns correctly.

## Required Implementation

### 1. Update Pass Dependencies

In `lib/Conversion/TxnToFunc/TxnToFuncPass.cpp`:

```cpp
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

void ConvertTxnToFuncPass::runOnOperation() {
  // Add CF dialect as legal
  target.addLegalDialect<mlir::cf::ControlFlowDialect>();
  // ... existing code
}
```

### 2. Replace IfToSCFIfPattern

Create a new `IfToCFPattern` that converts `txn.if` to `cf.cond_br`:

```cpp
class IfToCFPattern : public mlir::OpConversionPattern<sharp::txn::IfOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      sharp::txn::IfOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    
    // Get current block and split it at the txn.if position
    auto *currentBlock = rewriter.getInsertionBlock();
    auto splitPoint = rewriter.getInsertionPoint();
    auto *mergeBlock = rewriter.splitBlock(currentBlock, splitPoint);
    
    // Create blocks for then and else cases
    auto *thenBlock = rewriter.createBlock(mergeBlock);
    auto *elseBlock = rewriter.createBlock(mergeBlock);
    
    // Insert conditional branch in current block
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<mlir::cf::CondBranchOp>(
        op.getLoc(), adaptor.getCondition(), thenBlock, elseBlock);
    
    // Process then region
    rewriter.setInsertionPointToStart(thenBlock);
    for (auto &nestedOp : op.getThenRegion().front().without_terminator()) {
      rewriter.clone(nestedOp);
    }
    // Handle then terminator
    auto *thenTerminator = op.getThenRegion().front().getTerminator();
    if (isa<sharp::txn::YieldOp>(thenTerminator)) {
      rewriter.create<mlir::cf::BranchOp>(op.getLoc(), mergeBlock);
    }
    // txn.abort will be handled by AbortToReturnPattern
    
    // Process else region if it exists
    if (!op.getElseRegion().empty()) {
      rewriter.setInsertionPointToStart(elseBlock);
      for (auto &nestedOp : op.getElseRegion().front().without_terminator()) {
        rewriter.clone(nestedOp);
      }
      auto *elseTerminator = op.getElseRegion().front().getTerminator();
      if (isa<sharp::txn::YieldOp>(elseTerminator)) {
        rewriter.create<mlir::cf::BranchOp>(op.getLoc(), mergeBlock);
      }
    } else {
      // Empty else - branch directly to merge
      rewriter.setInsertionPointToStart(elseBlock);
      rewriter.create<mlir::cf::BranchOp>(op.getLoc(), mergeBlock);
    }
    
    // Continue with merge block
    rewriter.setInsertionPointToStart(mergeBlock);
    rewriter.eraseOp(op);
    
    return mlir::success();
  }
};
```

### 3. Update Terminator Patterns

Modify `YieldToSCFYieldPattern` to avoid conflicts:
- Add check: `if (op->getParentOfType<sharp::txn::IfOp>()) return failure();`
- Let the new `IfToCFPattern` handle `txn.yield` within `txn.if` operations

## Testing

Validate the implementation with:
- `test/Conversion/TxnToFunc/will-fire-guards.mlir`

Expected behavior:
```mlir
// Input
txn.if %cond {
  %c20 = arith.constant 20 : i32
  txn.call @reg::@write(%c20) : (i32) -> ()
  txn.yield
} else {
  txn.abort
}

// Output
cf.cond_br %cond, ^then, ^else
^then:
  %c20 = arith.constant 20 : i32
  call @reg_write(%c20) : (i32) -> ()
  cf.br ^merge
^else:
  %true = arith.constant true : i1
  func.return %true
^merge:
  // continue...
```

## Benefits

1. **Correct Semantics**: Handles early returns properly
2. **Architectural Alignment**: Matches hardware control flow patterns
3. **Robustness**: Supports all termination pattern combinations

## Status

- **Current**: Not implemented - causes conversion failures with mixed terminators
- **Priority**: High - blocks complex control flow patterns
- **Dependencies**: None - CF dialect is standard MLIR