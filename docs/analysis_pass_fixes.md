# Analysis Pass Fixes

## Overview

This document describes required fixes for analysis passes to properly handle test dialect operations and ensure correct validation behavior.

## Current Problems

### 1. PreSynthesisCheck Pass

**Issue**: The `sharp-pre-synthesis-check` pass incorrectly flags `test.use` operations as non-synthesizable, causing validation errors in test files.

**Error**:
```
error: operation 'test.use' is not allowed in synthesizable code
```

**Root Cause**: The `isAllowedOperation` function only permits operations from `txn`, `firrtl`, `builtin`, and `arith` dialects, but excludes the `test` dialect used for testing purposes.

**Solution**: Modify `lib/Analysis/PreSynthesisCheckPass.cpp`:

```cpp
bool isAllowedOperation(mlir::Operation *op) {
  // Allow operations from standard synthesizable dialects
  auto dialect = op->getDialect();
  auto dialectName = dialect->getNamespace();
  
  return dialectName == "txn" || 
         dialectName == "firrtl" || 
         dialectName == "builtin" || 
         dialectName == "arith" ||
         dialectName == "test";  // Add test dialect for testing
}
```

### 2. Analysis Integration Test

**Issue**: The `analysis-integration.mlir` test expects certain validation errors that are not being produced by the current analysis passes.

**FileCheck Failures**: Multiple DAG (order-independent) patterns are not matching the current output, indicating either:
1. Analysis passes are not detecting expected violations
2. Output format has changed
3. Test expectations are outdated

## Required Implementation

### 1. Fix PreSynthesisCheck Pattern

The immediate fix is to allow `test` dialect operations in synthesizable code validation:

```cpp
// In lib/Analysis/PreSynthesisCheckPass.cpp
bool PreSynthesisCheckPass::isAllowedOperation(mlir::Operation *op) {
  auto dialectName = op->getDialect()->getNamespace();
  
  // Standard synthesizable dialects
  if (dialectName == "txn" || dialectName == "firrtl" || 
      dialectName == "builtin" || dialectName == "arith") {
    return true;
  }
  
  // Test dialect operations are allowed for testing purposes
  if (dialectName == "test") {
    return true;
  }
  
  return false;
}
```

### 2. Analysis Pass Output Investigation

The analysis integration test needs investigation to determine:

1. **Which specific analysis is failing**: Run individual analysis passes to isolate issues
2. **Output format changes**: Compare expected vs actual output patterns
3. **Validation logic updates**: Check if analysis behavior has changed

### 3. Test Pattern Updates

If analysis behavior is correct but output format changed, update FileCheck patterns:

```mlir
// Update patterns from:
// CHECK-DAG: some old pattern
// To:
// CHECK-DAG: some new pattern matching current output
```

## Investigation Steps

1. **Run individual analysis passes**:
   ```bash
   sharp-opt input.mlir --sharp-infer-conflict-matrix | FileCheck ...
   sharp-opt input.mlir --sharp-reachability-analysis | FileCheck ...
   sharp-opt input.mlir --sharp-validate-method-attributes | FileCheck ...
   sharp-opt input.mlir --sharp-pre-synthesis-check | FileCheck ...
   ```

2. **Compare outputs**: Examine what each pass produces vs what tests expect

3. **Update expectations**: Modify test patterns to match correct behavior

## Testing

Validate fixes with:
- `test/Analysis/analysis-integration.mlir`

Expected behavior after fix:
- `test.use` operations should not trigger synthesis errors
- All FileCheck patterns should match correctly
- Analysis passes should detect actual violations (not test artifacts)

## Files to Modify

1. `lib/Analysis/PreSynthesisCheckPass.cpp` - Add test dialect support
2. `test/Analysis/analysis-integration.mlir` - Update FileCheck patterns if needed

## Status

- **Current**: Not implemented - test dialect causes false positives
- **Priority**: Medium - affects test reliability
- **Dependencies**: None - straightforward fix