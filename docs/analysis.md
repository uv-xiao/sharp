# Sharp Analysis Passes

## Overview

Sharp provides several analysis passes to validate, optimize, and prepare Txn modules for conversion to other dialects. These passes implement the theoretical foundations described in the execution model.

## Core Analysis Passes

### Conflict Matrix Inference (`--sharp-infer-conflict-matrix`)
**Implementation**: `lib/Analysis/ConflictMatrixInference.cpp`

**Code Positions**: 
- Lines 135-170: `applyInferenceRules` function
- Lines 42-43: ConflictMatrix type definition
- Lines 138-141: Self-conflict rule implementation

Automatically infers missing conflict relationships between actions in a schedule.

**Algorithm**:
1. **Self-conflict rule**: Any action conflicts with itself (A,A = C)
2. **Symmetry enforcement**: If A SA B and B SB A, mark as conflict
3. **Method-based inference**: If action A calls method M1 and action B calls method M2, and M1 conflicts with M2, then A conflicts with B
4. **Transitivity**: Propagate conflict relations through chains

**Input**: Schedule with partial conflict matrix
**Output**: Schedule with complete conflict matrix

### Reachability Analysis (`--sharp-reachability-analysis`)
**Implementation**: `lib/Analysis/ReachabilityAnalysis.cpp`

**Code Positions**:
- Lines 60-87: `ReachabilityAnalysisPass` class definition
- Lines 89-104: Main `runOnOperation` function
- Lines 44-58: `ReachabilityState` struct for tracking conditions
- Lines 82-86: `updateCallOp` function to add condition operands

Computes reachability conditions for method calls within actions by tracking control flow through `txn.if` operations.

**Algorithm**:
1. Walk through each action (rule/action method)
2. Track path conditions through `txn.if` branches
3. For each `txn.call`, compute the condition under which it's reachable
4. Add condition operands to `txn.call` operations

**Key for conflict_inside calculation**: Enables precise tracking of which method calls can actually execute simultaneously.

### Action Scheduling (`--sharp-action-scheduling`)
**Implementation**: `lib/Analysis/ActionScheduling.cpp`

Automatically generates valid schedules for modules with incomplete or missing schedules.

**Algorithm**:
1. Collect all actions (rules and action methods) from module
2. Build dependency graph based on conflict matrix
3. Apply topological sort with conflict constraints
4. Generate schedule that respects SB/SA ordering

**Modes**:
- **Optimal**: Minimize total schedule length
- **Heuristic**: Fast scheduling for large modules

### Combinational Loop Detection (`--sharp-detect-combinational-loops`)
**Implementation**: `lib/Analysis/CombinationalLoopDetection.cpp`

Detects circular dependencies through combinational paths that would create invalid hardware.

**Algorithm**:
1. Build signal dependency graph
2. Identify combinational paths (value methods, Wire primitives)
3. Detect cycles using depth-first search
4. Report complete cycle paths for debugging

### Method Attribute Validation (`--sharp-validate-method-attributes`)
**Implementation**: `lib/Analysis/MethodAttributeValidation.cpp`

Validates signal naming attributes and method constraints for FIRRTL generation.

**Checks**:
- Signal name uniqueness within modules
- Valid attribute combinations
- Timing constraint consistency

### Pre-Synthesis Check (`--sharp-pre-synthesis-check`)
**Implementation**: `lib/Analysis/PreSynthesisCheck.cpp`

Validates that modules can be successfully synthesized to hardware.

**Validation Rules**:
- No unsupported operations in synthesizable modules
- Proper clocking and reset handling
- Valid primitive instantiation

## Specialized Analysis

### Value Method Conflict Check
**Implementation**: `lib/Analysis/ValueMethodConflictCheck.cpp`

Ensures value methods maintain conflict-free property with all actions.

### Action Call Validation
**Implementation**: `lib/Analysis/ActionCallValidation.cpp`

Validates that action methods are called correctly according to execution model constraints.

## Integration with Conversion

These analysis passes prepare modules for conversion:

1. **ConflictMatrixInference** � Complete conflict matrices for will-fire logic
2. **ReachabilityAnalysis** � Condition operands for conflict_inside calculation  
3. **ActionScheduling** � Valid schedules for FIRRTL module generation
4. **PreSynthesisCheck** � Ensures synthesizability before TxnToFIRRTL

## Example Usage

```bash
# Complete analysis pipeline
sharp-opt input.mlir \
  --sharp-infer-conflict-matrix \
  --sharp-reachability-analysis \
  --sharp-action-scheduling \
  --sharp-pre-synthesis-check \
  -o analyzed.mlir
```

## Implementation Notes

- All passes preserve the original module structure
- Analysis results are stored as attributes on operations
- Passes can be run independently or in pipeline
- Error reporting includes location information for debugging