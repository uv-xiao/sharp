# Sharp Analysis Passes

## Overview

Sharp provides several analysis passes to validate, optimize, and prepare Txn modules for conversion to other dialects. These passes implement the theoretical foundations described in the execution model.

## Core Analysis Passes

### Conflict Matrix Inference (`--sharp-infer-conflict-matrix`)
**Implementation**: `lib/Analysis/ConflictMatrixInference.cpp`

**Work on**: `ModuleOp` and `PrimitiveOp`

**Code Positions**: 
- Lines 178-276: `applyInferenceRules` function
- Lines 38-44: ConflictRelation enum with UNK support
- Lines 183-186: Self-conflict rule implementation  
- Lines 426-465: `getMethodCalls` implementation for extracting method calls
- Lines 467-582: `getMethodConflict` implementation for primitive conflict lookup
- Lines 513-537: Parametric primitive name reconstruction

Automatically infers missing conflict relationships between actions in a schedule.

**Conflict Relations**:
- **SB (0)**: Sequence Before
- **SA (1)**: Sequence After  
- **C (2)**: Conflict
- **CF (3)**: Conflict-Free (user-provided, cannot be overridden)
- **UNK (4)**: Unknown (can be overridden by inference)

**Algorithm**:
1. **Self-conflict rule**: Any action conflicts with itself (A,A = C)
2. **Symmetry enforcement**: If A SA B and B SB A, mark as conflict
3. **Method-based inference**: If action A calls method M1 and action B calls method M2, and M1 C/SA/SB M2, then A C/SA/SB B
   - Analyzes action bodies to extract all method calls using `getMethodCalls`
   - Reconstructs parametric primitive names (e.g., "Register<i1>") from base name + type arguments
   - Looks up conflict relationships between methods from primitive/module conflict matrices using `getMethodConflict`
   - Propagates the strongest conflict relationship found between any pair of method calls
   - Properly handles instance method calls (`@instance::@method`)
   - Only applies conflicts from scheduled action methods (value methods are ignored)
4. **Transitivity**: Propagate conflict relations through chains
5. **Default assignment**: UNK if no other relationship determined

**Topological Processing**: Modules are processed in dependency order (primitives first, then modules that use them) to ensure primitive conflict matrices are available for inference.

**Integration with PrimitiveGen**: Run `--sharp-primitive-gen` before this pass to ensure all referenced primitives have valid definitions with conflict matrices.

**Conflict Override Logic**: 
- **Preserves user CF**: User-provided CF (3) relationships cannot be overridden
- **Overrides UNK only**: Only UNK (4) relationships can be replaced with inferred conflicts
- **Creates UNK by default**: New relationships start as UNK (4) and can be strengthened by inference

**Input**: Schedule with partial conflict matrix
**Output**: Schedule with complete conflict matrix

**Working Example** (Toggle module):
```mlir
// Input:
txn.module @Toggle {
  %0 = txn.instance @state of @Register<i1> : !txn.module<"Register">
  txn.action_method @toggle() {
    %1 = txn.call @state::@read() : () -> i1  // Value method call
    %2 = arith.xori %1, %true : i1
    txn.call @state::@write(%2) : (i1) -> ()  // Action method call
  }
  txn.rule @default {
    %1 = txn.call @state::@read() : () -> i1  // Value method call
    txn.call @state::@write(%1) : (i1) -> ()  // Action method call
  }
  txn.schedule [@toggle, @default]  // No conflict matrix specified
}

// Output after PrimitiveGen + ConflictMatrixInference:
txn.schedule [@toggle, @default] {
  conflict_matrix = {
    "default,default" = 2 : i32,    // Self-conflict (C)
    "default,toggle" = 2 : i32,     // Conflict due to shared write method (C)
    "toggle,default" = 2 : i32,     // Conflict due to shared write method (C)
    "toggle,toggle" = 2 : i32       // Self-conflict (C)
  }
}

// Reasoning: Both actions call @state::@write, and Register primitive defines
// write,write = C, so actions conflict through method-based inference.
```

### Primitive Generation (`--sharp-primitive-gen`)
**Implementation**: `lib/Analysis/PrimitiveGen.cpp`

**Code Positions**:
- Lines 47-85: `generateMissingPrimitives` function
- Lines 103-129: `generatePrimitive` function  
- Lines 131-167: `parsePrimitiveName` function for type parsing

Automatically generates missing primitive definitions for all primitives referenced in instance operations.

**Algorithm**:
1. **Primitive Discovery**: Scans all `txn.instance` operations to find primitive references
2. **Existence Check**: Verifies if each referenced primitive already exists as a `txn.module` or `txn.primitive`
3. **Generation**: For missing primitives, calls appropriate constructor function:
   - `Register<T>`: Single-port register with value method `read` (not scheduled) and action method `write`
   - `Wire<T>`: Combinational wire with action methods `read` and `write` (read SA write conflict)
   - `FIFO<T>`: Queue with enq/deq/first/notEmpty/notFull methods
   - `Memory<T>`: Address-based storage with read/write/clear methods
   - `SpecFIFO<T>`: Unbounded FIFO for verification
   - `SpecMemory<T>`: Memory with configurable latency
4. **Type Parsing**: Extracts data types from parametric names (e.g., `Register<i32>`)

**Key Primitive Semantics**:
- **Register**: `read` is a value method (combinational, not scheduled), `write` is an action method (conflicts with other writes)
- **Wire**: Both `read` and `write` are action methods with read SA write ordering constraint

**Critical Integration**: Must run before `--sharp-infer-conflict-matrix` to ensure primitive conflict matrices are available for analysis.

**Example**:
```mlir
// Before:
txn.instance @reg of @Register<i32> : !txn.module<"Register">
// Missing: Register<i32> primitive definition

// After PrimitiveGen:
txn.primitive @Register<i32> type = "hw" interface = !txn.module<"Register<i32>"> {
  txn.fir_value_method @read() : () -> i32  // Value method (not scheduled)
  txn.fir_action_method @write() : (i32) -> ()  // Action method
  txn.clock_by @clk
  txn.reset_by @rst
  txn.schedule [@write] {conflict_matrix = {"write,write" = 2 : i32}}  // Only action methods
}
```

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
# Complete analysis pipeline with primitive generation
sharp-opt input.mlir \
  --sharp-primitive-gen \
  --sharp-infer-conflict-matrix \
  --sharp-reachability-analysis \
  --sharp-action-scheduling \
  --sharp-pre-synthesis-check \
  -o analyzed.mlir

# Minimal conflict inference (most common usage)
sharp-opt input.mlir \
  --sharp-primitive-gen \
  --sharp-infer-conflict-matrix \
  -o with-conflicts.mlir

# Debug conflict inference
sharp-opt input.mlir \
  --sharp-primitive-gen \
  --sharp-infer-conflict-matrix \
  --debug-only=sharp-conflict-matrix-inference
```

## Implementation Notes

- All passes preserve the original module structure
- Analysis results are stored as attributes on operations
- Passes can be run independently or in pipeline
- Error reporting includes location information for debugging