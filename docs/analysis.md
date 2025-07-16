# Sharp Analysis Passes

## Overview

Sharp provides a comprehensive analysis framework to validate, optimize, and prepare Txn modules for conversion to other dialects (FIRRTL, Verilog). The analysis infrastructure implements the theoretical foundations described in the execution model and is organized around a mandatory core pipeline with additional utility passes.

The framework features robust error handling through the `AnalysisError` utility, strict dependency enforcement, and comprehensive state tracking to ensure correct analysis ordering.

## Core Analysis Pipeline

The Sharp analysis framework requires passes to be run in a specific order to ensure proper dependency resolution. Each pass validates its prerequisites and adds completion attributes for dependency tracking.

### 1. Primitive Generation (`--sharp-primitive-gen`)
**Implementation**: `lib/Analysis/PrimitiveGen.cpp`  
**Dependencies**: None (foundation pass)  
**Adds**: `sharp.primitive_gen_complete` attribute

Automatically generates missing primitive definitions for all primitives referenced in instance operations. This is the foundation pass that must run before any other analysis.

**Supported Primitives**:
- `Register<T>`: Single-port register with value method `read` and action method `write`
- `Wire<T>`: Combinational wire with action methods `read` and `write` (read SA write)
- `FIFO<T>`: Queue with enq/deq/first/notEmpty/notFull methods
- `Memory<T>`: Address-based storage with read/write/clear methods
- `SpecFIFO<T>`: Unbounded FIFO for verification (spec primitive)
- `SpecMemory<T>`: Memory with configurable latency for verification (spec primitive)

**Algorithm**:
1. **Primitive Discovery**: Scans all `txn.instance` operations to find primitive references
2. **Existence Check**: Verifies if each referenced primitive already exists
3. **Type Parsing**: Extracts data types from parametric names (e.g., `Register<i32>`)
4. **Generation**: Calls appropriate constructor functions to create missing primitives

**Example**:
```mlir
// Before:
txn.instance @reg of @Register<i32> : !txn.module<"Register">

// After PrimitiveGen:
txn.primitive @Register<i32> type = "hw" interface = !txn.module<"Register<i32>"> {
  txn.fir_value_method @read() : () -> i32  // Value method (not scheduled)
  txn.fir_action_method @write() : (i32) -> ()  // Action method
  txn.clock_by @clk
  txn.reset_by @rst
  txn.schedule [@write] {conflict_matrix = {"write,write" = 2 : i32}}
}
```

### 2. Conflict Matrix Inference (`--sharp-infer-conflict-matrix`)
**Implementation**: `lib/Analysis/ConflictMatrixInference.cpp`  
**Dependencies**: Requires `sharp.primitive_gen_complete`  
**Adds**: `sharp.conflict_matrix_inferred` attribute

Automatically infers missing conflict relationships between actions in schedules by analyzing method calls and primitive conflict matrices.

**Conflict Relations**:
- **SB (0)**: Sequence Before
- **SA (1)**: Sequence After  
- **C (2)**: Conflict
- **CF (3)**: Conflict-Free (user-provided, cannot be overridden)
- **UNK (4)**: Unknown (can be overridden by inference)

**Inference Rules**:
1. **Self-conflict**: Any action conflicts with itself (A,A = C)
2. **Symmetry enforcement**: If A SA B and B SB A, mark as conflict
3. **Method-based inference**: If action A calls method M1 and action B calls method M2, and M1 conflicts with M2, then A conflicts with B
4. **Transitivity**: Propagate conflict relations through chains
5. **Default assignment**: UNK if no other relationship determined

**Key Features**:
- Reconstructs parametric primitive names from base name + type arguments
- Handles instance method calls (`@instance::@method`)
- Preserves user-provided CF relationships
- Processes modules in topological order (primitives first)

### 3. Reachability Analysis (`--sharp-reachability-analysis`)
**Implementation**: `lib/Analysis/ReachabilityAnalysis.cpp`  
**Dependencies**: Requires `sharp.primitive_gen_complete`  
**Adds**: `sharp.reachability_analyzed` attribute

Computes reachability conditions for method calls within actions by tracking control flow through `txn.if` operations.

**Algorithm**:
1. Walk through each action (rule/action method)
2. Track path conditions through `txn.if` branches
3. For each `txn.call` or `txn.abort`, compute the condition under which it's reachable
4. Add condition operands to `txn.call` and `txn.abort` operations

**Purpose**: Enables precise `conflict_inside`/`conflict_with_earlier`/`reach_abort` calculation for will-fire logic generation in FIRRTL conversion.

**Example**:
```mlir
txn.rule @example {
  %cond = arith.cmpi eq, %x, %c0 : i32
  txn.if %cond {
    txn.call @inst::@method1() : () -> ()  // reachability = %cond
  } else {
    txn.abort // reachability = !%cond
  }
}
```

### 4. General Semantic Validation (`--sharp-general-check`)
**Implementation**: `lib/Analysis/GeneralCheck.cpp`  
**Dependencies**: Requires both `sharp.conflict_matrix_inferred` and `sharp.reachability_analyzed`  
**Adds**: `sharp.general_checked` attribute

Comprehensive semantic validation that combines multiple validation checks for Sharp Txn modules according to the core execution model.

**Validation Checks**:
1. **Schedule Constraints**: Ensures schedules only contain actions (rules and action methods), not value methods
2. **Schedule Completeness**: Validates schedules include ALL actions in the module
3. **Value Method Conflicts**: Validates that value methods are conflict-free with all scheduled actions  
4. **Action Call Constraints**: Ensures actions cannot call other actions within the same module

**Replaces Individual Passes**:
- `--sharp-validate-schedule`
- `--sharp-check-value-method-conflicts` 
- `--sharp-validate-action-calls`

### 5. Pre-Synthesis Validation (`--sharp-pre-synthesis-check`)
**Implementation**: `lib/Analysis/PreSynthesisCheck.cpp`  
**Dependencies**: Requires `sharp.general_checked`  
**Adds**: `sharp.pre_synthesis_checked` attribute

Comprehensive synthesis validation that identifies any constructs preventing successful translation to FIRRTL/Verilog hardware.

**Validation Checks**:
1. **Primitive Synthesizability**: Detects non-synthesizable (spec) primitives like `SpecFIFO` and `SpecMemory`
2. **Operation Allowlist**: Validates only synthesizable operations from approved dialects are used
3. **Method Attribute Validation**: Validates signal naming and FIRRTL generation attributes
4. **Hierarchical Propagation**: Marks parent modules as non-synthesizable if they use non-synthesizable children

**Approved Operations**:
- **txn dialect**: All operations allowed
- **arith dialect**: Limited subset (addi, subi, muli, andi, ori, xori, cmpi, etc.)
- **Disallowed**: Floating-point operations, division, remainder, math dialect

**Replaces Individual Pass**:
- `--sharp-validate-method-attributes` (now integrated)

## Utility and Helper Passes

### Action Scheduling (`--sharp-action-scheduling`)
**Implementation**: `lib/Analysis/ActionScheduling.cpp`

Automatically generates complete schedules for modules with incomplete or missing schedules. Essential for fixing schedule completeness violations detected by `--sharp-general-check`.

**Algorithm**:
1. **Action Discovery**: Collect all actions (rules and action methods) from module
2. **Dependency Analysis**: Build dependency graph based on inferred conflict matrix  
3. **Optimization**: For small modules (≤10 actions), uses exact algorithm; for larger modules, uses heuristic
4. **Schedule Generation**: Create complete schedule that includes ALL actions

**Use Cases**:
- Fix incomplete schedules detected by general check
- Automatic scheduling during development
- Rapid prototyping with automatic conflict resolution

### Function Inlining (`--sharp-inline-functions`)
**Implementation**: `lib/Analysis/InlineFunctions.cpp`

Inlines all `txn.func_call` operations by replacing them with the body of the called function. Functions are syntax sugar for combinational logic and must be inlined before lowering.

**Process**:
1. Find all `txn.func_call` operations
2. Create mapping from function arguments to call operands
3. Clone function body, replacing arguments with operands
4. Replace `txn.return` with returned values
5. Remove unused `txn.func` operations

### Primitive Action Collection (`--sharp-collect-primitive-actions`)
**Implementation**: `lib/Analysis/CollectPrimitiveActions.cpp`

Collects all primitive action calls made by each action for use in dynamic mode of TxnToFIRRTL conversion. Adds `primitive_calls` attribute to each action containing the list of primitive instance paths.

**Tracing**:
- Direct primitive calls (e.g., `@reg::@write`)
- Indirect calls through module methods that eventually call primitives
- Enables fine-grained conflict detection at primitive action level

## Infrastructure and Support

### Error Handling and Diagnostics
**Implementation**: `lib/Analysis/AnalysisError.cpp`, `include/sharp/Analysis/AnalysisError.h`

Sharp features a structured error reporting system that provides consistent, actionable diagnostics across all analysis passes.

**AnalysisError Utility Features**:
- **Fluent API**: Chainable method calls for building structured error messages
- **Categorization**: `ValidationFailure`, `MissingDependency`, `SynthesisViolation`, etc.
- **Consistent Format**: `[PassName] Pass failed - Category: Details. Reason: Explanation. Solution: Action`
- **Location Tracking**: Precise source location information for debugging

**Example Usage**:
```cpp
return AnalysisError(operation, "GeneralCheck")
       .setCategory(ErrorCategory::ValidationFailure)
       .setDetails("schedule missing actions: [action1, action2]")
       .setReason("Schedules must include ALL actions in the module")
       .setSolution("Add missing actions or use sharp-action-scheduling")
       .emit(), failure();
```

### Pass Dependency Enforcement

Each pass validates its dependencies using module attributes and emits structured errors if prerequisites are missing:

**Error Format**:
```
[PassName] Pass failed - missing dependency: [dependency-pass] must be run before [current-pass].
[Detailed explanation of why dependency is needed].
Please run [dependency-pass] first to ensure [specific requirement].
```

**State Tracking Attributes**:
- `sharp.primitive_gen_complete` - Primitives generated
- `sharp.conflict_matrix_inferred` - Conflicts inferred
- `sharp.reachability_analyzed` - Reachability computed
- `sharp.general_checked` - Semantic validation passed
- `sharp.pre_synthesis_checked` - Synthesis validation passed

## Testing and Examples

### Chapter 4 Analysis Tutorial
**Location**: `examples/sharp-tutorial/chapter4-analysis/`

The tutorial provides comprehensive examples demonstrating the analysis pipeline:

**Test Files**:
- `complex_module.mlir`: Valid module demonstrating all analysis passes
- `non_synthesizable.mlir`: Module with spec primitives and disallowed operations
- `incomplete_schedule.mlir`: Module with missing actions in schedule

**Test Script**: `run.sh` demonstrates the complete analysis pipeline and validates:
- ✅ Primitive generation completes successfully
- ✅ Conflict matrix inference adds complete conflict relationships
- ✅ Reachability analysis adds condition operands
- ✅ General semantic validation passes
- ✅ Pre-synthesis check validates synthesizability
- ✅ Dependency enforcement catches missing prerequisites
- ✅ Schedule completeness validation detects incomplete schedules
- ✅ Action scheduling fixes incomplete schedules

### Running the Analysis Pipeline

**Complete Pipeline**:
```bash
sharp-opt input.mlir \
  --sharp-primitive-gen \
  --sharp-infer-conflict-matrix \
  --sharp-reachability-analysis \
  --sharp-general-check \
  --sharp-pre-synthesis-check \
  -o analyzed.mlir
```

**Development Validation**:
```bash
sharp-opt input.mlir \
  --sharp-primitive-gen \
  --sharp-infer-conflict-matrix \
  --sharp-reachability-analysis \
  --sharp-general-check \
  -o validated.mlir
```

**FIRRTL Conversion Pipeline**:
```bash
sharp-opt input.mlir \
  --sharp-primitive-gen \
  --sharp-infer-conflict-matrix \
  --sharp-reachability-analysis \
  --sharp-general-check \
  --sharp-pre-synthesis-check \
  --convert-txn-to-firrtl \
  -o converted.mlir
```

## Integration with Conversion

These analysis passes prepare modules for conversion:

1. **PrimitiveGen** → Ensures all referenced primitives have valid definitions
2. **ConflictMatrixInference** → Complete conflict matrices for will-fire logic
3. **ReachabilityAnalysis** → Condition operands for conflict_inside calculation  
4. **GeneralCheck** → Core semantic validation for execution model compliance
5. **PreSynthesisCheck** → Synthesis validation before TxnToFIRRTL conversion

## Deprecated Passes

The following individual passes have been consolidated into unified passes:

**Replaced by GeneralCheck**:
- `--sharp-validate-schedule` → use `--sharp-general-check`
- `--sharp-check-value-method-conflicts` → use `--sharp-general-check`  
- `--sharp-validate-action-calls` → use `--sharp-general-check`

**Replaced by PreSynthesisCheck**:
- `--sharp-validate-method-attributes` → use `--sharp-pre-synthesis-check`

## Implementation Notes

- All passes preserve the original module structure
- Analysis results are stored as attributes on operations
- Passes implement robust dependency checking with clear error messages
- Error reporting uses the structured `AnalysisError` utility for consistency
- The framework supports both individual pass execution and pipeline mode
- State tracking enables proper dependency resolution and incremental analysis