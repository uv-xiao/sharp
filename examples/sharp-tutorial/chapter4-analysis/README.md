# Chapter 4: Analysis Passes

## Overview

Sharp provides powerful analysis passes to verify and optimize your hardware designs. This chapter covers:
- General semantic validation
- Conflict matrix inference
- Reachability analysis  
- Pre-synthesis checking

## Available Analysis Passes

### 1. General Check
**Pass**: `--sharp-general-check`

Comprehensive semantic validation for all Sharp Txn modules:
- **Schedule Validation**: Ensures schedules only contain actions (not value methods)
- **Value Method Constraints**: Validates value methods are conflict-free with all actions
- **Action Call Validation**: Ensures actions don't call other actions in the same module

Run this pass on any Sharp module to verify it follows the core execution model.

### 2. Conflict Matrix Inference
**Pass**: `--sharp-infer-conflict-matrix`

Automatically infers conflict relationships between methods based on:
- Shared resource access
- Read/write patterns
- Control flow dependencies

### 2. Pre-Synthesis Check
**Pass**: `--sharp-pre-synthesis-check`

Verifies designs are synthesizable by checking for:
- Non-synthesizable primitives (spec types)
- Multi-cycle operations
- Unsupported constructs

### 3. Action Scheduling
**Pass**: `--sharp-action-scheduling`

Automatically generates complete schedules for modules with incomplete or missing schedules:
- **Dependency Analysis**: Analyzes conflict relationships between actions
- **Topological Sorting**: Orders actions to respect conflict constraints  
- **Complete Schedule Generation**: Ensures all actions are included in the schedule
- **Conflict Matrix Integration**: Works with inferred conflict matrices

**When to Use:**
- Modules have incomplete schedules missing some actions
- Modules have no schedule but contain actions
- Quick prototyping where manual scheduling is tedious

### 4. Reachability Analysis
**Pass**: `--sharp-reachability-analysis`

Computes conditions under which methods can be called:
- Tracks control flow through `txn.if` statements
- Identifies when method calls can reach abort conditions
- Adds condition operands to method calls based on reachability
- Critical for will-fire logic generation in FIRRTL conversion
- Enables optimization by identifying unreachable code paths

## Examples: Analyzing Modules

### Example 1: Schedule Completeness Validation

Let's first demonstrate schedule completeness validation with an incomplete schedule:

#### incomplete_schedule.mlir

```mlir
txn.module @IncompleteScheduleExample {
  %data = txn.instance @data of @Register<i32> : index
  %flag = txn.instance @flag of @Register<i1> : index
  
  // Action method 1
  txn.action_method @processData(%value: i32) {
    %current = txn.call @data::@read() : () -> i32
    %sum = arith.addi %current, %value : i32
    txn.call @data::@write(%sum) : (i32) -> ()
    txn.yield
  }
  
  // Action method 2 (MISSING from schedule)
  txn.action_method @updateFlag() {
    %data_val = txn.call @data::@read() : () -> i32
    %zero = arith.constant 0 : i32
    %is_positive = arith.cmpi sgt, %data_val, %zero : i32
    txn.call @flag::@write(%is_positive) : (i1) -> ()
    txn.yield
  }
  
  // Rule (MISSING from schedule)
  txn.rule @defaultRule {
    %flag_val = txn.call @flag::@read() : () -> i1
    txn.if %flag_val {
      %zero = arith.constant 0 : i32
      txn.call @data::@write(%zero) : (i32) -> ()
      txn.yield
    } else {
      txn.yield
    }
    txn.yield
  }
  
  // INCOMPLETE SCHEDULE: Missing @updateFlag and @defaultRule
  txn.schedule [@processData] {
    conflict_matrix = {
      "processData,processData" = 2 : i32  // C
    }
  }
}
```

**Testing Schedule Completeness:**
```bash
# This will FAIL - incomplete schedule detected
sharp-opt incomplete_schedule.mlir \
  --sharp-primitive-gen \
  --sharp-infer-conflict-matrix \
  --sharp-reachability-analysis \
  --sharp-general-check

# Error output:
# error: [GeneralCheck] Schedule completeness validation failed - incomplete schedule: 
# schedule in module 'IncompleteScheduleExample' is missing 2 action(s): [defaultRule, updateFlag]
# Reason: Schedules must include ALL actions (rules and action methods) in the module.
```

**Fixing with Action Scheduling:**
```bash
# This will SUCCEED - action-scheduling fixes incomplete schedule
sharp-opt incomplete_schedule.mlir \
  --sharp-primitive-gen \
  --sharp-infer-conflict-matrix \
  --sharp-reachability-analysis \
  --sharp-action-scheduling \
  --sharp-general-check
```

The `--sharp-action-scheduling` pass automatically:
1. Identifies missing actions (`@updateFlag`, `@defaultRule`)
2. Analyzes conflict relationships between all actions
3. Generates a complete schedule including all actions
4. Updates conflict matrix for the complete action set

### Example 2: Complex Module Analysis

Let's create a module with various analysis challenges to demonstrate all passes:

### complex_module.mlir

```mlir
// A module with various analysis challenges
txn.module @ComplexModule {
  // State elements
  %data = txn.instance @data of @Register<i32> : index
  %flag = txn.instance @flag of @Register<i1> : index
  %temp = txn.instance @temp of @Wire<i32> : index
  
  // Action that reads and writes same register
  txn.action_method @readModifyWrite(%delta: i32) {
    %current = txn.call @data::@read() : () -> i32
    %new = arith.addi %current, %delta : i32
    txn.call @data::@write(%new) : (i32) -> ()
    txn.yield
  }
  
  // Action with conditional execution and potential abort
  txn.action_method @conditionalUpdate(%cond: i1, %value: i32) {
    %flag_val = txn.call @flag::@read() : () -> i1
    %should_update = arith.andi %cond, %flag_val : i1
    
    txn.if %should_update {
      // Only write if conditions are met
      txn.call @data::@write(%value) : (i32) -> ()
      txn.yield
    } else {
      // Abort if conditions not satisfied
      txn.abort
    }
    txn.yield
  }
  
  // Value method using wire
  txn.value_method @getProcessed() -> i32 {
    %data_val = txn.call @data::@read() : () -> i32
    %two = arith.constant 2 : i32
    %doubled = arith.muli %data_val, %two : i32
    txn.call @temp::@write(%doubled) : (i32) -> ()
    %result = txn.call @temp::@read() : () -> i32
    txn.return %result : i32
  }
  
  // Action with nested conditionals and multiple abort paths
  txn.action_method @updateFlag() {
    %data_val = txn.call @data::@read() : () -> i32
    %zero = arith.constant 0 : i32
    %max_val = arith.constant 100 : i32
    
    %is_zero = arith.cmpi eq, %data_val, %zero : i32
    %is_too_large = arith.cmpi sgt, %data_val, %max_val : i32
    
    txn.if %is_too_large {
      // Invalid data - abort immediately
      txn.abort
    } else {
      txn.if %is_zero {
        // Set flag to true for zero value
        %true = arith.constant true
        txn.call @flag::@write(%true) : (i1) -> ()
        txn.yield
      } else {
        // Check if value is positive
        %is_positive = arith.cmpi sgt, %data_val, %zero : i32
        txn.if %is_positive {
          // Set flag to false for positive value
          %false = arith.constant false
          txn.call @flag::@write(%false) : (i1) -> ()
          txn.yield
        } else {
          // Negative value - abort
          txn.abort
        }
        txn.yield
      }
      txn.yield
    }
    txn.yield
  }
  
  // Action that demonstrates complex reachability analysis
  txn.action_method @complexProcessor(%enable: i1, %threshold: i32) {
    %flag_val = txn.call @flag::@read() : () -> i1
    %data_val = txn.call @data::@read() : () -> i32
    
    txn.if %enable {
      txn.if %flag_val {
        // Path 1: Both enable and flag are true
        %exceeds_threshold = arith.cmpi sgt, %data_val, %threshold : i32
        txn.if %exceeds_threshold {
          // Reachable only when: enable AND flag AND (data > threshold)
          %doubled = arith.muli %data_val, %data_val : i32
          txn.call @data::@write(%doubled) : (i32) -> ()
          txn.yield
        } else {
          // Reachable only when: enable AND flag AND (data <= threshold)
          txn.call @temp::@write(%data_val) : (i32) -> ()
          txn.yield
        }
        txn.yield
      } else {
        // Path 2: Enable true but flag false - conditional abort
        %is_negative = arith.cmpi slt, %data_val, %threshold : i32
        txn.if %is_negative {
          // Abort if data is negative when flag is false
          txn.abort
        } else {
          // Safe path when enable=true, flag=false, data>=0
          %incremented = arith.addi %data_val, %threshold : i32
          txn.call @data::@write(%incremented) : (i32) -> ()
          txn.yield
        }
        txn.yield
      }
      txn.yield
    } else {
      // Path 3: Enable false - always abort
      txn.abort
    }
    txn.yield
  }
  
  // Partial schedule - let inference complete it (note: getProcessed is a value method and not scheduled)
  txn.schedule [@readModifyWrite, @conditionalUpdate, @updateFlag, @complexProcessor] {
    conflict_matrix = {
      // Only specify some conflicts
      "readModifyWrite,conditionalUpdate" = 2 : i32  // C
    }
  }
}
```

## Running Analysis Passes

### 1. Infer Complete Conflict Matrix

```bash
sharp-opt complex_module.mlir --sharp-infer-conflict-matrix
```

Expected output shows inferred conflicts:
- Methods accessing same registers conflict
- Wire accesses properly ordered  
- Missing conflicts filled in

### 2. Analyze Reachability Conditions

```bash
sharp-opt complex_module.mlir --sharp-reachability-analysis
```

Reachability analysis tracks complex control flow:
- **@conditionalUpdate**: Write call reachable when `should_update` is true
- **@updateFlag**: Multiple paths with different abort conditions:
  - Immediate abort when `data > 100`
  - Normal execution when `data == 0` (sets flag to true)
  - Normal execution when `0 < data <= 100` (sets flag to false)
  - Abort when `data < 0`
- **@complexProcessor**: Demonstrates nested conditional reachability:
  - Data write reachable when: `enable AND flag AND (data > threshold)`
  - Temp write reachable when: `enable AND flag AND (data <= threshold)`
  - Alternative data write when: `enable AND !flag AND (data >= threshold)`
  - Abort when: `!enable OR (enable AND !flag AND data < threshold)`

**Why Reachability Analysis Matters:**
- **FIRRTL Conversion**: Essential for generating correct will-fire logic that prevents actions from executing when they would abort
- **Optimization**: Identifies unreachable code paths that can be eliminated
- **Verification**: Helps detect potential deadlocks or unreachable states
- **Simulation**: Enables more efficient simulation by tracking execution conditions

**Interpreting Reachability Output:**
When you run reachability analysis, you'll see method calls annotated with conditions:
```mlir
// Before reachability analysis:
txn.call @data::@write(%value) : (i32) -> ()

// After reachability analysis:
txn.call @data::@write if %condition : i1 then(%value) : (i32) -> ()
```
The `if %condition : i1 then` clause shows the exact condition under which this call is reachable.

### 3. General Check

**Pass**: `--sharp-general-check`

Comprehensive semantic validation for all Sharp Txn modules:
- **Schedule Validation**: Ensures schedules only contain actions (not value methods)  
- **Schedule Completeness**: Validates schedules include ALL actions in the module
- **Value Method Constraints**: Validates value methods are conflict-free with all actions
- **Action Call Validation**: Ensures actions don't call other actions in the same module

Run this pass on any Sharp module to verify it follows the core execution model.

**New: Schedule Completeness Validation**
The general check now ensures that every action (rule or action method) in a module appears in the schedule. Incomplete schedules lead to unscheduled actions that cannot execute, violating Sharp's execution model.

### 4. Pre-Synthesis Checking

**Pass**: `--sharp-pre-synthesis-check`

Comprehensive synthesis validation that identifies constructs preventing hardware conversion:

**Validation Categories:**
1. **Spec Primitives**: `SpecFIFO`, `SpecMemory` are simulation-only
2. **Disallowed Operations**: Floating-point math, complex arithmetic  
3. **Non-Hardware Dialects**: Operations outside the synthesis allowlist
4. **Method Attributes**: Signal name conflicts, invalid prefix/postfix combinations
5. **Hierarchical Violations**: Using non-synthesizable child modules

This enhanced pass now includes method attribute validation (formerly `--sharp-validate-method-attributes`).

### non_synthesizable.mlir

```mlir
// Multiple pre-synthesis violations for testing
txn.module @NonSynthesizable {
  // Violation 1: Using spec primitives (simulation-only)
  %spec_fifo = txn.instance @spec_fifo of @SpecFIFO<i32> : index
  %spec_mem = txn.instance @spec_mem of @SpecMemory<i32> : index
  
  txn.action_method @useSpecPrimitives(%data: i32) {
    txn.call @spec_fifo::@enqueue(%data) : (i32) -> ()
    %val = txn.call @spec_mem::@read() : () -> i32
    txn.yield
  }
  
  txn.schedule [@useSpecPrimitives]
}

txn.module @DisallowedOperations {
  %reg = txn.instance @reg of @Register<f32> : index
  
  txn.action_method @useFloatingPoint(%x: f32, %y: f32) {
    // Violation 2: Floating-point operations not in synthesis allowlist
    %sum = arith.addf %x, %y : f32
    %sin_val = math.sin %x : f32  // math dialect not allowed
    txn.call @reg::@write(%sum) : (f32) -> ()
    txn.yield
  }
  
  txn.schedule [@useFloatingPoint]
}
```

Check for violations:
```bash
sharp-opt non_synthesizable.mlir --sharp-pre-synthesis-check
```

Expected errors:
- `error: operation 'arith.addf' is not allowed in synthesizable code`
- `error: operation 'math.sin' is not allowed in synthesizable code`
- `error: operation 'arith.sitofp' is not allowed in synthesizable code`
- `error: Module 'DisallowedOperations' is non-synthesizable: contains non-synthesizable operations`

## Building and Testing

Create a run script to test all analyses:

### run.sh

```bash
#!/bin/bash

SHARP_ROOT="../../.."
SHARP_OPT="$SHARP_ROOT/build/bin/sharp-opt"

echo "=== Chapter 4: Analysis Passes ==="
echo ""

echo "1. Testing primitive generation (required first):"
echo "----------------------------------------"
if $SHARP_OPT complex_module.mlir --sharp-primitive-gen > /dev/null 2>&1; then
    echo "✅ Primitive generation completed successfully"
else
    echo "❌ Primitive generation failed"
fi
echo ""

echo "2. Testing conflict matrix inference (requires primitive-gen):"
echo "----------------------------------------"
$SHARP_OPT complex_module.mlir --sharp-primitive-gen --sharp-infer-conflict-matrix 2>&1 | grep -A 20 "conflict_matrix"
echo ""

echo "3. Testing reachability analysis (requires primitive-gen):"
echo "----------------------------------------"
echo "Running reachability analysis to track conditional execution..."
$SHARP_OPT complex_module.mlir --sharp-primitive-gen --sharp-reachability-analysis 2>&1 | grep -A 5 -B 2 "if.*then" | head -15
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ Reachability analysis completed successfully"
else
    echo "❌ Reachability analysis failed"
fi
echo ""

echo "4. Testing general semantic validation (requires conflict-matrix + reachability):"
echo "----------------------------------------"
if $SHARP_OPT complex_module.mlir --sharp-primitive-gen --sharp-infer-conflict-matrix --sharp-reachability-analysis --sharp-general-check > /dev/null 2>&1; then
    echo "✅ Module passes general semantic validation"
else
    echo "❌ General semantic validation failed"
fi
echo ""

echo "5. Testing pre-synthesis check (requires general-check):"
echo "----------------------------------------"
if $SHARP_OPT complex_module.mlir --sharp-primitive-gen --sharp-infer-conflict-matrix --sharp-reachability-analysis --sharp-general-check --sharp-pre-synthesis-check 2>&1 | grep -q "error"; then
    echo "❌ Unexpected synthesis errors"
else
    echo "✅ Valid module is synthesizable"
fi
echo ""

echo "6. Testing dependency enforcement (should fail):"
echo "----------------------------------------"
echo "Testing general-check without dependencies (should fail):"
if $SHARP_OPT complex_module.mlir --sharp-general-check 2>&1 | grep -q "missing dependency"; then
    echo "✅ Dependency enforcement working correctly"
else
    echo "❌ Dependency enforcement failed"
fi
echo ""

echo "7. Testing pre-synthesis violations:"
echo "----------------------------------------"
echo "Testing disallowed operation violations:"
if $SHARP_OPT non_synthesizable.mlir --sharp-primitive-gen --sharp-infer-conflict-matrix --sharp-reachability-analysis --sharp-general-check --sharp-pre-synthesis-check 2>&1 | grep -q "non-synthesizable"; then
    echo "✅ Non-synthesizable modules detected correctly"
else
    echo "❌ Failed to detect non-synthesizable modules"
fi
echo ""

echo "8. Testing complete analysis pipeline (correct order):"
echo "----------------------------------------"
$SHARP_OPT complex_module.mlir --sharp-primitive-gen --sharp-infer-conflict-matrix --sharp-reachability-analysis --sharp-general-check --sharp-pre-synthesis-check > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Valid module passes all analysis checks in correct order"
else
    echo "❌ Analysis pipeline failed"
fi
```

## Understanding Analysis Results

### Conflict Matrix Output
The inferred matrix shows relationships:
- `0` (SB): First sequences before second
- `1` (SA): First sequences after second
- `2` (C): Methods conflict
- `3` (CF): Conflict-free

### Synthesis Check Results
Identifies non-synthesizable constructs:
```
error: spec primitive 'SpecFIFO' cannot be synthesized
```

## Exercises

1. **Create a deadlock scenario**: Design methods with circular SB/SA dependencies
2. **Optimize conflicts**: Refactor a module to minimize conflicts
3. **Add custom analysis**: What other properties would you analyze?

## Advanced Topics

### Custom Analysis Passes
You can create your own analysis passes:
```cpp
struct MyAnalysisPass : public OperationPass<ModuleOp> {
  void runOnOperation() override {
    // Your analysis logic
  }
};
```

### Analysis-Driven Optimization
Use analysis results to:
- Reorder operations for better performance
- Eliminate dead code
- Merge compatible methods

## Key Takeaways

- **General Check catches semantic errors**: The new consolidated general check validates core execution model constraints across all Sharp code
- **Value methods are not scheduled**: Only actions (rules and action methods) can appear in schedules - value methods are automatically computed once per cycle
- **Conflict inference saves effort**: Automatically completes partial conflict matrices based on method calls and primitive semantics
- **Reachability analysis enables optimization**: Tracks conditional execution paths for better will-fire logic generation
- **Pre-synthesis checks avoid late failures**: Comprehensive validation ensures modules can be converted to hardware before attempting synthesis
- **Analysis results guide optimization**: Use analysis output to understand conflicts, reachability, and synthesizability constraints
