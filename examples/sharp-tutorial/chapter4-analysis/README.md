# Chapter 4: Analysis Passes

## Overview

Sharp provides powerful analysis passes to verify and optimize your hardware designs. This chapter covers:
- Conflict matrix inference
- Combinational loop detection
- Pre-synthesis checking
- Reachability analysis

## Available Analysis Passes

### 1. Conflict Matrix Inference
**Pass**: `--sharp-infer-conflict-matrix`

Automatically infers conflict relationships between methods based on:
- Shared resource access
- Read/write patterns
- Control flow dependencies

### 2. Combinational Loop Detection
**Pass**: `--sharp-detect-combinational-loops`

Identifies cycles in combinational logic that would cause:
- Unstable hardware
- Simulation divergence
- Synthesis failures

### 3. Pre-Synthesis Check
**Pass**: `--sharp-pre-synthesis-check`

Verifies designs are synthesizable by checking for:
- Non-synthesizable primitives (spec types)
- Multi-cycle operations
- Unsupported constructs

### 4. Reachability Analysis
**Pass**: `--sharp-reachability-analysis`

Computes conditions under which methods can be called:
- Adds condition operands to method calls
- Helps optimize scheduling
- Enables dead code elimination

## Example: Analyzing a Complex Module

Let's create a module with potential issues to demonstrate analysis:

### complex_module.mlir

```mlir
// A module with various analysis challenges
txn.module @ComplexModule {
  // State elements
  %data = txn.instance @data of @Register<i32> : !txn.module<"Register">
  %flag = txn.instance @flag of @Register<i1> : !txn.module<"Register">
  %temp = txn.instance @temp of @Wire<i32> : !txn.module<"Wire">
  
  // Action that reads and writes same register
  txn.action_method @readModifyWrite(%delta: i32) {
    %current = txn.call @data::@read() : () -> i32
    %new = arith.addi %current, %delta : i32
    txn.call @data::@write(%new) : (i32) -> ()
    txn.yield
  }
  
  // Action that might have conflicts
  txn.action_method @conditionalUpdate(%cond: i1, %value: i32) {
    %flag_val = txn.call @flag::@read() : () -> i1
    %should_update = arith.andi %cond, %flag_val : i1
    // In real hardware, would use conditional logic
    txn.call @data::@write(%value) : (i32) -> ()
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
  
  // Action with potential combinational loop
  txn.action_method @updateFlag() {
    %data_val = txn.call @data::@read() : () -> i32
    %zero = arith.constant 0 : i32
    %is_zero = arith.cmpi eq, %data_val, %zero : i32
    txn.call @flag::@write(%is_zero) : (i1) -> ()
    txn.yield
  }
  
  // Partial schedule - let inference complete it
  txn.schedule [@readModifyWrite, @conditionalUpdate, @getProcessed, @updateFlag] {
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

### 2. Check for Combinational Loops

Create a module with a loop:

### loop_example.mlir

```mlir
// Module with combinational loop through wires
txn.module @LoopExample {
  %wire_a = txn.instance @wire_a of @Wire<i32> : !txn.module<"Wire">
  %wire_b = txn.instance @wire_b of @Wire<i32> : !txn.module<"Wire">
  
  // Creates a->b->a loop
  txn.rule @loop_rule_a {
    %b_val = txn.call @wire_b::@read() : () -> i32
    %one = arith.constant 1 : i32
    %inc = arith.addi %b_val, %one : i32
    txn.call @wire_a::@write(%inc) : (i32) -> ()
    txn.yield
  }
  
  txn.rule @loop_rule_b {
    %a_val = txn.call @wire_a::@read() : () -> i32
    %two = arith.constant 2 : i32
    %double = arith.muli %a_val, %two : i32
    txn.call @wire_b::@write(%double) : (i32) -> ()
    txn.yield
  }
  
  txn.schedule [@loop_rule_a, @loop_rule_b]
}
```

Run loop detection:
```bash
sharp-opt loop_example.mlir --sharp-detect-combinational-loops
```

### 3. Pre-Synthesis Checking

Example with non-synthesizable elements:

### non_synth.mlir

```mlir
// Module using spec primitives (not synthesizable)
txn.module @NonSynthesizable {
  // This would fail synthesis - spec primitives are for verification only
  %spec_fifo = txn.instance @spec_fifo of @SpecFIFO<i32> : !txn.module<"SpecFIFO">
  
  txn.action_method @useSpecFIFO(%data: i32) {
    txn.call @spec_fifo::@enqueue(%data) : (i32) -> ()
    txn.yield
  }
  
  txn.schedule [@useSpecFIFO]
}
```

Check synthesizability:
```bash
sharp-opt non_synth.mlir --sharp-pre-synthesis-check
```

## Building and Testing

Create a run script to test all analyses:

### run.sh

```bash
#!/bin/bash

SHARP_ROOT="../../.."
SHARP_OPT="$SHARP_ROOT/build/bin/sharp-opt"

echo "=== Chapter 4: Analysis Passes ==="
echo ""

echo "1. Testing conflict matrix inference:"
echo "----------------------------------------"
$SHARP_OPT complex_module.mlir --sharp-infer-conflict-matrix 2>&1 | grep -A 20 "conflict_matrix"
echo ""

echo "2. Testing combinational loop detection:"
echo "----------------------------------------"
if $SHARP_OPT loop_example.mlir --sharp-detect-combinational-loops 2>&1 | grep -q "error"; then
    echo "✅ Loop detected as expected"
else
    echo "❌ Loop detection failed"
fi
echo ""

echo "3. Testing pre-synthesis check:"
echo "----------------------------------------"
# Note: This will fail because SpecFIFO isn't implemented yet
echo "Skipping - SpecFIFO not yet implemented"
echo ""

echo "4. Testing valid module analysis:"
echo "----------------------------------------"
$SHARP_OPT complex_module.mlir --sharp-infer-conflict-matrix --sharp-pre-synthesis-check > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Valid module passes all checks"
else
    echo "❌ Analysis failed"
fi
```

## Understanding Analysis Results

### Conflict Matrix Output
The inferred matrix shows relationships:
- `0` (SB): First sequences before second
- `1` (SA): First sequences after second
- `2` (C): Methods conflict
- `3` (CF): Conflict-free

### Loop Detection Messages
Look for cycles in the dependency graph:
```
error: combinational loop detected: wire_a -> wire_b -> wire_a
```

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

- Analysis passes catch errors early in development
- Conflict inference saves manual specification effort
- Loop detection prevents hardware bugs
- Pre-synthesis checks avoid late-stage failures
- Analysis results guide optimization

## Next Chapter

Chapter 5 explores translation passes:
- Converting Txn to FIRRTL
- Generating Verilog
- Handling primitives in translation
- Verification of translated designs