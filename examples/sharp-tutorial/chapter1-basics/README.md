# Chapter 1: Basic Concepts

## Introduction

Sharp uses a transaction-based model for hardware description. In this model:
- Hardware is composed of **modules** containing state and behavior
- Behavior is expressed through **methods** that execute atomically
- Methods can be **value methods** (read-only) or **action methods** (state-modifying)
- **Rules** define autonomous behavior that executes when conditions are met

## Key Concepts

### 1. Atomic Transactions
Every method execution is atomic - it either completes fully or not at all. This provides:
- Predictable behavior
- No partial state updates
- Clear reasoning about concurrency

### 2. Conflict Relations
Methods can have four types of conflicts:
- **SB (0)**: Sequence Before - first must execute before second
- **SA (1)**: Sequence After - first must execute after second  
- **C (2)**: Conflict - cannot execute in same cycle
- **CF (3)**: Conflict Free - can execute in any order

### 3. One-Rule-at-a-Time (1RaaT) Semantics
Rules execute one at a time, ensuring deterministic behavior even with concurrent hardware.

## Your First Sharp Module

Let's create a simple toggle module:

### toggle.mlir

```mlir
// A module that toggles between 0 and 1
txn.module @Toggle {
  // We'll use a Register primitive to store state
  %state = txn.instance @state of @Register<i1> : !txn.module<"Register">
  
  // Value method to read the current state
  txn.value_method @read() -> i1 {
    %val = txn.call @state::@read() : () -> i1
    txn.return %val : i1
  }
  
  // Action method to toggle the state
  txn.action_method @toggle() {
    %current = txn.call @state::@read() : () -> i1
    %one = arith.constant 1 : i1
    %new = arith.xori %current, %one : i1
    txn.call @state::@write(%new) : (i1) -> ()
    txn.yield
  }
  
  // Schedule declares all methods/rules
  txn.schedule [@read, @toggle]
}
```

## Building and Running

### 1. Parse and Verify

```bash
sharp-opt toggle.mlir --mlir-print-op-generic
```

This shows the internal representation of your module.

### 2. Run Analysis

```bash
sharp-opt toggle.mlir --sharp-infer-conflict-matrix
```

This infers conflicts between methods. Since `toggle` modifies state that `read` accesses, they conflict.

### 3. Generate Simulation

```bash
../../tools/generate-workspace.sh toggle.mlir toggle_sim
cd toggle_sim
mkdir build && cd build
cmake .. && make
./Toggle_sim --cycles 10 --verbose
```

## Exercises

1. **Add a reset method**: Create an action method that sets the state to 0
2. **Add a set method**: Create an action method that sets the state to a given value
3. **Experiment with conflicts**: What happens if two methods try to write different values?

## Key Takeaways

- Sharp modules encapsulate state and behavior
- Methods provide atomic operations on that state
- The transaction model ensures predictable hardware behavior
- Conflict analysis helps verify correct concurrent execution

## Next Chapter

In Chapter 2, we'll explore more complex modules with multiple methods and learn about scheduling constraints.