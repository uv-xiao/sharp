# Chapter 2: Modules and Methods

## Overview

In this chapter, we'll explore Sharp's module system in depth, learning how to:
- Create modules with multiple methods
- Understand the difference between value and action methods
- Pass parameters to methods
- Return values from methods
- Work with different data types

## Value Methods vs Action Methods

### Value Methods
- **Read-only**: Cannot modify module state
- **Pure functions**: Always return the same value for the same inputs
- **Can be called concurrently**: Multiple value methods can execute simultaneously
- **Must have return values**: Always return data
- **No conflicts**: Value methods have no conflicts with any other method, and they don't appear in the schedule.

### Action Methods
- **State-modifying**: Can change module state
- **Have side effects**: May affect module behavior
- **Subject to scheduling**: Conflicts determine execution order
- **Optional return values**: Use `txn.return` without arguments for void returns

## Example: A Configurable Counter (counter.mlir)

Let's build a more sophisticated counter with multiple methods:


## Working with Data Types

Sharp supports standard integer types from MLIR builtin, as well as FIRRTL types.

## Building and Testing

### 1. Check the module syntax
```bash
sharp-opt counter.mlir
```

### 2. Infer missing conflicts
```bash
sharp-opt counter.mlir --sharp-primitive-gen --sharp-infer-conflict-matrix
```

Notice how the pass fills in missing conflict relations!

Acturally, action `@increment` and `@setStep` are conflict-free, but if user specifies a conflict relation, the pass will respect it.

## Key Takeaways

- Modules encapsulate related state and behavior
- Value methods provide safe concurrent reads
- Action methods manage state changes with conflict control
- The conflict matrix ensures deterministic hardware behavior
- Method parameters enable flexible, reusable components