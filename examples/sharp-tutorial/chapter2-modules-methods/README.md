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

### Action Methods
- **State-modifying**: Can change module state
- **Have side effects**: May affect module behavior
- **Subject to scheduling**: Conflicts determine execution order
- **Optional return values**: Use `txn.yield` for void returns

## Example: A Configurable Counter

Let's build a more sophisticated counter with multiple methods:

### counter.mlir

```mlir
// A counter with increment, decrement, and configurable step
txn.module @Counter {
  // State: current value and step size
  %value = txn.instance @value of @Register<i32> : !txn.module<"Register">
  %step = txn.instance @step of @Register<i32> : !txn.module<"Register">
  
  // Value method: read current count
  txn.value_method @getValue() -> i32 {
    %val = txn.call @value::@read() : () -> i32
    txn.return %val : i32
  }
  
  // Value method: read step size
  txn.value_method @getStep() -> i32 {
    %val = txn.call @step::@read() : () -> i32
    txn.return %val : i32
  }
  
  // Action method: increment by step
  txn.action_method @increment() {
    %current = txn.call @value::@read() : () -> i32
    %step_val = txn.call @step::@read() : () -> i32
    %new = arith.addi %current, %step_val : i32
    txn.call @value::@write(%new) : (i32) -> ()
    txn.yield
  }
  
  // Action method: decrement by step
  txn.action_method @decrement() {
    %current = txn.call @value::@read() : () -> i32
    %step_val = txn.call @step::@read() : () -> i32
    %new = arith.subi %current, %step_val : i32
    txn.call @value::@write(%new) : (i32) -> ()
    txn.yield
  }
  
  // Action method: set custom step
  txn.action_method @setStep(%new_step: i32) {
    txn.call @step::@write(%new_step) : (i32) -> ()
    txn.yield
  }
  
  // Action method: reset to zero
  txn.action_method @reset() {
    %zero = arith.constant 0 : i32
    txn.call @value::@write(%zero) : (i32) -> ()
    txn.yield
  }
  
  // Schedule with conflict information
  txn.schedule [@getValue, @getStep, @increment, @decrement, @setStep, @reset] {
    conflict_matrix = {
      // Value methods don't conflict with each other
      "getValue,getStep" = 3 : i32,    // CF
      
      // Value methods can run with any action
      "getValue,increment" = 3 : i32,   // CF
      "getValue,decrement" = 3 : i32,   // CF
      "getValue,setStep" = 3 : i32,     // CF
      "getValue,reset" = 3 : i32,       // CF
      "getStep,increment" = 3 : i32,    // CF
      "getStep,decrement" = 3 : i32,    // CF
      "getStep,setStep" = 3 : i32,      // CF
      "getStep,reset" = 3 : i32,        // CF
      
      // Actions conflict with each other
      "increment,decrement" = 2 : i32,  // C
      "increment,setStep" = 2 : i32,    // C
      "increment,reset" = 2 : i32,      // C
      "decrement,setStep" = 2 : i32,    // C
      "decrement,reset" = 2 : i32,      // C
      "setStep,reset" = 2 : i32         // C
    }
  }
}
```

## Method Parameters

Methods can accept parameters:
- Parameters are typed: `%param: i32`
- Multiple parameters supported: `@method(%a: i32, %b: i64)`
- Parameters are available as SSA values in method body

## Working with Data Types

Sharp supports standard integer types:
- `i1` - Boolean (1 bit)
- `i8` - Byte (8 bits)
- `i16` - Short (16 bits) 
- `i32` - Integer (32 bits)
- `i64` - Long (64 bits)

## Building and Testing

### 1. Check the module syntax
```bash
sharp-opt counter.mlir
```

### 2. Infer missing conflicts
```bash
sharp-opt counter.mlir --sharp-infer-conflict-matrix
```

Notice how the pass fills in missing conflict relations!

### 3. Generate and run simulation
```bash
../../tools/generate-workspace.sh counter.mlir counter_sim
cd counter_sim && mkdir build && cd build
cmake .. && make
./Counter_sim --cycles 10 --verbose
```

## Exercises

1. **Add a multiply method**: Create an action method that multiplies the counter by a given factor
2. **Add bounds checking**: Create value methods `isAtMax()` and `isAtMin()` that check bounds
3. **Create a bidirectional counter**: Add a direction register and make increment/decrement respect it
4. **Experiment with conflicts**: What happens if you mark conflicting actions as CF?

## Advanced Topics

### Method Composition
Methods can call other methods within the same module:
```mlir
txn.action_method @doubleIncrement() {
  txn.call @this.increment() : () -> ()
  txn.call @this.increment() : () -> ()
  txn.yield
}
```

### Conditional Execution
While not shown here, methods can include control flow (covered in later chapters).

## Key Takeaways

- Modules encapsulate related state and behavior
- Value methods provide safe concurrent reads
- Action methods manage state changes with conflict control
- The conflict matrix ensures deterministic hardware behavior
- Method parameters enable flexible, reusable components

## Next Chapter

Chapter 3 will dive deeper into scheduling and conflict resolution, showing how to:
- Understand conflict inference rules
- Design for maximum concurrency
- Use timing attributes for multi-cycle operations