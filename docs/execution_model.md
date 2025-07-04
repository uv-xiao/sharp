# Sharp Execution Model

This document describes the execution model for Sharp, inspired by Koika's one-rule-at-a-time semantics with method extensions for modular hardware design.

## Overview

Sharp adopts a **one-rule-at-a-time (1RaaT)** execution model where rules execute atomically and sequentially within each clock cycle. This model provides:

1. **Sequential Semantics**: Rules appear to execute in program order
2. **Atomic Execution**: Each rule completes entirely before the next begins
3. **Deterministic Behavior**: No race conditions or timing ambiguities
4. **Modular Composition**: Methods enable hierarchical design

## Core Concepts

### Rules and Actions

A **rule** is a guarded atomic action that executes when its guard condition is true:

```mlir
txn.rule @doSomething {
    %guard = txn.call @canProceed() : () -> i1
    txn.if %guard {
        // Action body executes atomically
        %value = txn.call @getValue() : () -> i32
        txn.call @setValue(%value) : (i32) -> ()
    }
}
```

### Methods

Methods extend the basic 1RaaT model with modular interfaces:

1. **Value Methods**: Pure functions that read state without side effects
2. **Action Methods**: Procedures that can modify state atomically

```mlir
txn.value_method @getValue() -> i32 {
    %state = txn.read @register : !txn.ref<i32>
    txn.return %state
}

txn.action_method @setValue(%v: i32) {
    txn.write @register, %v : !txn.ref<i32>, i32
}
```

### Scheduling

Within each cycle, the scheduler determines rule execution order based on:

1. **Explicit Ordering**: User-specified schedule constraints
2. **Conflict Relations**: 
   - **SB (Sequenced Before)**: Rule A must execute before Rule B
   - **SA (Sequenced After)**: Rule A must execute after Rule B  
   - **C (Conflict)**: Rules cannot execute in the same cycle
   - **CF (Conflict-Free)**: Rules can execute in any order

## Execution Semantics

### Single Cycle Execution

Each clock cycle follows these phases:

1. **Scheduling Phase**:
   ```
   1. Evaluate all rule guards
   2. Select enabled rules based on guards and conflicts
   3. Determine execution order respecting SB/SA constraints
   ```

2. **Execution Phase**:
   ```
   for each scheduled rule in order:
       1. Read all required state atomically
       2. Execute action body (compute new values)
       3. Write all state updates atomically
   ```

3. **Commit Phase**:
   ```
   1. Apply all state updates simultaneously
   2. Advance to next cycle
   ```

### Method Call Semantics

Method calls within actions follow these rules:

1. **Value Methods**:
   - Can be called multiple times
   - Always return consistent values within a cycle
   - No side effects on state

2. **Action Methods**:
   - Execute atomically as part of calling rule
   - Can only be called once per cycle per instance
   - State changes visible to subsequent operations

### Multi-Cycle Operations

For modeling complex hardware operations that span multiple cycles:

```mlir
txn.action_method @multiCycleOp(%data: i32) {
    // Static latency (known at compile time)
    txn.launch {latency=3} {
        // This block executes 3 cycles later
        txn.write @result, %data : !txn.ref<i32>, i32
    }
}

txn.action_method @dynamicOp(%data: i32) {
    // Dynamic latency (determined at runtime)
    txn.launch until %done {
        %ready = txn.call @isReady() : () -> i1
        txn.if %ready {
            txn.write @result, %data : !txn.ref<i32>, i32
        }
        txn.return %ready : i1
    }
}
```

## Conflict Resolution

### Conflict Matrix

The scheduler uses a conflict matrix to determine compatible rule executions:

```
        R1   R2   R3
   R1   -    SB   CF
   R2   SA   -    C
   R3   CF   C    -
```

Where:
- `SB`: R1 must execute before R2
- `SA`: R2 must execute after R1  
- `C`: R2 and R3 cannot execute in same cycle
- `CF`: R1 and R3 can execute in any order

### Method Conflicts

Method calls introduce implicit conflicts:

1. **Action-Action**: Two calls to the same action method conflict
2. **Action-Value**: Action methods conflict with value methods they affect
3. **Value-Value**: Value methods never conflict

## Implementation Strategy

### Transaction-Level Simulation

At the transaction level, the execution model is implemented as:

```cpp
class Simulator {
    void executeCycle() {
        // Phase 1: Schedule
        auto schedule = scheduler.computeSchedule(rules, conflicts);
        
        // Phase 2: Execute
        for (auto& rule : schedule) {
            auto transaction = beginTransaction();
            rule.execute(transaction);
            transaction.commit();
        }
        
        // Phase 3: Advance
        advanceClock();
    }
};
```

### RTL Translation

For hardware synthesis, the execution model maps to:

1. **Guard Evaluation**: Combinational logic for all guards
2. **Conflict Resolution**: Priority encoders and arbitration
3. **Sequential Execution**: Muxing and state machines
4. **State Updates**: Register writes on clock edge

## Examples

### Simple Counter

```mlir
txn.module @Counter {
    txn.state @count : i32
    
    txn.value_method @getValue() -> i32 {
        %v = txn.read @count : !txn.ref<i32>
        txn.return %v : i32
    }
    
    txn.action_method @increment() {
        %v = txn.read @count : !txn.ref<i32>
        %one = arith.constant 1 : i32
        %next = arith.addi %v, %one : i32
        txn.write @count, %next : !txn.ref<i32>, i32
    }
    
    txn.rule @autoIncrement {
        %v = txn.call @getValue() : () -> i32
        %max = arith.constant 100 : i32
        %cond = arith.cmpi ult, %v, %max : i32
        txn.if %cond {
            txn.call @increment() : () -> ()
        }
    }
}
```

### Pipeline Stage

```mlir
txn.module @PipelineStage {
    txn.state @valid : i1
    txn.state @data : i32
    
    txn.action_method @enqueue(%v: i32) {
        %false = arith.constant false : i1
        %is_empty = txn.read @valid : !txn.ref<i1>
        %not_valid = arith.xori %is_empty, %false : i1
        txn.if %not_valid {
            txn.write @data, %v : !txn.ref<i32>, i32
            %true = arith.constant true : i1
            txn.write @valid, %true : !txn.ref<i1>, i1
        }
    }
    
    txn.action_method @dequeue() -> i32 {
        %is_valid = txn.read @valid : !txn.ref<i1>
        %result = txn.read @data : !txn.ref<i32>
        txn.if %is_valid {
            %false = arith.constant false : i1
            txn.write @valid, %false : !txn.ref<i1>, i1
        }
        txn.return %result : i32
    }
}
```

## Comparison with Other Models

### vs. Verilog Always Blocks
- Sharp: Atomic rule execution, no race conditions
- Verilog: Concurrent execution, potential races

### vs. SystemVerilog Assertions  
- Sharp: Rules are synthesizable hardware
- SVA: Primarily for verification, not synthesis

### vs. Chisel/FIRRTL
- Sharp: Explicit scheduling and conflicts
- Chisel: Implicit scheduling through connections

### vs. Bluespec
- Sharp: Similar atomic semantics, simpler conflict model
- Bluespec: Complex implicit conditions, compiler-driven scheduling

## Future Extensions

1. **Nested Transactions**: Allow sub-transactions within rules
2. **Priority Schemes**: User-defined rule priorities
3. **Fairness Guarantees**: Ensure rules get scheduled fairly
4. **Speculative Execution**: Tentative execution with rollback
5. **Distributed Scheduling**: Multi-clock domain coordination