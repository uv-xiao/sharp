# Sharp Execution Model

This document describes the execution model for Sharp, inspired by Koika's one-rule-at-a-time semantics with method extensions for modular hardware design.

## Overview

Sharp adopts a **one-rule-at-a-time (1RaaT)** execution model where actions execute atomically and sequentially within each clock cycle. This model provides:

1. **Sequential Semantics**: Actions appear to execute in program order
2. **Atomic Execution**: Each action completes entirely before the next begins
3. **Deterministic Behavior**: No race conditions or timing ambiguities
4. **Modular Composition**: Methods enable hierarchical design

## Core Concepts

### Terminology

- **Action**: Either a rule or an action method. These are the schedulable units that can modify state.
- **Rule**: A guarded atomic action defined within a module that executes autonomously
- **Action Method**: A method that can modify state, callable from parent modules
- **Value Method**: A pure function that reads state without side effects (not schedulable)

### Rules

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
   - Must be conflict-free with all other actions
   - Not included in schedules
   - Example: Wire's read cannot be a value method since "read SA write" is required

2. **Action Methods**: Procedures that can modify state atomically
   - Included in schedules alongside rules
   - Can be called by parent modules

```mlir
txn.value_method @getValue() -> i32 {
    %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
    %state = txn.call @reg.read() : () -> i32
    txn.return %state
}

txn.action_method @setValue(%v: i32) {
    %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
    txn.call @reg.write(%v) : (i32) -> ()
}
```

### Scheduling

The schedule specifies the execution order of **actions only** (rules and action methods). Value methods are not included in schedules.

Within each cycle, actions execute based on:

1. **Explicit Ordering**: User-specified schedule constraints
2. **Conflict Relations**: 
   - **SB (Sequenced Before)**: Action A must execute before Action B
   - **SA (Sequenced After)**: Action A must execute after Action B  
   - **C (Conflict)**: Actions cannot execute in the same cycle
   - **CF (Conflict-Free)**: Actions can execute in any order

### Method Call Restrictions

1. Actions cannot call other actions in the same module
2. Actions can call value methods in the same module
3. Actions can call methods (both value and action) of child module instances

## Execution Semantics

### Single Cycle Execution

Each clock cycle follows these phases (note: there is no scheduling phase since the schedule is already specified in the MLIR file):

1. **Value Phase**:
   - Calculate the value of all value methods
   - The values remain unchanged until the next cycle

2. **Execution Phase**:
   ```
   for each scheduled action in order:
       if the action is an action method:
           stall until this action method is enabled by an action from the parent module 
           or all actions calling this action method have aborted in the current cycle
           check guard and conflict matrix
           execute the action method and record aborting or success (return value)
       if the action is a rule:
           check guard and conflict matrix
           execute the rule and record aborting or success
   ```

3. **Commit Phase**:
   - Apply all state updates due to recorded execution success
   - Advance to next cycle

### Method Call Semantics

Method calls within actions follow these rules:

1. **Value Methods**:
   - Must be conflict-free with all other actions
   - Can be called multiple times
   - Always return consistent values within a cycle (calculated once in Value Phase)
   - No side effects on state

2. **Action Methods**:
   - Execute atomically as part of calling action
   - Can only be called once per cycle per instance
   - State changes visible to subsequent operations within the same action

### Multi-Cycle Execution

Multi-cycle operations allow actions to span multiple clock cycles. The execution model extends as follows:

1. **Value Phase**: Same as single cycle execution

2. **Execution Phase**:
   ```
   for each scheduled action in order:
       if the action is single-cycle:
           // Same as single cycle execution
       if the action is multi-cycle:
           // History updates
           for every execution that started in the past but not finished:
               update inner execution status
               check if a new launch can be triggered in the current cycle
                   if yes: trigger the launch
               record panic if a required action fails in the current cycle 
                   (conflict or guard violation) -- only static launch can cause panic
               record the execution success in the current cycle
           
           // New execution starts
           if the action is an action method:
               stall until this action method is enabled by an action from the parent module 
               or all actions calling this action method have aborted in the current cycle
               check guard and conflict matrix
               try execute the "per-cycle actions" in the current cycle
               start a new execution if no aborting
           if the action is a rule:
               check guard and conflict matrix
               try execute the "per-cycle actions" in the current cycle
               start a new execution if no aborting
   ```

3. **Commit Phase**: Same as single cycle execution

### Launch Operations

Multi-cycle actions can contain per-cycle actions and launches. Launch operations allow deferred execution with dependencies:

```mlir
txn.instance @reg of @Register<i32> : !txn.module<"Register">
txn.instance @fifo of @SpecFIFO<i32> : !txn.module<"SpecFIFO">

txn.action_method @multiCycleAction(%data: i32) {multicycle = true} {
    // Per-cycle actions execute immediately
    txn.call @reg.write(%data) : (i32) -> ()
    
    // Multi-cycle actions are enclosed by txn.future
    txn.future {
      // Static latency launch
      %done1 = txn.launch {latency=3} {
          // This block executes 3 cycles later
          // If after 3 cycles, this block fails, a panic will be raised
          txn.call @reg.write(%data) : (i32) -> ()
      }
      
      // Dynamic latency launch
      %done2 = txn.launch until %done1 {
          // This launch only starts when %done1 is true
          // If the enqueue fails, NO panic will be raised
          // Instead, the block will be tried again in the next cycle until it succeeds
          txn.call @fifo.enqueue(%data) : (i32) -> ()
      }

      // Combined: dependency with static latency
      %done3 = txn.launch until %done2 {latency=1} {
        // This launch only starts 1 cycle after %done2 is true
        // If the launch fails, a panic will be raised (due to the static latency)
        // ...
      }
    }
}

txn.rule @multiCycleRule {multicycle = true} {
    // Rules can also be multi-cycle
    txn.future {
      // Contains one or multiple launches, similar to action methods
    }
}
```

**Key Semantics**:
- Static launches (`{latency=n}`) must succeed after the specified delay or panic
- Dynamic launches (`until %cond`) retry until successful
- Launches can depend on completion of previous launches
- Per-cycle actions execute before any launches start

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
2. **Value methods must be conflict-free**: Value methods cannot have conflicts with any actions
3. **Value-Value**: Value methods never conflict with each other

## Implementation Strategy

### Transaction-Level Simulation

Sharp implements a complete event-driven simulation infrastructure with the 1RaaT execution model:

```cpp
class Simulator {
    void executeCycle() {
        // Phase 1: Value Phase - Calculate all value method results
        for (auto& valueMethod : module->valueMethods) {
            valueMethodCache[valueMethod->name] = valueMethod->compute();
        }
        
        // Phase 2: Execution Phase - Execute actions in schedule order
        for (auto& actionName : module->schedule) {
            auto* action = module->getAction(actionName);
            
            if (action->isActionMethod()) {
                // Wait for parent module enablement
                if (!action->isEnabled() && !allCallersAborted(action)) {
                    continue; // Stall
                }
            }
            
            // Check guard and conflicts
            if (action->checkGuard() && !hasConflict(action)) {
                ExecutionResult result = action->execute();
                recordResult(action, result);
                
                // Handle multi-cycle actions
                if (action->isMultiCycle()) {
                    handleMultiCycleExecution(action);
                }
            }
        }
        
        // Phase 3: Commit Phase - Apply successful state updates
        commitSuccessfulUpdates();
        advanceTime();
    }
};
```

**Key Features**:
- **Event-driven architecture** with dependency tracking
- **Performance metrics** collection (cycles, method calls, conflicts)
- **Breakpoint support** for debugging
- **Multi-cycle operations** through continuation events

### RTL Translation

Sharp provides complete translation to synthesizable hardware through FIRRTL:

1. **Guard Evaluation**: Combinational logic for all guards
2. **Conflict Resolution**: Will-fire signals with conflict matrix checking
3. **Sequential Execution**: Ready/enable protocol for methods
4. **State Updates**: Register writes with proper clock/reset

**Translation Pipeline**:
```bash
# Generate Verilog through FIRRTL
sharp-opt input.mlir --txn-export-verilog -o output.v

# Or step-by-step:
sharp-opt input.mlir --convert-txn-to-firrtl | \
  circt-opt --lower-firrtl-to-hw | \
  circt-opt --export-verilog -o output.v
```

**Key Features**:
- Automatic conflict matrix inference
- Parametric primitive instantiation
- Method interface generation
- Proper clock/reset handling

## Examples

### Simple Counter

```mlir
txn.module @Counter {
    %count = txn.instance @count of @Register<i32> : !txn.module<"Register">
    
    txn.value_method @getValue() -> i32 {
        %v = txn.call @count.read() : () -> i32
        txn.return %v : i32
    }
    
    txn.action_method @increment() {
        %v = txn.call @count.read() : () -> i32
        %one = arith.constant 1 : i32
        %next = arith.addi %v, %one : i32
        txn.call @count.write(%next) : (i32) -> ()
        txn.return
    }
    
    txn.rule @autoIncrement {
        %v = txn.call @getValue() : () -> i32
        %max = arith.constant 100 : i32
        %cond = arith.cmpi ult, %v, %max : i32
        txn.if %cond {
            txn.call @increment() : () -> ()
        }
        txn.yield
    }
    
    txn.schedule [@increment, @autoIncrement]  
}
```

### Pipeline Stage

```mlir
txn.module @PipelineStage {
    %valid = txn.instance @valid of @Register<i1> : !txn.module<"Register">
    %data = txn.instance @data of @Register<i32> : !txn.module<"Register">
    
    txn.action_method @enqueue(%v: i32) {
        %is_empty = txn.call @valid.read() : () -> i1
        %true = arith.constant true : i1
        %not_valid = arith.xori %is_empty, %true : i1
        txn.if %not_valid {
            txn.call @data.write(%v) : (i32) -> ()
            txn.call @valid.write(%true) : (i1) -> ()
        }
        txn.return
    }
    
    txn.action_method @dequeue() -> i32 {
        %is_valid = txn.call @valid.read() : () -> i1
        %result = txn.call @data.read() : () -> i32
        txn.if %is_valid {
            %false = arith.constant false : i1
            txn.call @valid.write(%false) : (i1) -> ()
        }
        txn.return %result : i32
    }
    
    txn.schedule [@enqueue, @dequeue]
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

## Advanced Simulation Modes

### Concurrent Simulation (DAM Methodology)

Sharp implements DAM (Discrete-event simulation with Adaptive Multiprocessing) for high-performance multi-module simulation:

```cpp
// Each module runs in its own context with local time
class Context {
    uint64_t localTime;
    EventQueue localQueue;
    
    void run() {
        while (!done) {
            auto event = localQueue.pop();
            localTime = event->time;
            event->execute();
        }
    }
};
```

**Key Principles**:
- **Asynchronous time**: No global synchronization barrier
- **Time-bridging channels**: Handle inter-module communication
- **Lazy synchronization**: Only sync when necessary
- **Thread affinity**: Pin contexts to CPU cores for performance

### RTL Simulation Integration

Sharp integrates with CIRCT's arcilator for cycle-accurate RTL simulation:

```bash
# Convert to Arc dialect and simulate
sharp-opt input.mlir --sharp-arcilator -o arc.mlir
arcilator arc.mlir --trace  # Generates VCD waveforms
```

### Hybrid TL-RTL Simulation

Mix transaction-level and RTL modules with configurable synchronization:

```mlir
sim.bridge_config @bridge {
    sim.tl_module @TLProducer
    sim.rtl_module @RTLConsumer
    sim.sync_mode "lockstep"  // or "decoupled", "adaptive"
}
```

**Synchronization Modes**:
- **Lockstep**: TL and RTL advance together
- **Decoupled**: Allow bounded time divergence
- **Adaptive**: Dynamically adjust based on activity

### JIT Compilation

Experimental JIT mode for long-running simulations:

```bash
# Compile and execute directly
sharp-opt input.mlir --sharp-simulate=mode=jit
```

## Performance Optimization

### Simulation Statistics

All simulation modes collect comprehensive metrics:
- Total cycles executed
- Method call counts and conflicts
- Rule firing patterns
- Event queue depths
- For concurrent: speedup metrics

### Optimization Techniques

1. **Conflict Caching**: Precompute and cache conflict checks
2. **Event Batching**: Group non-conflicting events
3. **Parallel Rule Evaluation**: Use SIMD for guard evaluation
4. **Memory Pooling**: Reuse event objects

## Future Extensions

1. **Formal Verification Integration**: Connect to model checkers
2. **Hardware-in-the-Loop**: Co-simulation with FPGA prototypes
3. **Distributed Simulation**: Multi-machine simulation for large designs
4. **Incremental Compilation**: Faster iteration cycles