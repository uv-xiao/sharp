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

Sharp supports multi-cycle operations through two mechanisms:

#### 1. Timing Attributes (Currently Supported)
```mlir
txn.action_method @multiCycleOp(%data: i32) attributes {timing = "static(3)"} {
    // Static latency operations via timing attributes
    // The FIRRTL conversion handles the delay
    %result = txn.instance @result of @Register<i32> : !txn.module<"Register">
    txn.call @result.write(%data) : (i32) -> ()
    txn.return
}

txn.action_method @dynamicOp(%data: i32) attributes {timing = "dynamic"} {
    // Dynamic operations can use spec primitives
    %fifo = txn.instance @fifo of @SpecFIFO<i32> : !txn.module<"SpecFIFO">
    txn.call @fifo.enqueue(%data) : (i32) -> ()
    txn.return
}
```

#### 2. Launch Operations (Future Enhancement)
```mlir
txn.action_method @futureMultiCycle(%data: i32) {
    // Static latency launch (not yet implemented)
    txn.launch {latency=3} {
        // This block executes 3 cycles later
        %result = txn.instance @result of @Register<i32> : !txn.module<"Register">
        txn.call @result.write(%data) : (i32) -> ()
    }
    
    // Dynamic latency launch (not yet implemented)
    txn.launch until %done {
        %ready = txn.call @isReady() : () -> i1
        txn.if %ready {
            txn.call @process(%data) : (i32) -> ()
        }
        txn.return %ready : i1
    }
}
```

**Simulation Support**:
- Static timing through `"static(n)"` attributes (current)
- txn.launch for explicit multi-cycle blocks (planned)
- Dynamic timing through spec primitives
- Continuation events for deferred execution
- Proper dependency tracking across cycles

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

Sharp implements a complete event-driven simulation infrastructure with the 1RaaT execution model:

```cpp
class Simulator {
    void executeCycle() {
        // Phase 1: Scheduling - Determine which rules can fire
        std::vector<Event*> scheduledEvents;
        for (auto& rule : module->rules) {
            if (rule->guard() && !hasConflict(rule, scheduledEvents)) {
                scheduledEvents.push_back(createEvent(rule));
            }
        }
        
        // Phase 2: Execution - Execute rules atomically
        for (auto* event : scheduledEvents) {
            event->execute();
            if (event->hasContinuation()) {
                // Multi-cycle operations create continuation events
                scheduleEvent(event->getContinuation(), event->getDelay());
            }
        }
        
        // Phase 3: Commit - Apply all state updates atomically
        commitTransactions();
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
    
    txn.schedule [@getValue, @increment, @autoIncrement]  
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