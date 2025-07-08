# Sharp Execution Model

## Overview

Sharp adopts a **one-rule-at-a-time (1RaaT)** execution model where actions execute atomically and sequentially within each clock cycle, providing deterministic behavior without race conditions.

## Core Concepts

### Terminology
- **Action**: Rule or action method - the schedulable units that modify state
- **Rule**: Guarded atomic action executing autonomously
- **Action Method**: State-modifying method callable from parent modules  
- **Value Method**: Pure function reading state without side effects


### Scheduling and Conflicts

Conflict relations:
- **SB (0)**: Sequenced Before - A executes before B
- **SA (1)**: Sequenced After - A executes after B
- **C (2)**: Conflict - A and B cannot execute same cycle
- **CF (3)**: Conflict-Free - A and B can execute any order

Key constraints:
- Schedules contain only actions (rules/action methods)
- Value methods must be conflict-free with all actions
- Actions cannot call other actions in same module

## Multi-Cycle Operations

Launch operations enable deferred execution:

```mlir
txn.future {
  // Static latency - must succeed after delay or panic
  %done1 = txn.launch after 3 {
    txn.call @reg::@write(%data) : (i32) -> ()
    txn.yield
  }
  
  // Dynamic latency - retry until successful
  %done2 = txn.launch until %done1 {
    txn.call @fifo::@enqueue(%data) : (i32) -> ()
    txn.yield
  }
}
```

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


## Implementation Strategy

### Transaction-Level Simulation

**Implementation**: `lib/Simulation/Core/Simulator.cpp`, `lib/Simulation/Core/SimModule.cpp`

- Event-driven architecture with three-phase execution
- Performance metrics collection
- Multi-cycle support via continuations

**Concurrent mode** (DAM Methodology)

**Implementation**: `lib/Simulation/Concurrent/ConcurrentSimulator.cpp`, `lib/Simulation/Concurrent/Context.cpp`

Sharp implements DAM (Dataflow Abstract Machine) for high-performance multi-module simulation:

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

**Code Position**: Context.cpp implements this pattern

**Key Principles**:
- **Asynchronous time**: No global synchronization barrier
- **Time-bridging channels**: Handle inter-module communication
- **Lazy synchronization**: Only sync when necessary
- **Thread affinity**: Pin contexts to CPU cores for performance

### RTL Translation
- Automatic conflict matrix inference
- Will-fire signal generation
- Ready/enable method protocols
- Synthesizable FIRRTL/Verilog output

## Example

```mlir
txn.module @Counter {
  %count = txn.instance @count of @Register<i32>
  
  txn.value_method @getValue() -> i32 {
    %v = txn.call @count::@read() : () -> i32
    txn.return %v : i32
  }
  
  txn.rule @increment {
    %v = txn.call @getValue() : () -> i32
    %limit = arith.constant 100 : i32
    %cond = arith.cmpi ult, %v, %limit : i32
    txn.if %cond {
      %one = arith.constant 1 : i32
      %next = arith.addi %v, %one : i32
      txn.call @count::@write(%next) : (i32) -> ()
    }
    txn.yield
  }
  
  txn.schedule [@increment]
}
```

## Documentation vs Implementation Mismatches

### Issues Found

1. **Three-Phase Execution Model**
   - **Documentation**: Describes Value Phase → Execution Phase → Commit Phase
   - **Implementation**: Simulation code doesn't clearly implement this three-phase model
   - **Location**: `lib/Simulation/Core/Simulator.cpp` should implement this pattern
   - **Impact**: Execution semantics may not match specification

2. **Method Call Semantics**
   - **Documentation**: Value methods calculated once per cycle in Value Phase
   - **Implementation**: Current simulation may not enforce this constraint
   - **Impact**: Value method results might be inconsistent within a cycle

3. **Multi-Cycle Execution**
   - **Documentation**: Detailed multi-cycle execution model with launch operations
   - **Implementation**: Multi-cycle support appears incomplete in simulation infrastructure
   - **Location**: `lib/Simulation/` directories don't show complete multi-cycle implementation

4. **DAM Implementation**
   - **Documentation**: Describes sophisticated DAM methodology
   - **Implementation**: `lib/Simulation/Concurrent/` exists but may not fully implement described features
   - **Location**: ConcurrentSimulator.cpp and Context.cpp

### Missing Components

1. **Action Stalling Logic**: Documentation describes action method stalling but implementation unclear
2. **Abort Propagation**: Multi-cycle abort handling not fully implemented  
3. **Condition Tracking**: Launch until/after conditions not integrated with simulation
4. **Time Synchronization**: DAM time-bridging channels not fully implemented

### Fixes Needed

1. **Implement proper three-phase execution** in Simulator.cpp
2. **Add value method caching** to enforce once-per-cycle calculation
3. **Complete multi-cycle support** with launch operation handling
4. **Enhance DAM implementation** with proper time synchronization
5. **Add abort propagation** throughout execution pipeline