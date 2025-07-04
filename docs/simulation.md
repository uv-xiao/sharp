# Sharp Simulation Framework Design

## Overview

This document outlines the design and implementation of Sharp's multi-level simulation framework, incorporating insights from EQueue (event-driven simulation), DAM (high-performance dataflow simulation), and Sharp's own transaction-based semantics. The framework supports simulation at multiple abstraction levels: transaction-level, RTL, and hybrid modes.

## Motivation and Goals

Sharp's simulation framework aims to:

1. **Enable rapid design exploration** through fast transaction-level simulation
2. **Support verification** with cycle-accurate RTL simulation via CIRCT's arcilator
3. **Allow progressive refinement** from high-level specs to detailed implementations
4. **Provide performance estimation** at various abstraction levels
5. **Enable mixed-level simulation** where different components run at different abstraction levels

## Key Design Principles

### 1. Separation of Concerns
- **Structure**: Hardware components and their connections
- **Behavior**: Transaction semantics and scheduling
- **Timing**: Performance models and cycle-level details
- **Verification**: Spec vs implementation checking

### 2. Event-Driven Architecture
Following EQueue's approach, we use discrete-event simulation where:
- Each module maintains an event queue
- Events are method calls or rule firings
- Events execute based on dependencies and conflicts
- Different modules can execute concurrently

### 3. Multi-Cycle Operations
Inspired by DAM and the requirements for spec primitives:
- Actions can span multiple cycles
- Actions can trigger other actions in sequence
- Proper handling of causality chains

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Sharp Simulation Framework            │
├─────────────────────────────────────────────────────────┤
│                    Simulation Engine API                 │
├─────────────────────────────────────────────────────────┤
│   Transaction-Level    │    RTL-Level    │    Hybrid    │
│      Simulator         │   (Arcilator)   │   Bridge     │
├─────────────────────────────────────────────────────────┤
│              Event Queue & Scheduling Core               │
├─────────────────────────────────────────────────────────┤
│         Performance Models & Timing Annotations          │
└─────────────────────────────────────────────────────────┘
```

## Transaction-Level Simulation

### Event Model

Each transaction-level event represents:
```cpp
struct Event {
    uint64_t time;           // Simulation time
    ModuleID module;         // Target module
    MethodID method;         // Method/rule to execute
    vector<Value> args;      // Arguments
    EventID dependencies[];  // Events that must complete first
    EventID triggers[];      // Events triggered by this one
};
```

### Module State

Each simulated module maintains:
```cpp
class SimModule {
    State state;                    // Module state variables
    EventQueue pending;             // Pending events
    ConflictMatrix conflicts;       // From txn.schedule
    map<MethodID, Function> impls;  // Method implementations
};
```

### Execution Model

1. **Event Scheduling**:
   - Events are added to the global event queue
   - Dependencies are tracked using DAG structure
   - Conflicts are resolved using the conflict matrix

2. **Event Execution**:
   ```
   while (!event_queue.empty()) {
       Event e = event_queue.pop_ready();
       
       // Check conflicts with executing events
       if (has_conflicts(e)) {
           defer_event(e);
           continue;
       }
       
       // Execute the event
       Result r = modules[e.module].execute(e.method, e.args);
       
       // Handle multi-cycle operations
       if (r.is_continuation) {
           schedule_continuation(e, r.next_cycle);
       }
       
       // Trigger dependent events
       for (auto triggered : e.triggers) {
           event_queue.add(triggered);
       }
   }
   ```

3. **Multi-Cycle Operations**:
   - Spec primitives can return continuations
   - Continuations are scheduled for future cycles
   - State changes are atomic at cycle boundaries

### Spec Primitives

Special primitives for modeling and verification:

```mlir
// Spec FIFO with unbounded capacity
%fifo = txn.primitive @SpecFIFO<i32> {
    txn.value_method @canEnq() -> i1 { ... }
    txn.action_method @enq(%data: i32) { ... }
    txn.value_method @canDeq() -> i1 { ... }
    txn.action_method @deq() -> i32 { ... }
}

// Multi-cycle spec action
txn.action_method @complexOperation(%n: i32) {
    // Cycle 1: Initiate
    %state = txn.spec.begin_multi_cycle()
    
    // Cycles 2-n: Process
    txn.spec.continue(%state) {
        // Trigger other actions
        txn.call @otherAction()
    }
    
    // Final cycle: Complete
    txn.spec.complete(%state)
}
```

## RTL-Level Simulation

For RTL simulation, Sharp leverages CIRCT's arcilator:

1. **Conversion Pipeline**:
   ```
   Txn → FIRRTL → HW → Arc → Arcilator
   ```

2. **Integration Points**:
   - Method interfaces become ports
   - Will-fire signals for scheduling
   - Ready/enable handshaking

3. **Cycle-Accurate Execution**:
   - Each clock cycle evaluates all combinational logic
   - State updates on clock edges
   - Precise timing matches hardware

## Hybrid Simulation

The hybrid mode allows mixing transaction-level and RTL simulation:

### Bridge Components

```mlir
// Bridge between TL and RTL domains
txn.bridge @TLtoRTL(%tl_module: !txn.module, %rtl_module: !hw.module) {
    // Map TL method calls to RTL signals
    txn.bridge.method @getValue() -> i32 {
        %ready = hw.read %rtl_module.getValue_ready
        %data = hw.read %rtl_module.getValue_data
        txn.bridge.wait_until(%ready)
        txn.return %data
    }
}
```

### Synchronization

1. **Time Alignment**:
   - TL simulation runs in logical time
   - RTL simulation runs in clock cycles
   - Bridge maintains time correspondence

2. **Event Coordination**:
   - TL events can trigger RTL actions
   - RTL signals can generate TL events
   - Proper handling of timing boundaries

## Performance Modeling

### Component Models

Each component type has associated performance characteristics:

```cpp
struct PerformanceModel {
    uint32_t latency;        // Operation latency
    uint32_t throughput;     // Operations per cycle
    PowerModel power;        // Power consumption
    ResourceUsage resources; // Area, memory, etc.
};
```

### Annotations

Performance annotations in the IR:

```mlir
txn.module @Processor attributes {
    sharp.performance = {
        clock_freq = 1000000000 : i64,  // 1 GHz
        pipeline_depth = 5 : i32
    }
} {
    txn.action_method @execute(%instr: i32) attributes {
        sharp.latency = 5 : i32,
        sharp.throughput = 1 : i32
    } { ... }
}
```

### Metrics Collection

The simulation framework collects:
- Cycle counts per module/method
- Resource utilization over time
- Communication bandwidth
- Power consumption estimates
- Critical path analysis

## Implementation Plan

### Phase 1: Transaction-Level Core (2 weeks)
1. Basic event queue implementation
2. Single-module simulation
3. Simple conflict resolution
4. Basic performance metrics

### Phase 2: Multi-Module Support (2 weeks)
1. Inter-module communication
2. Dependency tracking
3. Concurrent execution
4. Testbench infrastructure

### Phase 3: Spec Primitives (1 week)
1. Spec FIFO implementation
2. Multi-cycle operation support
3. Verification primitives
4. Assertion checking

### Phase 4: RTL Integration (2 weeks)
1. Arcilator integration
2. TL-to-RTL lowering
3. Co-simulation support
4. Timing verification

### Phase 5: Hybrid Simulation (2 weeks)
1. Bridge component design
2. Time synchronization
3. Mixed-level debugging
4. Performance analysis

### Phase 6: Advanced Features (ongoing)
1. Parallel simulation (DAM-inspired)
2. Visualization tools
3. Advanced performance models
4. Formal verification integration

## Example Usage

### Simple Counter Simulation

```python
from sharp.simulation import Simulator, SimModule

# Define simulation behavior
class CounterSim(SimModule):
    def __init__(self):
        self.count = 0
        
    def getValue(self):
        return self.count
        
    def increment(self):
        self.count += 1
        
    def decrement(self):
        self.count -= 1

# Create and run simulation
sim = Simulator()
counter = sim.add_module("Counter", CounterSim())

# Schedule events
sim.schedule(0, counter, "increment")
sim.schedule(1, counter, "increment")
sim.schedule(2, counter, "getValue", callback=print)

sim.run(max_cycles=10)
```

### Testbench Example

```mlir
// Testbench module that generates stimuli
txn.module @TestBench {
    %dut = txn.instance @DUT of @ProcessingUnit
    
    txn.rule @generateInput {
        %data = txn.spec.random_int(0, 255)
        txn.call %dut::@process(%data)
    }
    
    txn.rule @checkOutput {
        %result = txn.call %dut::@getResult()
        txn.spec.assert(%result >= 0)
    }
}
```

### Hybrid Simulation Example

```python
# Mix transaction-level testbench with RTL DUT
sim = HybridSimulator()

# Transaction-level testbench
testbench = sim.add_tl_module("TestBench", TestBenchSim())

# RTL design under test
dut = sim.add_rtl_module("DUT", "generated_dut.v")

# Connect them
sim.connect(testbench.out_port, dut.in_port)

# Run hybrid simulation
sim.run(max_cycles=1000000)
```

## Testing Strategy

1. **Unit Tests**: Each simulation component
2. **Integration Tests**: Multi-module scenarios
3. **Regression Tests**: Known designs with expected results
4. **Performance Tests**: Simulation speed benchmarks
5. **Correctness Tests**: TL vs RTL equivalence

## Future Enhancements

1. **Distributed Simulation**: Run large designs across multiple machines
2. **Hardware Acceleration**: Use FPGAs to accelerate simulation
3. **Machine Learning Integration**: Learn performance models from runs
4. **Interactive Debugging**: Step through transactions visually
5. **Formal Integration**: Connect to model checkers and theorem provers

## Conclusion

Sharp's simulation framework provides a flexible, multi-level approach to hardware design verification and performance analysis. By combining insights from EQueue's event-driven model, DAM's high-performance techniques, and Sharp's transaction semantics, we create a powerful tool for hardware designers to explore, verify, and optimize their designs efficiently.