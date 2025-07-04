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

Sharp follows a **one-rule-at-a-time (1RaaT)** execution model inspired by Koika, where rules execute atomically and sequentially within each clock cycle. See `docs/execution_model.md` for complete details.

1. **Cycle Execution Phases**:
   - **Scheduling Phase**: Evaluate guards, resolve conflicts, determine execution order
   - **Execution Phase**: Execute scheduled rules atomically in order
   - **Commit Phase**: Apply all state updates simultaneously

2. **Event-Based Implementation**:
   ```
   void executeCycle() {
       // Phase 1: Schedule - determine which rules can execute
       auto enabled_rules = evaluateGuards();
       auto schedule = resolveConflicts(enabled_rules, conflict_matrix);
       
       // Phase 2: Execute - run rules atomically
       for (auto& rule : schedule) {
           Event e = createRuleEvent(rule);
           e.execute();  // Atomic execution
           
           // Handle multi-cycle operations
           if (e.has_continuation()) {
               schedule_continuation(e, e.next_cycle());
           }
       }
       
       // Phase 3: Commit - update state atomically
       commitStateUpdates();
       advanceClock();
   }
   ```

3. **Multi-Cycle Operations**:
   - Static latency: `txn.launch {latency=N} { ... }`
   - Dynamic latency: `txn.launch until %cond { ... }`
   - Continuations scheduled for future cycles
   - State changes remain atomic at cycle boundaries

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

// Multi-cycle operations with static latency
txn.action_method @readMemory(%addr: i32) -> i32 {
    txn.launch {latency=3} {
        // This block executes 3 cycles later
        %data = txn.primitive.call @Memory::read(%addr) : (i32) -> i32
        txn.return %data : i32
    }
}

// Multi-cycle operations with dynamic latency  
txn.action_method @waitForReady(%data: i32) {
    %done = txn.launch until %ready {
        // This block repeats until %ready is true
        %ready = txn.call @isDeviceReady() : () -> i1
        txn.if %ready {
            txn.call @sendData(%data) : (i32) -> ()
        }
        txn.return %ready : i1
    }
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
   - TL simulation runs in logical time (cycle-based)
   - RTL simulation runs in clock cycles (event-based)
   - Bridge maintains time correspondence

2. **Event Coordination**:
   - TL events can trigger RTL actions
   - RTL signals can generate TL events
   - Proper handling of timing boundaries

3. **Execution Interface**:
   ```cpp
   class HybridSimulationInterface {
       // Synchronize TL and RTL execution
       void synchronize() {
           // Wait for both simulators to reach sync point
           tl_sim.runUntilBarrier();
           rtl_sim.runUntilBarrier();
           
           // Exchange interface signals
           exchangeMethodCalls();
           exchangeStateUpdates();
           
           // Advance both simulators
           tl_sim.advanceCycle();
           rtl_sim.advanceClock();
       }
       
       // Method call translation
       void bridgeMethodCall(MethodCall& call) {
           if (call.from_tl) {
               // TL → RTL: Set RTL input signals
               rtl_sim.setSignal(call.method + "_en", true);
               rtl_sim.setSignal(call.method + "_data", call.args);
               // Wait for RTL ready signal
               while (!rtl_sim.getSignal(call.method + "_ready"))
                   rtl_sim.step();
           } else {
               // RTL → TL: Create TL event
               Event e = Event::methodCall(call.module, call.method, call.args);
               tl_sim.scheduleEvent(e);
           }
       }
   };
   ```

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

### Phase 1: Transaction-Level Core
1. Basic event queue implementation
2. Single-module simulation
3. Simple conflict resolution
4. Basic performance metrics

### Phase 2: Multi-Module Support
1. Inter-module communication
2. Dependency tracking
3. Concurrent execution
4. Testbench infrastructure

### Phase 3: Spec Primitives
1. Spec FIFO implementation
2. Multi-cycle operation support
3. Verification primitives
4. Assertion checking

### Phase 4: RTL Integration
1. Arcilator integration
2. TL-to-RTL lowering
3. Co-simulation support
4. Timing verification

### Phase 5: Hybrid Simulation
1. Bridge component design
2. Time synchronization
3. Mixed-level debugging
4. Performance analysis

### Phase 6: Advanced Features
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