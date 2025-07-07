# Sharp Simulation Framework

## Overview

Sharp provides a comprehensive simulation framework supporting multiple abstraction levels:
- **Transaction-level (TL)** simulation for rapid design exploration
- **RTL simulation** via CIRCT's arcilator for cycle-accurate verification
- **Concurrent simulation** using DAM methodology for high performance
- **Hybrid TL-RTL** simulation for progressive refinement
- **JIT compilation** for optimized execution

## Key Design Principles

### 1. Event-Driven Architecture
Following EQueue's approach, Sharp uses discrete-event simulation where:
- Each module maintains an event queue
- Events represent method calls or rule firings
- Events execute based on dependencies and conflicts
- Different modules can execute concurrently

### 2. One-Rule-at-a-Time (1RaaT) Semantics
Sharp implements Koika-inspired execution model:
- Rules execute atomically and sequentially within each cycle
- Three-phase execution: Scheduling → Execution → Commit
- Conflict matrix from `txn.schedule` guides scheduling decisions
- Deterministic behavior with no race conditions

### 3. Multi-Level Abstraction
The framework supports simulation at different abstraction levels:
- **Specification Level**: Using spec primitives (SpecFIFO, SpecMemory)
- **Transaction Level**: Fast functional simulation with timing annotations
- **RTL Level**: Cycle-accurate simulation via FIRRTL→Arc conversion
- **Hybrid**: Mix TL and RTL components with proper synchronization

## Quick Start

### 1. Transaction-Level Simulation

Generate and run C++ simulation from a Txn module:

```bash
# Generate C++ simulation code
sharp-opt input.mlir --sharp-simulate -o sim.cpp

# Compile and run
clang++ -std=c++17 sim.cpp -o sim
./sim

# Or use JIT mode (experimental)
sharp-opt input.mlir --sharp-simulate=mode=jit
```

### 2. Concurrent Simulation (DAM)

For multi-module designs with parallel execution:

```bash
# Generate concurrent simulation
sharp-opt input.mlir --sharp-concurrent-sim -o concurrent_sim.cpp

# Compile with threading support
clang++ -std=c++17 -pthread concurrent_sim.cpp -o concurrent_sim
./concurrent_sim
```

### 3. RTL Simulation via Arcilator

Convert to RTL and simulate with CIRCT's arcilator:

```bash
# Convert Txn to Arc dialect
sharp-opt input.mlir --sharp-arcilator -o arc.mlir

# Run with arcilator (requires CIRCT tools)
arcilator arc.mlir --trace  # Generates VCD waveforms
```

**Key Insight**: The arcilator integration leverages CIRCT's high-performance RTL simulation engine, providing orders of magnitude speedup over traditional RTL simulators while maintaining cycle accuracy.

### 4. Hybrid TL-RTL Simulation

Mix transaction-level and RTL modules:

```bash
# Generate hybrid simulation
sharp-opt input.mlir --sharp-hybrid-sim -o hybrid_sim.cpp

# Compile and run
clang++ -std=c++17 hybrid_sim.cpp -o hybrid_sim
./hybrid_sim
```

## Detailed Usage

### Writing Simulatable Modules

```mlir
// Example: Simple counter module
txn.module @Counter {
  // Create instance of Register primitive
  %count = txn.instance @count of @Register<i32> : !txn.module<"Register">
  
  // Value method - read current count
  txn.value_method @getValue() -> i32 {
    %val = txn.call @count.read() : () -> i32
    txn.return %val : i32
  }
  
  // Action method - increment counter
  txn.action_method @increment() {
    %old = txn.call @count.read() : () -> i32
    %one = arith.constant 1 : i32
    %new = arith.addi %old, %one : i32
    txn.call @count.write(%new) : (i32) -> ()
    txn.return
  }
  
  // Automatic rule
  txn.rule @autoIncrement {
    %val = txn.call @getValue() : () -> i32
    %limit = arith.constant 100 : i32
    %cond = arith.cmpi ult, %val, %limit : i32
    txn.if %cond {
      txn.call @increment() : () -> ()
    }
    txn.yield
  }
  
  // Schedule with conflict matrix
  txn.schedule [@autoIncrement, @getValue, @increment] {
    // No conflicts for this simple example
  }
}
```

### Transaction-Level Simulation Features

The `--sharp-simulate` pass generates C++ code implementing:

1. **Event-driven execution**: Methods and rules execute as discrete events
2. **Conflict checking**: Respects the conflict matrix from schedule
3. **1RaaT model**: One-rule-at-a-time execution with three phases:
   - **Scheduling Phase**: Evaluate guards, check conflicts, select rules
   - **Execution Phase**: Execute actions, track dependencies
   - **Commit Phase**: Apply state updates atomically at cycle boundary
4. **Performance metrics**: Cycle count, method call statistics, conflict resolution

**Event Model**:
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

Generated code structure:
```cpp
// Generated simulation includes:
class Counter : public SimModule {
  // State variables
  int32_t count_data;
  
  // Methods
  int32_t getValue();
  void increment();
  
  // Rules  
  void autoIncrement();
  
  // Simulation infrastructure
  void scheduleEvents();
  void executeEvents();
};
```

### Multi-Cycle Operations

Sharp supports multi-cycle operations through `txn.future` and `txn.launch` constructs:

```mlir
txn.action_method @multiCycleOp(%data: i32) attributes {multicycle = true} {
  // Per-cycle actions execute immediately
  %current = txn.call @reg::@read() : () -> i32
  txn.call @reg::@write(%data) : (i32) -> ()
  
  txn.future {
    // Static launch - executes after fixed delay
    %done1 = txn.launch after 3 {
      %val = arith.constant 100 : i32
      txn.call @reg::@write(%val) : (i32) -> ()
      txn.yield
    }
    
    // Dynamic launch - waits for condition
    %done2 = txn.launch until %done1 {
      txn.call @fifo::@enqueue(%data) : (i32) -> ()
      txn.yield
    }
    
    // Combined - condition + delay
    %done3 = txn.launch until %done2 after 1 {
      txn.call @status::@write(%true) : (i1) -> ()
      txn.yield
    }
  }
  txn.return
}
```

**Simulation Infrastructure**:
- `LaunchState`: Tracks individual launch execution (Pending → Running → Completed)
- `MultiCycleExecution`: Manages all launches for a multi-cycle action
- `MultiCycleSimModule`: Extended base class with `updateMultiCycleExecutions()`

**Key Semantics**:
- Per-cycle actions execute immediately before any launches
- Static launches (`after N`) must succeed or panic
- Dynamic launches (`until %cond`) retry until successful
- Launch dependencies create execution chains

### Concurrent Simulation (DAM Methodology)

The `--sharp-concurrent-sim` pass implements DAM (Discrete-event simulation with Adaptive Multiprocessing) principles from Zhang et al.:

1. **Independent contexts**: Each module runs in its own thread with local time
2. **Asynchronous time**: No global synchronization barrier - modules can run arbitrarily far into the future
3. **Time-bridging channels**: Handle inter-module communication with backpressure
4. **Adaptive synchronization**: Lazy pairwise sync only when modules interact

**Key Innovation**: DAM achieves near-linear speedup by eliminating the global synchronization bottleneck of traditional parallel discrete-event simulation. Each module maintains its own logical time and only synchronizes when absolutely necessary.

Example multi-module system:
```mlir
txn.module @Producer {
  txn.action_method @send(i32) { ... }
}

txn.module @Consumer {  
  txn.action_method @receive() -> i32 { ... }
}

// Generated code creates separate contexts
```

Usage:
```bash
# Set thread affinity for performance
taskset -c 0-3 ./concurrent_sim

# Enable performance statistics
./concurrent_sim --stats
```

### RTL Simulation Integration

The `--sharp-arcilator` pass converts to Arc dialect:

1. **Full pipeline**: Txn → FIRRTL → HW → Arc
2. **Preserves semantics**: Methods become module ports
3. **Cycle-accurate**: Matches hardware behavior
4. **VCD support**: Waveform generation for debugging

Example workflow:
```bash
# Step 1: Generate Arc
sharp-opt counter.mlir --sharp-arcilator -o counter_arc.mlir

# Step 2: Run arcilator
arcilator counter_arc.mlir --trace --trace-format=vcd

# Step 3: View waveforms
gtkwave counter.vcd
```

### Hybrid Simulation

Mix transaction-level and RTL modules:

```mlir
// Configure bridge
sim.bridge_config @bridge {
  sim.tl_module @TLProducer
  sim.rtl_module @RTLConsumer
  sim.method_map @send -> @receive
  sim.sync_mode "lockstep"  // or "decoupled", "adaptive"
}
```

Synchronization modes:
- **lockstep**: TL and RTL advance together each cycle
- **decoupled**: Allow bounded time divergence (e.g., 10 cycles)
- **adaptive**: Dynamically adjust based on activity

### Spec Primitives for Verification

Sharp provides spec-level primitives for golden models and verification:

```mlir
// Unbounded FIFO for specification
%fifo = txn.instance @fifo of @SpecFIFO<i32> : !txn.module<"SpecFIFO">

// Memory with configurable latency  
%mem = txn.instance @mem of @SpecMemory<i32, 1024> : !txn.module<"SpecMemory">

// Multi-cycle spec operations (planned)
txn.action_method @complexOperation(%n: i32) {
    // Initiate multi-cycle operation
    %handle = txn.spec.begin_multi_cycle()
    
    // Continue over multiple cycles
    txn.spec.continue(%handle) {
        txn.call @otherAction()
    }
    
    // Complete operation
    txn.spec.complete(%handle)
    txn.return
}
```

These primitives:
- Are non-synthesizable (caught by pre-synthesis check)
- Support unbounded capacity and non-deterministic timing
- Enable modeling of complex protocols and algorithms
- Bridge specification and implementation for verification

### Performance Analysis

All simulation modes support performance metrics:

```cpp
// In generated main():
simulator.enableStats(true);
simulator.run(1000);  // Run 1000 cycles
simulator.printStats();
```

Output includes:
- Total cycles executed
- Method call counts
- Rule firing statistics
- Conflict resolution data
- For concurrent: speedup metrics

### Advanced Features

#### Custom Scheduling

Override default scheduling:
```cpp
simulator.setSchedulingPolicy(SchedulingPolicy::Priority);
simulator.setPriority("incrementRule", 10);
```

#### Breakpoints and Debugging

```cpp
simulator.setBreakpoint("Counter.increment", []() {
    std::cout << "Increment called\n";
    return true;  // Continue execution
});
```

#### State Inspection

```cpp
auto* counter = simulator.getModule<Counter>("counter");
std::cout << "Current count: " << counter->count_data << "\n";
```

## Command Reference

### sharp-simulate

```bash
sharp-opt input.mlir --sharp-simulate[=options]
```

Options:
- `mode=translate` (default): Generate C++ code
- `mode=jit`: JIT compile and execute (experimental)
- `output=<file>`: Output filename (default: stdout)

### sharp-concurrent-sim

```bash
sharp-opt input.mlir --sharp-concurrent-sim[=options]
```

Options:
- `channels=bounded`: Use bounded channels (default: unbounded)
- `channel-size=N`: Set channel capacity
- `stats=true`: Enable performance statistics

### sharp-arcilator

```bash
sharp-opt input.mlir --sharp-arcilator
```

No options - outputs Arc dialect to stdout.

### sharp-hybrid-sim

```bash
sharp-opt input.mlir --sharp-hybrid-sim[=options]
```

Options:
- `sync=lockstep|decoupled|adaptive`: Synchronization mode
- `divergence=N`: Max time divergence for decoupled mode

## Examples

Complete examples in `test/Simulation/`:
- `counter.mlir`: Basic counter with auto-increment
- `pipeline.mlir`: Multi-stage pipeline
- `concurrent-system.mlir`: Producer-consumer with channels
- `hybrid-example.mlir`: Mixed TL-RTL design

## Simulation Methodology

### Choosing the Right Simulation Level

1. **Use Spec-Level Simulation When**:
   - Exploring algorithms without implementation details
   - Creating golden models for verification
   - Modeling complex protocols with unbounded resources
   
2. **Use Transaction-Level Simulation When**:
   - Rapid design space exploration is needed
   - Functional correctness is the primary concern
   - Performance estimation (not cycle accuracy) is sufficient
   
3. **Use RTL Simulation When**:
   - Cycle-accurate behavior must be verified
   - Preparing for synthesis and implementation
   - Debugging timing-related issues

4. **Use Hybrid Simulation When**:
   - Some components are still at spec/TL level
   - Progressive refinement from TL to RTL
   - Need to verify TL models against RTL implementation

### Performance Optimization Strategies

1. **Event Queue Optimization**:
   - Use priority queues for efficient event scheduling
   - Batch non-conflicting events for parallel execution
   - Cache conflict checking results

2. **Concurrent Simulation Tuning**:
   - Map modules to cores based on communication patterns
   - Use bounded channels to prevent runaway modules
   - Monitor time divergence between contexts

3. **Memory Management**:
   - Pool event objects to reduce allocation overhead
   - Use copy-on-write for large state updates
   - Implement incremental state checkpointing

### Verification Methodology

1. **Reference Model Checking**:
   ```cpp
   // Compare TL simulation against spec model
   auto spec_result = spec_model.execute(input);
   auto tl_result = tl_model.execute(input);
   assert(spec_result == tl_result);
   ```

2. **Assertion-Based Verification**:
   - Embed assertions in transaction methods
   - Check invariants at cycle boundaries
   - Monitor protocol compliance

3. **Coverage Analysis**:
   - Track which rules fire and how often
   - Monitor conflict resolution patterns
   - Identify unreachable states

## Limitations

1. **JIT mode**: Control flow (txn.if) not fully supported
2. **Hybrid simulation**: Full arcilator C API integration pending
3. **Spec primitives**: SpecFIFO and SpecMemory implementations pending
4. **State operations**: txn.state not yet implemented

## Performance Tips

1. **Use concurrent simulation** for multi-module designs
2. **Set thread affinity** for concurrent simulation
3. **Choose appropriate sync mode** for hybrid simulation
4. **Profile with stats enabled** to identify bottlenecks
5. **Consider JIT** for long-running simulations (when stable)