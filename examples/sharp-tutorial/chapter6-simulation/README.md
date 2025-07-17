# Chapter 6: Simulation Modes

## Overview

Sharp provides multiple simulation approaches for different needs:
- **Transaction-Level (TL)**: Fast behavioral simulation
- **RTL Simulation**: Cycle-accurate with Arcilator
- **JIT Compilation**: Direct execution
- **Hybrid Simulation**: Mix TL and RTL

## Simulation Modes Comparison

| Mode | Speed | Accuracy | Use Case |
|------|-------|----------|----------|
| TL   | Fast  | Behavioral | Early development |
| RTL  | Slow  | Cycle-accurate | Verification |
| JIT  | Fastest | Behavioral | Performance testing |
| Hybrid | Medium | Mixed | System integration |

## Transaction-Level Simulation

We've been using TL simulation throughout the tutorial:

### pipeline.mlir

```mlir
// Three-stage pipeline example
txn.module @Pipeline {
  // Pipeline registers
  %stage1 = txn.instance @stage1 of @Register<i32> : index
  %stage2 = txn.instance @stage2 of @Register<i32> : index
  %stage3 = txn.instance @stage3 of @Register<i32> : index
  
  // Input new data
  txn.action_method @input(%data: i32) {
    txn.call @stage1::@write(%data) : (i32) -> ()
    txn.yield
  }
  
  // Advance pipeline
  txn.action_method @advance() {
    // Read all stages
    %s1 = txn.call @stage1::@read() : () -> i32
    %s2 = txn.call @stage2::@read() : () -> i32
    %s3 = txn.call @stage3::@read() : () -> i32
    
    // Process
    %two = arith.constant 2 : i32
    %p1 = arith.addi %s1, %two : i32
    %p2 = arith.muli %s2, %two : i32
    
    // Write next stage
    txn.call @stage2::@write(%p1) : (i32) -> ()
    txn.call @stage3::@write(%p2) : (i32) -> ()
    txn.yield
  }
  
  // Get output
  txn.value_method @output() -> i32 {
    %val = txn.call @stage3::@read() : () -> i32
    txn.return %val : i32
  }
  
  // Auto-advance rule
  txn.rule @clock {
    txn.call @this.advance() : () -> ()
    txn.yield
  }
  
  txn.schedule [@input, @advance, @output, @clock] {
    conflict_matrix = {
      "input,advance" = 2 : i32,    // C
      "advance,clock" = 2 : i32,    // C
      "input,clock" = 2 : i32       // C
    }
  }
}
```

### Running TL Simulation

```bash
# Generate and run TL simulation
../../tools/generate-workspace.sh pipeline.mlir pipeline_sim
cd pipeline_sim && mkdir build && cd build
cmake .. && make
./Pipeline_sim --cycles 10 --verbose
```

## RTL Simulation with Arcilator

Arcilator provides cycle-accurate RTL simulation:

### counter_rtl.mlir

```mlir
// Simple counter for RTL simulation
txn.module @RTLCounter {
  %count = txn.instance @count of @Register<i8> : index
  
  txn.value_method @read() -> i8 {
    %val = txn.call @count::@read() : () -> i8
    txn.return %val : i8
  }
  
  txn.action_method @increment() {
    %val = txn.call @count::@read() : () -> i8
    %one = arith.constant 1 : i8
    %next = arith.addi %val, %one : i8
    txn.call @count::@write(%next) : (i8) -> ()
    txn.yield
  }
  
  txn.rule @tick {
    txn.call @this.increment() : () -> ()
    txn.yield
  }
  
  txn.schedule [@read, @increment, @tick]
}
```

### Arcilator Pipeline

```bash
# Convert to Arc dialect for RTL simulation
sharp-opt counter_rtl.mlir --sharp-arcilator > counter_arc.mlir

# Run with arcilator (if available)
# arcilator counter_arc.mlir --simulate --cycles=10
```

## JIT Compilation Mode

JIT mode compiles to machine code for maximum performance:

```bash
# Run with JIT (currently limited support)
sharp-opt pipeline.mlir --sharp-simulate="mode=jit"
```

## Hybrid Simulation

Mix transaction-level and RTL components:

### hybrid_system.mlir

```mlir
// System with both TL and RTL components
txn.module @HybridSystem {
  // TL component - behavioral FIFO
  %tl_fifo = txn.instance @tl_fifo of @FIFO<i32> : index
  
  // RTL component - synthesizable counter
  %rtl_counter = txn.instance @rtl_counter of @Register<i32> : index
  
  // Bridge between domains
  txn.action_method @transfer() {
    %empty = txn.call @tl_fifo::@isEmpty() : () -> i1
    %not_empty = arith.xori %empty, %true : i1
    
    // Transfer from TL to RTL
    %data = txn.call @tl_fifo::@dequeue() : () -> i32
    %count = txn.call @rtl_counter::@read() : () -> i32
    %sum = arith.addi %data, %count : i32
    txn.call @rtl_counter::@write(%sum) : (i32) -> ()
    
    txn.yield
  }
  
  txn.schedule [@transfer]
}
```

## Performance Monitoring

### perf_test.mlir

```mlir
// Module for performance testing
txn.module @PerfTest {
  %acc = txn.instance @acc of @Register<i64> : index
  
  // Computation-heavy method
  txn.action_method @compute(%n: i32) {
    %acc_val = txn.call @acc::@read() : () -> i64
    %n_ext = arith.extsi %n : i32 to i64
    
    // Simulate heavy computation
    %c2 = arith.constant 2 : i64
    %r1 = arith.muli %n_ext, %c2 : i64
    %r2 = arith.addi %acc_val, %r1 : i64
    
    txn.call @acc::@write(%r2) : (i64) -> ()
    txn.yield
  }
  
  txn.value_method @result() -> i64 {
    %val = txn.call @acc::@read() : () -> i64
    txn.return %val : i64
  }
  
  txn.schedule [@compute, @result]
}
```

## Simulation Scripts

### run.sh

```bash
#!/bin/bash

SHARP_ROOT="../../.."
SHARP_OPT="$SHARP_ROOT/build/bin/sharp-opt"

echo "=== Chapter 6: Simulation Modes ==="
echo ""

echo "1. Transaction-Level Simulation:"
echo "----------------------------------------"
$SHARP_ROOT/tools/generate-workspace.sh pipeline.mlir pipeline_tl
cd pipeline_tl && mkdir -p build && cd build && cmake .. > /dev/null 2>&1 && make > /dev/null 2>&1
if [ -f Pipeline_sim ]; then
    echo "✅ TL simulation built"
    echo "Running 5 cycles:"
    ./Pipeline_sim --cycles 5 --stats
else
    echo "❌ TL simulation build failed"
fi
cd ../../..
echo ""

echo "2. RTL Preparation (Arcilator):"
echo "----------------------------------------"
if $SHARP_OPT counter_rtl.mlir --sharp-arcilator > counter_arc.mlir 2>&1; then
    echo "✅ Arc conversion successful"
    echo "Generated $(wc -l < counter_arc.mlir) lines of Arc IR"
else
    echo "❌ Arc conversion failed"
fi
echo ""

echo "3. JIT Compilation Test:"
echo "----------------------------------------"
# JIT has limited support currently
echo "Testing JIT mode availability..."
if $SHARP_OPT counter_rtl.mlir --sharp-simulate="mode=jit" 2>&1 | grep -q "error"; then
    echo "⚠️  JIT mode has limited support"
else
    echo "✅ JIT mode available"
fi
echo ""

echo "4. Performance Comparison Setup:"
echo "----------------------------------------"
$SHARP_ROOT/tools/generate-workspace.sh perf_test.mlir perf_sim
if [ -d perf_sim ]; then
    echo "✅ Performance test module ready"
    echo "Build with: cd perf_sim && mkdir build && cd build && cmake .. && make"
    echo "Run with: ./PerfTest_sim --cycles 1000000 --stats"
else
    echo "❌ Performance test setup failed"
fi
```

## Debugging Simulations

### Debug Features
- **Verbose mode**: `--verbose` prints cycle-by-cycle activity
- **Statistics**: `--stats` shows performance metrics
- **Breakpoints**: Set in C++ generated code
- **Waveforms**: RTL simulation can generate VCD files

### Common Issues
1. **Conflicts**: Check conflict matrix for deadlocks
2. **Initialization**: Ensure registers have reset values
3. **Timing**: Verify multi-cycle operations
4. **State**: Check primitive state in generated C++

## Exercises

1. **Profile performance**: Compare TL vs generated C++ performance
2. **Add instrumentation**: Count method calls and conflicts
3. **Create testbench**: Build a test harness for the pipeline
4. **Debug state**: Add state dumping to simulation

## Advanced Topics

### Custom Simulation Primitives
Create simulation-only primitives:
```cpp
class CustomPrimitive : public SimModule {
  // Custom behavior for simulation
};
```

### Co-simulation
Run Sharp modules with:
- SystemVerilog testbenches
- C++ models
- Python scripts

### Distributed Simulation
For large designs:
- Partition across multiple threads
- Use DAM methodology
- Network distribution

## Key Takeaways

- Multiple simulation modes serve different purposes
- TL simulation provides fast feedback
- RTL simulation ensures cycle accuracy
- JIT offers maximum performance
- Hybrid simulation bridges abstraction levels

## Next Chapter

Chapter 7 explores the Python frontend:
- Building modules in Python
- Type system and operations
- Integration with MLIR
- Pythonic design patterns