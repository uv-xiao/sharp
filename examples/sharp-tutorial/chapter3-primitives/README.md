# Chapter 3: Hardware Primitives

## Overview

This chapter introduces Sharp's hardware primitives - the building blocks for creating stateful hardware. We'll explore:
- Register and Wire primitives
- FIFO (First-In-First-Out) queues
- Memory arrays
- Software semantics for simulation

## Built-in Primitives

### Register
- **Purpose**: State-holding element synchronized to clock
- **Methods**: 
  - `read()` - Get current value
  - `write(value)` - Set value for next cycle
- **Semantics**: Updates on clock edge

### Wire
- **Purpose**: Combinational connection
- **Methods**:
  - `read()` - Get current value
  - `write(value)` - Set value immediately
- **Semantics**: Updates instantly (combinational)

### FIFO
- **Purpose**: Queue with bounded capacity
- **Methods**:
  - `enqueue(value)` - Add to back of queue
  - `dequeue()` - Remove from front
  - `isEmpty()` - Check if empty
  - `isFull()` - Check if at capacity
- **Semantics**: Preserves ordering, blocks when full

## Example: Producer-Consumer with FIFO

Let's build a system that demonstrates FIFO usage:

### producer_consumer.mlir

```mlir
// Producer-Consumer pattern using FIFO
txn.module @ProducerConsumer {
  // FIFO buffer between producer and consumer
  %buffer = txn.instance @buffer of @FIFO<i32> : !txn.module<"FIFO">
  
  // State for producer
  %prod_count = txn.instance @prod_count of @Register<i32> : !txn.module<"Register">
  
  // State for consumer  
  %cons_sum = txn.instance @cons_sum of @Register<i32> : !txn.module<"Register">
  
  // Producer action: generate sequential values
  txn.action_method @produce() {
    // Check if FIFO has space
    %full = txn.call @buffer::@isFull() : () -> i1
    %not_full = arith.xori %full, %true : i1
    
    // Only produce if not full (simplified - real hardware would use guards)
    %count = txn.call @prod_count::@read() : () -> i32
    txn.call @buffer::@enqueue(%count) : (i32) -> ()
    
    // Increment counter
    %one = arith.constant 1 : i32
    %next = arith.addi %count, %one : i32
    txn.call @prod_count::@write(%next) : (i32) -> ()
    
    txn.yield
  }
  
  // Consumer action: sum received values
  txn.action_method @consume() {
    // Check if FIFO has data
    %empty = txn.call @buffer::@isEmpty() : () -> i1
    %not_empty = arith.xori %empty, %true : i1
    
    // Only consume if not empty
    %value = txn.call @buffer::@dequeue() : () -> i32
    
    // Add to running sum
    %sum = txn.call @cons_sum::@read() : () -> i32
    %new_sum = arith.addi %sum, %value : i32
    txn.call @cons_sum::@write(%new_sum) : (i32) -> ()
    
    txn.yield
  }
  
  // Value methods for monitoring
  txn.value_method @getProducerCount() -> i32 {
    %val = txn.call @prod_count::@read() : () -> i32
    txn.return %val : i32
  }
  
  txn.value_method @getConsumerSum() -> i32 {
    %val = txn.call @cons_sum::@read() : () -> i32
    txn.return %val : i32
  }
  
  txn.value_method @getBufferEmpty() -> i1 {
    %val = txn.call @buffer::@isEmpty() : () -> i1
    txn.return %val : i1
  }
  
  // Rules for autonomous operation
  txn.rule @autoProducer {
    // Producer runs when buffer not full
    %full = txn.call @buffer::@isFull() : () -> i1
    %not_full = arith.xori %full, %true : i1
    // In real implementation, would use guards
    txn.call @this.produce() : () -> ()
    txn.yield
  }
  
  txn.rule @autoConsumer {
    // Consumer runs when buffer not empty
    %empty = txn.call @buffer::@isEmpty() : () -> i1
    %not_empty = arith.xori %empty, %true : i1
    // In real implementation, would use guards
    txn.call @this.consume() : () -> ()
    txn.yield
  }
  
  // Schedule all methods
  txn.schedule [@produce, @consume, @getProducerCount, @getConsumerSum, 
                @getBufferEmpty, @autoProducer, @autoConsumer] {
    // Key conflicts:
    // - produce conflicts with consume (both modify FIFO)
    // - Rules conflict with their respective actions
    conflict_matrix = {
      "produce,consume" = 2 : i32,      // C
      "consume,produce" = 2 : i32,      // C
      "autoProducer,produce" = 2 : i32, // C
      "autoConsumer,consume" = 2 : i32  // C
    }
  }
}
```

## Software Semantics

When simulating, each primitive has software behavior:

### Register Simulation
```cpp
int32_t register_data = 0;  // State variable

// read() method
int64_t value = register_data;

// write() method  
register_data = new_value;  // Updates for next cycle
```

### FIFO Simulation
```cpp
std::queue<int32_t> fifo_queue;
constexpr size_t fifo_depth = 16;

// enqueue() method
if (fifo_queue.size() < fifo_depth) {
    fifo_queue.push(value);
}

// dequeue() method
int64_t value = 0;
if (!fifo_queue.empty()) {
    value = fifo_queue.front();
    fifo_queue.pop();
}

// isEmpty() method
bool empty = fifo_queue.empty();

// isFull() method
bool full = (fifo_queue.size() >= fifo_depth);
```

## Building and Testing

### 1. Parse and check
```bash
sharp-opt producer_consumer.mlir
```

### 2. Generate simulation
```bash
../../tools/generate-workspace.sh producer_consumer.mlir prodcons_sim
cd prodcons_sim && mkdir build && cd build
cmake .. && make
./ProducerConsumer_sim --cycles 20 --verbose
```

## Additional Examples

### Memory Primitive

The Memory primitive provides address-based storage:

```mlir
// See memory_example.mlir
%mem = txn.instance @mem of @Memory<i64> : !txn.module<"Memory">

// Read from address
%data = txn.call @mem::@read(%addr) : (i32) -> i64

// Write to address
txn.call @mem::@write(%addr, %data) : (i32, i64) -> ()

// Clear all memory
txn.call @mem::@clear() : () -> ()
```

### SpecFIFO Primitive

SpecFIFO is an unbounded FIFO for verification:

```mlir
// See specfifo_example.mlir
%fifo = txn.instance @fifo of @SpecFIFO<i32> : !txn.module<"SpecFIFO">

// Always succeeds
txn.call @fifo::@enqueue(%data) : (i32) -> ()

// Check size
%size = txn.call @fifo::@size() : () -> i32

// Peek without removing
%front = txn.call @fifo::@peek() : () -> i32
```

## Running the Examples

```bash
# Test all primitive examples
./run.sh

# Test individual examples
sharp-opt producer_consumer.mlir
sharp-opt memory_example.mlir
sharp-opt specfifo_example.mlir
```

## Exercises

1. **Add FIFO size monitoring**: Create a value method that returns current FIFO occupancy
2. **Implement backpressure**: Make producer stop when FIFO is full (using proper guards)
3. **Build a cache controller**: Use Memory primitives for tag and data storage
4. **Create a packet router**: Use SpecFIFO for modeling network queues
5. **Build a ring buffer**: Use multiple registers to create a circular buffer

## Advanced Topics

### Parametric Primitives
Primitives can be parameterized by type:
```mlir
%fifo = txn.instance @data_fifo of @FIFO<i64> : !txn.module<"FIFO">
%reg = txn.instance @wide_reg of @Register<i128> : !txn.module<"Register">
```

### Custom Primitives
You can define your own primitives:
```mlir
txn.primitive @CustomPrimitive<T> type = "hw" {
  txn.fir_value_method @customRead() : () -> T
  txn.fir_action_method @customWrite() : (T) -> ()
  // Define interface and conflicts
}
```

## Key Takeaways

- Primitives provide the stateful building blocks for hardware
- Each primitive has both hardware semantics and software simulation
- FIFOs enable decoupled producer-consumer patterns
- Proper conflict management ensures correct concurrent behavior
- Software semantics enable fast transaction-level simulation

## Next Chapter

Chapter 4 will explore Sharp's analysis passes:
- Conflict matrix inference
- Combinational loop detection
- Pre-synthesis checking
- Performance analysis