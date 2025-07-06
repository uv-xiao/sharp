# Sharp Txn Primitives Documentation

## Overview

Sharp Txn primitives are the fundamental building blocks for hardware description in the Sharp framework. They provide a transaction-level interface that bridges high-level behavioral descriptions with low-level hardware implementations in FIRRTL.

## Architecture

### Separation of Concerns

The Sharp Txn primitive system maintains a clear separation between:

1. **Transaction-level interface** (txn primitive) - Declares the behavioral interface
2. **Hardware implementation** (FIRRTL module) - Contains the actual RTL implementation

This separation allows:
- Clean abstraction boundaries
- Reusable hardware implementations
- Support for both synthesizable (`hw`) and specification (`spec`) primitives
- Easier verification and testing

### Primitive Structure

A synthesizable (`hw`) primitive contains:

```mlir
txn.primitive @PrimitiveName type = "hw" interface = !txn.module<"PrimitiveName"> {
  // Method declarations
  txn.fir_value_method @read() {firrtl.port = "read_data"} : () -> i32
  txn.fir_action_method @write() {firrtl.data_port = "write_data", 
                                 firrtl.enable_port = "write_enable"} : (i32) -> ()
  
  // Clocking
  txn.clock_by @clk
  txn.reset_by @rst
  
  // Scheduling with conflict matrix
  txn.schedule [@read, @write] {conflict_matrix = {...}}
} {firrtl.impl = "PrimitiveName_impl"}
```

## Bridging Txn and FIRRTL

The connection between txn methods and FIRRTL ports is specified through attributes:

### Value Methods (Read-only)
- `firrtl.port`: Specifies which FIRRTL output port provides the data
- Example: `txn.fir_value_method @read() {firrtl.port = "read_data"} : () -> i32`

### Action Methods (Can modify state)
- `firrtl.data_port`: Specifies the FIRRTL input port for data
- `firrtl.enable_port`: Specifies the FIRRTL input port for enable signal
- Example: `txn.fir_action_method @write() {firrtl.data_port = "write_data", firrtl.enable_port = "write_enable"} : (i32) -> ()`

### Primitive Reference
- The primitive has a `firrtl.impl` attribute that references the corresponding FIRRTL module name

## Built-in Primitives

### Register

A stateful element that holds values across clock cycles.

**Txn Interface:**
- `read()`: Returns the current register value (conflict-free with all operations)
- `write(value)`: Updates the register value on the next clock edge (conflicts with other writes)

**Conflict Matrix:**
- read ↔ read: CF (Conflict-Free)
- read ↔ write: CF
- write ↔ read: CF
- write ↔ write: C (Conflict)

**FIRRTL Implementation:**
```mlir
firrtl.module @Register_impl(in %clock: !firrtl.clock, 
                            in %reset: !firrtl.uint<1>,
                            out %read_data: !firrtl.uint<32>,
                            in %read_enable: !firrtl.uint<1>,
                            in %write_data: !firrtl.uint<32>,
                            in %write_enable: !firrtl.uint<1>) {
  %c0_ui32 = firrtl.constant 0 : !firrtl.uint<32>
  %reg = firrtl.regreset %clock, %reset, %c0_ui32 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<32>, !firrtl.uint<32>
  %0 = firrtl.not %reset : (!firrtl.uint<1>) -> !firrtl.uint<1>
  %1 = firrtl.and %write_enable, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.when %1 : !firrtl.uint<1> {
    firrtl.connect %reg, %write_data : !firrtl.uint<32>
  }
  firrtl.connect %read_data, %reg : !firrtl.uint<32>
}
```

### Wire

A combinational connection between components.

**Txn Interface:**
- `read()`: Returns the current wire value (action method - NOT a value method due to SA conflict with write)
- `write(value)`: Updates the wire value combinationally (action method)

**Conflict Matrix:**
- read ↔ read: CF (Conflict-Free)
- read ↔ write: SB (Sequence Before - read must happen before write)
- write ↔ read: SA (Sequence After - write must happen after read)
- write ↔ write: C (Conflict)

**FIRRTL Implementation:**
```mlir
firrtl.module @Wire_impl(in %clock: !firrtl.clock,
                        in %reset: !firrtl.uint<1>,
                        out %read_data: !firrtl.uint<32>,
                        in %write_data: !firrtl.uint<32>,
                        in %write_enable: !firrtl.uint<1>) {
  %wire = firrtl.wire : !firrtl.uint<32>
  firrtl.when %write_enable : !firrtl.uint<1> {
    firrtl.connect %wire, %write_data : !firrtl.uint<32>
  }
  firrtl.connect %read_data, %wire : !firrtl.uint<32>
}
```

## Creating Primitives Programmatically

The Sharp framework provides C++ APIs to create primitives:

```cpp
// Create a Register primitive
auto regPrimitive = createRegisterPrimitive(builder, loc, "MyReg", i32Type);

// Create the corresponding FIRRTL module (must be within a firrtl.circuit)
auto regFIRRTL = createRegisterFIRRTLModule(builder, loc, "MyReg", i32Type);
```

### API Reference

```cpp
// In sharp/Dialect/Txn/TxnPrimitives.h

// Create txn primitives
PrimitiveOp createRegisterPrimitive(OpBuilder &builder, Location loc,
                                   StringRef name, Type dataType);
PrimitiveOp createWirePrimitive(OpBuilder &builder, Location loc,
                               StringRef name, Type dataType);

// Create FIRRTL implementations
circt::firrtl::FModuleOp createRegisterFIRRTLModule(OpBuilder &builder, Location loc,
                                                    StringRef name, Type dataType);
circt::firrtl::FModuleOp createWireFIRRTLModule(OpBuilder &builder, Location loc,
                                                StringRef name, Type dataType);
```

## Translation Process

During txn-to-FIRRTL translation:

1. The translator identifies synthesizable primitives (`type = "hw"`)
2. For each primitive instance, it looks up the `firrtl.impl` attribute
3. Method calls are translated to port connections:
   - Value method calls become connections from FIRRTL output ports
   - Action method calls become connections to FIRRTL input ports with enable signals
4. The conflict matrix guides the generation of scheduling logic

## Usage Examples

### Using Primitives in Txn Modules

```mlir
// Example: Counter using Register primitive
txn.module @Counter {
  // Create instance of Register primitive
  %count = txn.instance @count of @Register<i32> : !txn.module<"Register">
  
  txn.value_method @getValue() -> i32 {
    %val = txn.call @count.read() : () -> i32
    txn.return %val : i32
  }
  
  txn.action_method @increment() {
    %old = txn.call @count.read() : () -> i32
    %one = arith.constant 1 : i32
    %new = arith.addi %old, %one : i32
    txn.call @count.write(%new) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@getValue, @increment] {
    conflict_matrix = {}
  }
}
```

### Parametric Primitive Usage

```mlir
// Example: Using parametric primitives with different types
txn.module @DataPath {
  // 8-bit control register
  %ctrl = txn.instance @ctrl of @Register<i8> : !txn.module<"Register">
  
  // 32-bit data register  
  %data = txn.instance @data of @Register<i32> : !txn.module<"Register">
  
  // 64-bit accumulator
  %acc = txn.instance @acc of @Register<i64> : !txn.module<"Register">
  
  // Wire for combinational logic
  %temp = txn.instance @temp of @Wire<i32> : !txn.module<"Wire">
  
  txn.action_method @process(%input: i32) {
    // Read control register
    %ctrl_val = txn.call @ctrl.read() : () -> i8
    
    // Combinational logic through wire
    txn.call @temp.write(%input) : (i32) -> ()
    %temp_val = txn.call @temp.read() : () -> i32
    
    // Update data register based on control
    %zero = arith.constant 0 : i8
    %enable = arith.cmpi ne, %ctrl_val, %zero : i8
    txn.if %enable {
      txn.call @data.write(%temp_val) : (i32) -> ()
    }
    
    txn.return
  }
  
  txn.schedule [@process] {
    conflict_matrix = {}
  }
}
```

### Custom Primitive Declaration

```mlir
// Example: Declaring a custom FIFO primitive
txn.primitive @FIFO type = "hw" interface = !txn.module<"FIFO"> {
  // Value methods
  txn.fir_value_method @isEmpty() 
    {firrtl.port = "empty"} : () -> i1
  
  txn.fir_value_method @isFull() 
    {firrtl.port = "full"} : () -> i1
  
  // Action methods
  txn.fir_action_method @enqueue(%data: i32) 
    {firrtl.data_port = "enq_data", 
     firrtl.enable_port = "enq_valid"} : (i32) -> ()
  
  txn.fir_action_method @dequeue() 
    {firrtl.data_port = "deq_data",
     firrtl.enable_port = "deq_ready"} : () -> i32
  
  // Clocking
  txn.clock_by @clk
  txn.reset_by @rst
  
  // Conflict matrix
  txn.schedule [@isEmpty, @isFull, @enqueue, @dequeue] {
    conflict_matrix = #txn.conflict_dict<{
      "enqueue_dequeue" = #txn.C,  // Cannot enqueue and dequeue simultaneously
      "isEmpty_isFull" = #txn.CF,  // Status checks are conflict-free
    }>
  }
} {firrtl.impl = "FIFO_impl"}
```

### Converting to FIRRTL

```bash
# Convert a module using primitives to FIRRTL
sharp-opt --convert-txn-to-firrtl counter.mlir -o counter_firrtl.mlir

# The conversion will:
# 1. Generate FIRRTL modules for each primitive type used
# 2. Instantiate primitives as FIRRTL instances
# 3. Convert method calls to port connections
# 4. Add scheduling logic based on conflict matrices
```

### FIFO

A first-in-first-out queue with bounded capacity.

**Txn Interface:**
- `enqueue(value)`: Add element to FIFO
- `dequeue()`: Remove and return element from FIFO
- `isEmpty()`: Check if FIFO is empty
- `isFull()`: Check if FIFO is full

**Conflict Matrix:**
- enqueue ↔ isFull: SB (check before enqueue)
- dequeue ↔ isEmpty: SB (check before dequeue)
- enqueue ↔ dequeue: C (cannot do both simultaneously)

**Software Semantics:**
```cpp
std::queue<int64_t> fifo_data;
const size_t depth = 16;
```

## Specification Primitives

### Memory

A memory with address-based access for verification and specification.

**Txn Interface:**
- `read(addr)`: Read data from address
- `write(addr, data)`: Write data to address
- `clear()`: Reset all memory to zero

**Conflict Matrix:**
- read ↔ read: CF (parallel reads allowed)
- read ↔ write: C (conflict on same address)
- write ↔ write: C (conflict on same address)
- clear ↔ all: C (exclusive operation)

**Software Semantics:**
```cpp
std::unordered_map<int32_t, int64_t> memory_data;
static constexpr size_t MEMORY_SIZE = 1024;
```

### SpecFIFO

An unbounded FIFO for specification and verification.

**Txn Interface:**
- `enqueue(value)`: Add element (always succeeds)
- `dequeue()`: Remove and return element
- `isEmpty()`: Check if empty
- `size()`: Get current number of elements
- `peek()`: Look at front element without removing

**Conflict Matrix:**
- enqueue ↔ enqueue: SB (preserve order)
- dequeue ↔ dequeue: SB (preserve order)
- enqueue ↔ dequeue: SB (enqueue before dequeue)
- Status checks (isEmpty, size, peek): CF with each other

**Software Semantics:**
```cpp
std::queue<int64_t> fifo_data; // Unbounded
```

### SpecMemory

A memory with configurable read latency for modeling real memory systems.

**Txn Interface:**
- `read(addr)`: Read with configured latency
- `write(addr, data)`: Write data
- `setLatency(cycles)`: Configure read latency
- `getLatency()`: Get current latency
- `clear()`: Reset memory

**Conflict Matrix:**
- Similar to Memory
- setLatency ↔ read: C (changing latency affects reads)
- getLatency ↔ setLatency: SB (read before write)

**Timing:**
- read: `dynamic` (depends on configured latency)
- Other methods: combinational or single-cycle

**Software Semantics:**
```cpp
std::unordered_map<int32_t, int64_t> memory_data;
int32_t read_latency = 1;
```

## Future Extensions

The primitive system is designed to be extensible:

- Additional primitives can be added by implementing the constructor functions
- Custom primitives can define their own conflict matrices
- Non-synthesizable (`spec`) primitives can provide behavioral specifications
- Multi-cycle operations can be supported through timing attributes

## Implementation Notes

- All primitives must have a `schedule` operation as their last operation
- FIRRTL modules must be created within a `firrtl.circuit` context
- Type conversion from MLIR types to FIRRTL types is handled automatically
- Clock and reset signals are required even for combinational primitives (FIRRTL requirement)
- Spec primitives are marked with `spec` attribute and include `software_semantics`