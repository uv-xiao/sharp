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
- `read()`: Returns the current wire value
- `write(value)`: Updates the wire value combinationally

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