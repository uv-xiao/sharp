# Sharp Txn Primitives

This directory contains the synthesizable primitive implementations for the Sharp Txn dialect.

## Structure

- C++ implementations of primitives (Register.cpp, Wire.cpp, etc.)
  - each primitive should contain a constructor function to build the `PrimitiveOp` for the primitive;

## IR

- Every primitive should be a `PrimitiveOp` in the Sharp Txn dialect.
- The `PrimitiveOp` should have the following attributes:
  - `sym_name`: the name of the primitive
  - `type`: the type of the primitive, either "hw" or "spec"
- For `hw` primitives,
  - `PrimitiveOp`'s `body` region should contain the FIRRTL implementation of the primitive, a `firrtl.module` operation. (There should be a method to get the FIRRTL module from the `PrimitiveOp`.)
  - refer to lib/Dialect/Txn/primitives/bsv-embedding-rtl.pdf (only consider Default clock, Default reset, Schedule, and Interface) to see what the method operations should declare.
    - `fir_value_method`/`fir_action_method` operations define the interface methods of the primitive.
    - `schedule` with conflict matrix for Schedule.
    - `clock_by` for Default clock.
    - `reset_by` for Default reset.
- For `spec` primitives,
  - no FIRRTL module is needed.
  - `PrimitiveOp`'s `body` region should contain `rule`/`value_method`/`action_method` operations, each of which contains software description for the specification.
  - `schedule` with conflict matrix for Schedule.


## Primitives

### Register
- Stateful element that holds values across clock cycles
- Supports reset functionality
- Two methods: `read`(value) and `write`(action)

### Wire  
- Combinational connection between components
- Direct passthrough of signals
- Two methods: `read`(value) and `write`(action)
- `read` SB (sequence before) `write`

## Usage

These primitives are used during the txn-to-FIRRTL translation pass. The C++ implementations provide the interface, while the FIRRTL modules provide the hardware implementation that gets included in the generated FIRRTL output.