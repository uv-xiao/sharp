# PySharp: Pythonic Frontend for Sharp

PySharp provides a high-level Pythonic API for constructing hardware modules using Sharp's transaction-based semantics. Following the design pattern of CIRCT's PyCDE, PySharp imports from Sharp's namespace and provides an embedded DSL for hardware description.

## Overview

PySharp enables:
- **Type-safe hardware types** (i8, i16, i32, i64, etc.)
- **Signal-based computation** with operator overloading
- **Module construction** using builder pattern or decorators
- **Conflict matrix management** for transaction scheduling
- **Graceful fallback** when native bindings are unavailable

## Installation

PySharp is included with Sharp's Python bindings:

```bash
# Build Sharp with Python bindings
pixi run build

# Import from the build output
import sys
sys.path.insert(0, "/path/to/sharp/build/python_packages")

from sharp import pysharp
```

For development without native bindings:
```bash
# Import directly from source
sys.path.insert(0, "/path/to/sharp/lib/Bindings/Python")
import pysharp
```

## Basic Usage

### Types

PySharp provides hardware types for both standard integers and FIRRTL types:

```python
from sharp import pysharp

# Predefined integer types
print(pysharp.i8)    # i8
print(pysharp.i32)   # i32
print(pysharp.i64)   # i64

# FIRRTL types
print(pysharp.uint(16))  # uint<16>
print(pysharp.sint(32))  # sint<32>

# Custom integer types
custom = pysharp.IntType(24)  # i24
```

### Signals and Operations

Signals represent values in the hardware design with support for arithmetic operations:

```python
# Create signals
a = pysharp.Signal("a", pysharp.i32)
b = pysharp.Signal("b", pysharp.i32)

# Arithmetic operations
c = a + b  # Signal((a + b): i32)
d = a - 10 # Signal((a - 10): i32)

# Bitwise operations  
e = a & b  # Signal((a & b): i32)
f = a | 0xFF  # Signal((a | 255): i32)

# Constants
const = pysharp.constant(42, pysharp.i32)  # Signal(const_42: i32)
```

### Module Builder

The `ModuleBuilder` provides programmatic module construction:

```python
# Create a module builder
builder = pysharp.ModuleBuilder("Counter")

# Add state variables
count_state = builder.add_state("count", pysharp.i32)
valid_state = builder.add_state("valid", pysharp.i1)

# Add value method (combinational, no side effects)
get_value = builder.add_value_method("getValue")

# Add action methods (can modify state)
increment = builder.add_action_method("increment")
decrement = builder.add_action_method("decrement")

# Add rules (fire automatically)
auto_inc = builder.add_rule("autoIncrement")

# Set conflict relationships
builder.set_conflict("increment", "decrement", pysharp.ConflictRelation.C)
builder.set_conflict("autoIncrement", "increment", pysharp.ConflictRelation.SB)

print(builder)
# Output:
# ModuleBuilder(Counter)
#   States: ['count', 'valid']
#   Methods: ['getValue', 'increment', 'decrement']
#   Rules: ['autoIncrement']
```

### Conflict Relations

PySharp supports Sharp's conflict relation model:

```python
# Conflict relations between actions
pysharp.ConflictRelation.SB  # 0 - Sequenced Before
pysharp.ConflictRelation.SA  # 1 - Sequenced After
pysharp.ConflictRelation.C   # 2 - Conflict
pysharp.ConflictRelation.CF  # 3 - Conflict-Free
```

### Module Class and Decorators

For more Pythonic module definition:

```python
@pysharp.module("FIFO")
class FIFO:
    # State variables
    data = pysharp.State("data", pysharp.i32)
    valid = pysharp.State("valid", pysharp.i1)
    
    @pysharp.value_method
    def canEnqueue(self) -> pysharp.i1:
        # Check if we can enqueue
        return ~self.valid.read()
    
    @pysharp.value_method
    def canDequeue(self) -> pysharp.i1:
        # Check if we can dequeue
        return self.valid.read()
    
    @pysharp.action_method
    def enqueue(self, value: pysharp.i32):
        # Store data and mark as valid
        self.data.write(value)
        self.valid.write(pysharp.constant(True, pysharp.i1))
    
    @pysharp.action_method
    def dequeue(self) -> pysharp.i32:
        # Mark as invalid and return data
        self.valid.write(pysharp.constant(False, pysharp.i1))
        return self.data.read()
    
    @pysharp.rule
    def resetOnOverflow(self):
        # Automatic reset rule
        pass

# Create instance
fifo = pysharp.Module("FIFO")
fifo.set_conflict("enqueue", "dequeue", pysharp.ConflictRelation.C)
```

## Advanced Features

### State Operations

```python
# State variables support read/write operations
state = pysharp.State("counter", pysharp.i32)

# Read returns a Signal
value = state.read()  # Signal(read_counter: i32)

# Write returns a tuple (for now - will be integrated with MLIR builders)
write_op = state.write(value)  # ('write', State(counter: i32), Signal(...))
```

### Complex Expressions

```python
# Signals support chained operations
a = pysharp.Signal("a", pysharp.i16)
b = pysharp.Signal("b", pysharp.i16)
c = pysharp.Signal("c", pysharp.i16)

# Chained arithmetic
result = a + b + c  # Signal(((a + b) + c): i16)

# Mixed operations
result2 = (a + 100) & b  # Signal(((a + 100) & b): i16)

# Type preservation - operations preserve left operand type
sig8 = pysharp.Signal("x", pysharp.i8)
result3 = sig8 + 1  # Result has type i8
```

## Architecture

PySharp follows PyCDE's pattern of importing from its own namespace:

```python
# PySharp attempts to import from various locations
try:
    from . import _mlir_libs
    from ._mlir_libs._mlir import ir
    # ...
except ImportError:
    # Fallback - try alternative import paths
    from .sharp import ir
    # ...

# This allows PySharp to work with or without native bindings
```

## Comparison with Direct MLIR Construction

PySharp provides a more Pythonic alternative to direct MLIR construction:

```python
# PySharp approach
builder = pysharp.ModuleBuilder("Adder")
state = builder.add_state("sum", pysharp.i32)
add_method = builder.add_value_method("add")

# vs. Direct MLIR (would require native bindings)
# module = ir.Module.create()
# with ir.InsertionPoint(module.body):
#     txn_module = txn.ModuleOp(name="Adder")
#     # ... complex MLIR construction ...
```

## Testing

PySharp includes comprehensive tests in `integration_test/pysharp/`:

```bash
# Run standalone test
python integration_test/pysharp/test_standalone.py

# Individual test examples
python integration_test/pysharp/test_types.py
python integration_test/pysharp/test_module_builder.py
python integration_test/pysharp/test_signal_arithmetic.py
```

## Current Limitations

1. **Native binding issues**: The native Sharp extension has runtime loading issues, but PySharp works standalone
2. **MLIR generation**: Full MLIR generation requires fixing the native bindings
3. **Method bodies**: Method body construction is not yet implemented
4. **Type checking**: Type checking is basic - more sophisticated type inference planned

## Future Enhancements

- **Complete MLIR generation** once native bindings are fixed
- **Method body DSL** for defining method implementations
- **Module instantiation** and hierarchical design
- **Simulation support** for testing designs in Python
- **Better error messages** with source location tracking