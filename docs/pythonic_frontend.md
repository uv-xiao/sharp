# PySharp: Pythonic Frontend

## Overview

PySharp provides a Pythonic API for constructing hardware modules using Sharp's transaction-based semantics, following CIRCT's PyCDE pattern.

## Installation

```bash
# Build Sharp with Python bindings
pixi run build

# Set Python path
export PYTHONPATH=$PYTHONPATH:/path/to/sharp/build/python_packages:/path/to/sharp/build/python_packages/pysharp

# Import
import pysharp
```

## Basic Usage

### Types
```python
from pysharp import i8, i32, i64, uint, sint

# Standard integer types
x = i32  # 32-bit integer
y = uint(16)  # 16-bit unsigned FIRRTL type
```

### Signals and Operations
```python
from pysharp import Signal, constant

# Create signals
a = Signal("a", i32)
b = Signal("b", i32)

# Operations
c = a + b  # Addition
d = a & 0xFF  # Bitwise AND
e = constant(42, i32)  # Constant
```

### Module Builder
```python
from pysharp import ModuleBuilder, i32

# Create module
builder = ModuleBuilder("Counter")

# Add state
count = builder.add_state("count", i32)

# Add methods
get_value = builder.add_value_method("getValue")
get_value.returns(count.read())

increment = builder.add_action_method("increment")
with increment.body():
    val = count.read()
    count.write(val + 1)

# Add schedule
builder.add_schedule(["increment"])

# Generate MLIR
module = builder.build()
```

### Decorator Syntax
```python
from pysharp import module, value_method, action_method, rule, schedule

@module("Counter")
class Counter:
    def __init__(self):
        self.count = Register(i32)
    
    @value_method
    def getValue(self) -> i32:
        return self.count.read()
    
    @action_method
    def setValue(self, value: i32):
        self.count.write(value)
    
    @rule
    def autoIncrement(self):
        if self.getValue() < 100:
            self.setValue(self.getValue() + 1)
    
    @schedule
    def main_schedule(self):
        return ["setValue", "autoIncrement"]
```

## Advanced Features

### Conflict Management
```python
from pysharp import ConflictType

# Specify conflicts
schedule = builder.add_schedule(
    ["rule1", "rule2"],
    conflicts={
        ("rule1", "rule2"): ConflictType.C  # Cannot execute together
    }
)
```

### Parametric Modules
```python
def create_fifo(depth: int, data_type):
    builder = ModuleBuilder(f"FIFO_{depth}")
    # Implementation...
    return builder.build()

# Create different FIFOs
fifo8 = create_fifo(8, i32)
fifo16 = create_fifo(16, i64)
```

### Instance Creation
```python
# In module builder
reg = builder.add_instance("reg", "Register", i32)

# Use in methods
val = reg.call("read")
reg.call("write", new_value)
```

## Interoperability

PySharp generates standard Sharp MLIR:
```python
# Generate MLIR
mlir_module = builder.build()

# Convert to string
mlir_text = str(mlir_module)

# Save to file
with open("output.mlir", "w") as f:
    f.write(mlir_text)
```

## Examples

### Counter with Reset
```python
@module("ResetCounter")
class ResetCounter:
    def __init__(self):
        self.count = Register(i32)
    
    @action_method
    def increment(self):
        self.count.write(self.count.read() + 1)
    
    @action_method
    def reset(self):
        self.count.write(0)
    
    @schedule
    def schedule(self):
        return ["reset", "increment"], {
            ("reset", "increment"): ConflictType.SB  # Reset before increment
        }
```

### Pipeline Stage
```python
def pipeline_stage(name: str, process_fn):
    builder = ModuleBuilder(name)
    
    valid = builder.add_state("valid", i1)
    data = builder.add_state("data", i32)
    
    enqueue = builder.add_action_method("enqueue", [("input", i32)])
    with enqueue.body():
        if not valid.read():
            data.write(process_fn(enqueue.arg("input")))
            valid.write(True)
    
    dequeue = builder.add_action_method("dequeue")
    dequeue.returns(data.read())
    with dequeue.body():
        if valid.read():
            valid.write(False)
    
    return builder.build()
```