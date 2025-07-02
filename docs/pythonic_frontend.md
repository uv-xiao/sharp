# Pythonic Construction Frontend

Sharp provides a high-level Pythonic API for constructing hardware modules, similar to CIRCT's PyCDE. This allows hardware designers to use familiar Python syntax and patterns while generating Sharp Txn dialect IR.

## Overview

The Pythonic frontend enables:
- **Decorator-based module definition** using `@module`
- **Type-safe hardware types** (i8, i16, i32, i64, etc.)
- **Operator overloading** for arithmetic and logic operations
- **Automatic conflict matrix management**
- **Integration with existing Python tools and workflows**

## Quick Start

```python
from sharp.construction import module, ModuleBuilder, i32, i8, ConflictRelation

@module
def Counter():
    builder = ModuleBuilder("Counter")
    
    @builder.value_method(return_type=i32)
    def getValue(b):
        return b.constant(42)
        
    @builder.action_method(return_type=i32)
    def increment(b, current: i32):
        one = b.constant(1)
        return current + one
        
    @builder.rule
    def autoIncrement(b):
        # Rules fire automatically
        pass
        
    # Add conflict relationships
    builder.add_conflict("increment", "autoIncrement", ConflictRelation.C)
    
    return builder

# Generate MLIR
mlir_module = Counter.build()
print(mlir_module)
```

## Hardware Types

Sharp provides built-in hardware types:

```python
from sharp.construction import i1, i8, i16, i32, i64, i128, i256, BoolType, IntType

# Predefined types
bool_type = i1        # 1-bit boolean
byte_type = i8        # 8-bit integer  
word_type = i32       # 32-bit integer

# Custom types
custom_type = IntType(width=24)  # 24-bit integer
```

## Method Types

### Value Methods
Always combinational - output immediately depends on inputs:

```python
@builder.value_method(return_type=i32)
def compute(b, a: i32, data: i32):
    # Arithmetic operations
    sum_val = a + data
    doubled = sum_val * 2
    shifted = doubled << 1
    
    # Comparison and selection
    is_zero = sum_val == 0
    result = b.select(is_zero, b.constant(0), shifted)
    
    return result
```

### Action Methods
Can have side effects and optional return values:

```python
@builder.action_method(return_type=i32)
def process(b, input_data: i32):
    # Process the data
    processed = input_data | 1
    return processed

@builder.action_method()  # No return value
def reset(b):
    # Side effects only
    pass
```

### Rules
Fire automatically when their conditions are met:

```python
@builder.rule
def backgroundTask(b):
    counter = b.constant(1)
    # Background processing
    pass
```

## Operator Overloading

The API supports intuitive Python operators:

```python
@builder.value_method(return_type=i32)
def operators_demo(b, a: i32, flag: i1):
    # Arithmetic
    result = a + 10
    result = result * 2
    result = result - 5
    
    # Bitwise operations
    masked = result & 0xFF
    combined = masked | 0x100
    inverted = combined ^ 0xFFFF
    
    # Shifts
    left_shifted = result << 2
    right_shifted = result >> 1
    
    # Comparisons return i1 (boolean)
    is_equal = result == 42
    is_greater = result > 100
    
    # Select based on condition
    final = b.select(flag, result, b.constant(0))
    
    return final
```

## Builder Methods

The `MethodBuilder` provides fine-grained control:

```python
@builder.value_method(return_type=i32)
def manual_operations(b, x: i32, y: i32):
    # Manual constant creation
    c10 = b.constant(10, i32)
    c5 = b.constant(5, i32)
    
    # Manual operations
    sum_xy = b.add(x, y)
    product = b.mul(sum_xy, c10)
    difference = b.sub(product, c5)
    
    # Bitwise operations
    and_result = b.and_(difference, c10)
    or_result = b.or_(and_result, c5)
    
    # Comparisons
    is_less = b.cmp_lt(or_result, b.constant(100))
    
    # Conditional
    result = b.select(is_less, or_result, b.constant(42))
    
    return result
```

## Conflict Management

Define relationships between actions:

```python
# Add conflict relationships
builder.add_conflict("action1", "action2", ConflictRelation.C)   # Conflict
builder.add_conflict("rule1", "action1", ConflictRelation.SB)   # rule1 before action1
builder.add_conflict("method1", "method2", ConflictRelation.SA) # method1 after method2
builder.add_conflict("rule2", "rule3", ConflictRelation.CF)     # Conflict-free
```

## Advanced Example

```python
@module
def ProcessingUnit():
    builder = ModuleBuilder("ProcessingUnit")
    
    @builder.value_method(return_type=i32)
    def computeHash(b, data: i32, seed: i32):
        # Simple hash function
        step1 = data ^ seed
        step2 = step1 * 0x9E37
        step3 = step2 ^ (step2 >> 16)
        return step3
        
    @builder.value_method(return_type=i1)
    def isValid(b, value: i32):
        # Check if value is in valid range
        min_val = b.constant(10)
        max_val = b.constant(1000)
        
        above_min = value >= min_val
        below_max = value <= max_val
        
        return above_min & below_max
        
    @builder.action_method(return_type=i32)
    def processIfValid(b, input_val: i32):
        # Only process if input is valid
        valid = b.call("isValid", [input_val])  # Call value method
        seed = b.constant(0x12345678)
        
        hash_result = b.call("computeHash", [input_val, seed])
        
        # Return hash if valid, otherwise return 0
        return b.select(valid, hash_result, b.constant(0))
        
    @builder.rule
    def monitor(b):
        # Background monitoring rule
        pass
        
    # Set up conflicts
    builder.add_conflict("processIfValid", "monitor", ConflictRelation.C)
    
    return builder

# Build and use
processing_unit = ProcessingUnit.build()
```

## Type Safety

The API provides compile-time type checking:

```python
@builder.value_method(return_type=i8)
def type_safe_method(b, byte_val: i8, word_val: i32):
    # This works - same types
    result8 = byte_val + b.constant(1, i8)
    
    # This would be a type error in a stricter implementation
    # mixed = byte_val + word_val  # Different widths
    
    return result8
```

## Integration with MLIR

The generated MLIR can be processed with Sharp's analysis and conversion passes:

```python
# Generate MLIR
mlir_module = MyModule.build()

# Save to file
with open("generated.mlir", "w") as f:
    f.write(str(mlir_module))

# Or process programmatically using MLIR Python bindings
# (requires additional integration code)
```

## Comparison with PyCDE

| Feature | Sharp Pythonic API | CIRCT PyCDE |
|---------|-------------------|-------------|
| **Target** | Transaction-level hardware | RTL hardware |
| **Abstraction** | Actions, rules, conflicts | Modules, wires, always blocks |
| **Scheduling** | Automatic with conflict matrices | Manual or tool-assisted |
| **Primitives** | Register, Wire, FIFO | Standard Verilog constructs |
| **Analysis** | Built-in loop detection, reachability | Standard CIRCT passes |

## Installation and Setup

1. **Build Sharp with Python bindings**:
   ```bash
   pixi run build
   ```

2. **Set up Python path**:
   ```bash
   export PYTHONPATH=/path/to/sharp/build/lib/Bindings/Python:$PYTHONPATH
   ```

3. **Use in Python**:
   ```python
   from sharp.construction import module, ModuleBuilder, i32
   ```

## Future Enhancements

- **Module composition** - instantiate other modules
- **Primitive integration** - easy primitive definition
- **Simulation support** - execute modules in Python
- **Waveform generation** - debug with timing diagrams
- **Hardware parameter** - parameterized module generation