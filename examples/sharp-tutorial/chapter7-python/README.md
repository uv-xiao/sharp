# Chapter 7: Python Frontend

## Overview

Sharp provides a Python frontend for building hardware designs programmatically. This enables:
- Dynamic module generation
- Integration with Python libraries
- Rapid prototyping
- Algorithmic hardware generation

## Python API Basics

### Creating Modules

```python
from pysharp import *

# Create a module
with Module("Adder") as m:
    # Define methods
    @m.value_method
    def add(a: i32, b: i32) -> i32:
        return a + b
    
    # Define action methods
    @m.action_method
    def accumulate(value: i32):
        # Access module state
        acc = m.instance("acc", Register[i32]())
        current = acc.read()
        acc.write(current + value)
```

### Type System

Sharp's Python frontend supports hardware types:

```python
# Integer types
i1, i8, i16, i32, i64  # Signed integers
u1, u8, u16, u32, u64  # Unsigned integers

# Array types
Array[i32, 10]         # Fixed-size array
Array[i8, N]           # Parameterized size

# Module types
Register[i32]          # Typed register
FIFO[i64]             # Typed FIFO
Wire[i16]             # Typed wire
```

## Complete Examples

### counter.py

```python
#!/usr/bin/env python3
"""Simple counter module in Python"""

from pysharp import *

def create_counter(width=32):
    """Create a parameterized counter module"""
    
    int_type = IntType.get_signless(width)
    
    with Module(f"Counter{width}") as m:
        # State: counter register
        count = m.instance("count", Register[int_type]())
        
        @m.value_method
        def read() -> int_type:
            """Read current count"""
            return count.read()
        
        @m.action_method
        def increment():
            """Increment counter"""
            val = count.read()
            count.write(val + 1)
        
        @m.action_method
        def reset():
            """Reset counter to zero"""
            count.write(0)
        
        @m.rule
        def auto_increment():
            """Auto-increment every cycle"""
            if count.read() < (2**width - 1):
                m.call("increment")
        
        # Define schedule with conflicts
        m.schedule(
            methods=["read", "increment", "reset", "auto_increment"],
            conflicts={
                ("increment", "reset"): ConflictType.C,
                ("increment", "auto_increment"): ConflictType.C,
                ("reset", "auto_increment"): ConflictType.C
            }
        )
    
    return m

# Generate different counter sizes
if __name__ == "__main__":
    # Create 8-bit counter
    counter8 = create_counter(8)
    print(counter8.to_mlir())
    
    # Create 32-bit counter
    counter32 = create_counter(32)
    print(counter32.to_mlir())
```

### pipeline_gen.py

```python
#!/usr/bin/env python3
"""Generate parameterized pipeline stages"""

from pysharp import *

def create_pipeline(stages=3, width=32, operations=None):
    """Create a configurable pipeline
    
    Args:
        stages: Number of pipeline stages
        width: Bit width of data
        operations: List of operations per stage
    """
    
    if operations is None:
        operations = ["add", "mul", "xor"]
    
    int_type = IntType.get_signless(width)
    
    with Module(f"Pipeline{stages}x{width}") as m:
        # Create pipeline registers
        regs = []
        for i in range(stages + 1):
            reg = m.instance(f"stage{i}", Register[int_type]())
            regs.append(reg)
        
        @m.action_method
        def input(data: int_type):
            """Input data to pipeline"""
            regs[0].write(data)
        
        @m.action_method
        def advance():
            """Advance pipeline by one stage"""
            # Read all stages
            values = [reg.read() for reg in regs]
            
            # Apply operations and advance
            for i in range(stages):
                if i < len(operations):
                    op = operations[i]
                    if op == "add":
                        result = values[i] + 1
                    elif op == "mul":
                        result = values[i] * 2
                    elif op == "xor":
                        result = values[i] ^ 0xFF
                    else:
                        result = values[i]
                else:
                    result = values[i]
                
                regs[i + 1].write(result)
        
        @m.value_method
        def output() -> int_type:
            """Get pipeline output"""
            return regs[-1].read()
        
        @m.rule
        def clock():
            """Auto-advance pipeline"""
            m.call("advance")
        
        # Schedule with proper conflicts
        m.schedule(
            methods=["input", "advance", "output", "clock"],
            conflicts={
                ("input", "advance"): ConflictType.C,
                ("advance", "clock"): ConflictType.CF
            }
        )
    
    return m

# Example usage
if __name__ == "__main__":
    # Create different pipeline configurations
    pipe1 = create_pipeline(3, 32, ["add", "mul", "xor"])
    pipe2 = create_pipeline(5, 16, ["mul", "add", "add", "xor", "mul"])
    
    # Save to MLIR files
    with open("pipeline_3x32.mlir", "w") as f:
        f.write(pipe1.to_mlir())
    
    with open("pipeline_5x16.mlir", "w") as f:
        f.write(pipe2.to_mlir())
```

### matrix_mult.py

```python
#!/usr/bin/env python3
"""Generate systolic array for matrix multiplication"""

from pysharp import *

def create_pe(row, col):
    """Create a Processing Element for systolic array"""
    
    with Module(f"PE_{row}_{col}") as pe:
        # Local accumulator
        acc = pe.instance("acc", Register[i32]())
        
        # Input registers
        a_in = pe.instance("a_in", Register[i32]())
        b_in = pe.instance("b_in", Register[i32]())
        
        @pe.action_method
        def compute():
            """Multiply and accumulate"""
            a = a_in.read()
            b = b_in.read()
            prod = a * b
            acc.write(acc.read() + prod)
        
        @pe.action_method
        def propagate_a(a_out: Wire[i32]):
            """Pass A value to next PE"""
            a_out.write(a_in.read())
        
        @pe.action_method
        def propagate_b(b_out: Wire[i32]):
            """Pass B value to next PE"""
            b_out.write(b_in.read())
        
        @pe.value_method
        def get_result() -> i32:
            """Read accumulated result"""
            return acc.read()
        
        pe.schedule(["compute", "propagate_a", "propagate_b", "get_result"])
    
    return pe

def create_systolic_array(size=4):
    """Create NxN systolic array for matrix multiplication"""
    
    with Module(f"SystolicArray{size}x{size}") as m:
        # Create PE array
        pes = []
        for i in range(size):
            row = []
            for j in range(size):
                pe = m.instance(f"pe_{i}_{j}", create_pe(i, j))
                row.append(pe)
            pes.append(row)
        
        # Create interconnect wires
        h_wires = []  # Horizontal wires (A values)
        v_wires = []  # Vertical wires (B values)
        
        for i in range(size):
            h_row = []
            v_row = []
            for j in range(size + 1):
                if j < size:
                    h_row.append(m.instance(f"h_wire_{i}_{j}", Wire[i32]()))
                if i < size:
                    v_row.append(m.instance(f"v_wire_{i}_{j}", Wire[i32]()))
            h_wires.append(h_row)
            v_wires.append(v_row)
        
        # Input methods for matrix A (left side)
        for i in range(size):
            @m.action_method(name=f"input_a_{i}")
            def input_a(row=i, value: i32):
                h_wires[row][0].write(value)
        
        # Input methods for matrix B (top side)
        for j in range(size):
            @m.action_method(name=f"input_b_{j}")
            def input_b(col=j, value: i32):
                v_wires[0][col].write(value)
        
        @m.action_method
        def compute_cycle():
            """One computation cycle of systolic array"""
            # Each PE reads inputs, computes, and propagates
            for i in range(size):
                for j in range(size):
                    pe = pes[i][j]
                    
                    # Read inputs from wires
                    if j == 0:
                        a_val = h_wires[i][j].read()
                    else:
                        a_val = pes[i][j-1].a_in.read()
                    
                    if i == 0:
                        b_val = v_wires[i][j].read()
                    else:
                        b_val = pes[i-1][j].b_in.read()
                    
                    # Update PE inputs
                    pe.a_in.write(a_val)
                    pe.b_in.write(b_val)
                    
                    # Compute
                    pe.compute()
                    
                    # Propagate
                    if j < size - 1:
                        pe.propagate_a(h_wires[i][j+1])
                    if i < size - 1:
                        pe.propagate_b(v_wires[i+1][j])
        
        # Output methods
        for i in range(size):
            for j in range(size):
                @m.value_method(name=f"result_{i}_{j}")
                def get_result(row=i, col=j) -> i32:
                    return pes[row][col].get_result()
        
        # Build method list for scheduling
        methods = []
        methods.extend([f"input_a_{i}" for i in range(size)])
        methods.extend([f"input_b_{j}" for j in range(size)])
        methods.append("compute_cycle")
        methods.extend([f"result_{i}_{j}" for i in range(size) for j in range(size)])
        
        m.schedule(methods)
    
    return m

if __name__ == "__main__":
    # Generate different sizes
    sa_2x2 = create_systolic_array(2)
    sa_4x4 = create_systolic_array(4)
    
    print("Generated 2x2 systolic array")
    print("Generated 4x4 systolic array")
```

### advanced_features.py

```python
#!/usr/bin/env python3
"""Advanced Python frontend features"""

from pysharp import *
import math

def create_fft_stage(n, stage):
    """Create one stage of FFT butterfly network"""
    
    with Module(f"FFTStage{n}_S{stage}") as m:
        # Complex number as pair of i32
        complex_t = StructType([("real", i32), ("imag", i32)])
        
        # Input/output arrays
        inputs = [m.instance(f"in{i}", Wire[complex_t]()) for i in range(n)]
        outputs = [m.instance(f"out{i}", Wire[complex_t]()) for i in range(n)]
        
        # Twiddle factors (precomputed)
        stride = 2 ** (stage + 1)
        group_size = stride // 2
        
        @m.action_method
        def compute():
            """Perform butterfly operations"""
            for group_start in range(0, n, stride):
                for i in range(group_size):
                    # Butterfly indices
                    top = group_start + i
                    bot = top + group_size
                    
                    # Read inputs
                    a = inputs[top].read()
                    b = inputs[bot].read()
                    
                    # Twiddle factor (simplified)
                    angle = -2 * math.pi * i / stride
                    w_real = int(math.cos(angle) * 1024)  # Fixed point
                    w_imag = int(math.sin(angle) * 1024)
                    
                    # Complex multiply b * w
                    b_rot_real = (b.real * w_real - b.imag * w_imag) >> 10
                    b_rot_imag = (b.real * w_imag + b.imag * w_real) >> 10
                    
                    # Butterfly
                    outputs[top].write(
                        complex_t(a.real + b_rot_real, a.imag + b_rot_imag)
                    )
                    outputs[bot].write(
                        complex_t(a.real - b_rot_real, a.imag - b_rot_imag)
                    )
        
        m.schedule(["compute"])
    
    return m

def create_parameterized_fifo(depth, width, name="ParamFIFO"):
    """Create a parameterized FIFO with custom depth and width"""
    
    data_type = IntType.get_signless(width)
    
    with Module(name) as m:
        # Storage array
        storage = [m.instance(f"slot{i}", Register[data_type]()) 
                  for i in range(depth)]
        
        # Head and tail pointers
        ptr_width = math.ceil(math.log2(depth))
        ptr_type = IntType.get_signless(ptr_width)
        
        head = m.instance("head", Register[ptr_type]())
        tail = m.instance("tail", Register[ptr_type]())
        count = m.instance("count", Register[ptr_type]())
        
        @m.action_method
        def enqueue(data: data_type):
            """Add element to FIFO"""
            if count.read() < depth:
                storage[tail.read()].write(data)
                tail.write((tail.read() + 1) % depth)
                count.write(count.read() + 1)
        
        @m.action_method
        def dequeue() -> data_type:
            """Remove element from FIFO"""
            if count.read() > 0:
                data = storage[head.read()].read()
                head.write((head.read() + 1) % depth)
                count.write(count.read() - 1)
                return data
            return 0
        
        @m.value_method
        def is_empty() -> i1:
            """Check if FIFO is empty"""
            return count.read() == 0
        
        @m.value_method
        def is_full() -> i1:
            """Check if FIFO is full"""
            return count.read() == depth
        
        @m.value_method
        def occupancy() -> ptr_type:
            """Get number of elements"""
            return count.read()
        
        m.schedule(
            ["enqueue", "dequeue", "is_empty", "is_full", "occupancy"],
            conflicts={
                ("enqueue", "dequeue"): ConflictType.SB
            }
        )
    
    return m

# Integration with numpy for verification
def create_convolution_engine(kernel_size=3, channels=16):
    """Create hardware convolution engine"""
    
    with Module(f"Conv{kernel_size}x{kernel_size}x{channels}") as m:
        # This would integrate with numpy arrays for kernel weights
        # Implementation details omitted for brevity
        pass
    
    return m

if __name__ == "__main__":
    # Generate various parameterized modules
    fifo_shallow = create_parameterized_fifo(4, 32, "ShallowFIFO")
    fifo_deep = create_parameterized_fifo(256, 64, "DeepFIFO")
    
    fft_8_stage0 = create_fft_stage(8, 0)
    fft_8_stage1 = create_fft_stage(8, 1)
    fft_8_stage2 = create_fft_stage(8, 2)
    
    print("Generated parameterized hardware modules")
```

### run.sh

```bash
#!/bin/bash

SHARP_ROOT="../../.."
SHARP_OPT="$SHARP_ROOT/build/bin/sharp-opt"

echo "=== Chapter 7: Python Frontend ==="
echo ""

echo "1. Simple Counter Generation:"
echo "----------------------------------------"
python3 counter.py > counter_generated.mlir
if [ -f counter_generated.mlir ]; then
    echo "✅ Counter modules generated"
    echo "Validating generated MLIR..."
    if $SHARP_OPT counter_generated.mlir > /dev/null 2>&1; then
        echo "✅ Generated MLIR is valid"
    else
        echo "❌ Generated MLIR has errors"
    fi
else
    echo "❌ Counter generation failed"
fi
echo ""

echo "2. Pipeline Generator:"
echo "----------------------------------------"
python3 pipeline_gen.py
if [ -f pipeline_3x32.mlir ] && [ -f pipeline_5x16.mlir ]; then
    echo "✅ Pipeline variants generated"
    echo "  - pipeline_3x32.mlir (3 stages, 32-bit)"
    echo "  - pipeline_5x16.mlir (5 stages, 16-bit)"
else
    echo "❌ Pipeline generation failed"
fi
echo ""

echo "3. Systolic Array Generation:"
echo "----------------------------------------"
python3 matrix_mult.py > systolic_arrays.log
if [ $? -eq 0 ]; then
    echo "✅ Systolic arrays generated"
    cat systolic_arrays.log
else
    echo "❌ Systolic array generation failed"
fi
echo ""

echo "4. Advanced Features Demo:"
echo "----------------------------------------"
python3 advanced_features.py
if [ $? -eq 0 ]; then
    echo "✅ Advanced modules generated"
    echo "  - Parameterized FIFOs"
    echo "  - FFT stages"
    echo "  - Complex arithmetic"
else
    echo "❌ Advanced feature generation failed"
fi
echo ""

echo "5. Integration Example:"
echo "----------------------------------------"
# Show how Python-generated modules can be simulated
if [ -f counter_generated.mlir ]; then
    $SHARP_ROOT/tools/generate-workspace.sh counter_generated.mlir counter_sim
    if [ -d counter_sim ]; then
        echo "✅ Python-generated module ready for simulation"
        echo "Build with: cd counter_sim && mkdir build && cd build && cmake .. && make"
    fi
fi
```

## Python API Reference

### Module Context

```python
with Module("ModuleName") as m:
    # Module definition
    pass
```

### Method Decorators

```python
@m.value_method          # Pure functions, no side effects
@m.action_method         # Can modify state
@m.rule                  # Autonomous behavior
```

### Primitive Instantiation

```python
reg = m.instance("name", Register[i32]())
wire = m.instance("name", Wire[i8]())
fifo = m.instance("name", FIFO[i64]())
```

### Type Annotations

```python
def method(a: i32, b: i32) -> i32:
    return a + b
```

### Scheduling

```python
m.schedule(
    methods=["m1", "m2", "m3"],
    conflicts={
        ("m1", "m2"): ConflictType.C,
        ("m2", "m3"): ConflictType.CF
    }
)
```

## Best Practices

1. **Use Type Hints**: Always annotate parameter and return types
2. **Parameterize Modules**: Make designs reusable with parameters
3. **Generate Don't Duplicate**: Use Python loops for repetitive structures
4. **Validate Output**: Always check generated MLIR is valid
5. **Document Intent**: Add docstrings to methods

## Integration Patterns

### With NumPy

```python
import numpy as np

# Use NumPy for coefficient calculation
coeffs = np.array([...])

# Generate hardware from coefficients
for i, coeff in enumerate(coeffs):
    # Create hardware multiplier
    pass
```

### With Machine Learning

```python
# Generate hardware accelerator from trained model
def ml_to_hardware(model):
    with Module("Accelerator") as m:
        # Convert layers to hardware
        pass
```

## Exercises

1. **Create a parameterized FIR filter generator**
2. **Build a state machine compiler from Python**
3. **Generate a crossbar switch of arbitrary size**
4. **Create hardware from a mathematical expression**

## Common Patterns

### Factory Functions

```python
def create_module(config):
    """Factory for parameterized modules"""
    with Module(config.name) as m:
        # Build based on config
        pass
    return m
```

### Hardware Generators

```python
class HardwareGenerator:
    """Base class for hardware generators"""
    
    def generate(self, params):
        """Generate hardware from parameters"""
        pass
```

## Debugging Tips

1. **Print Generated MLIR**: Use `module.to_mlir()` to inspect output
2. **Validate Incrementally**: Test each generated component
3. **Use Assertions**: Add checks in generated code
4. **Compare with Hand-Written**: Verify against manual MLIR

## Key Takeaways

- Python frontend enables algorithmic hardware generation
- Full integration with Sharp's type system
- Parameterized designs are easy to create
- Seamless path from Python to simulation/synthesis

## Next Chapter

Chapter 8 covers advanced topics:
- Custom primitives
- Formal verification integration
- Performance optimization
- Real-world case studies