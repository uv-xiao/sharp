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