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