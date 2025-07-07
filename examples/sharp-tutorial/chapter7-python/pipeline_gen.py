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
        
        # Schedule with proper conflicts (only actions, not value methods)
        m.schedule(
            methods=["input", "advance", "clock"],
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