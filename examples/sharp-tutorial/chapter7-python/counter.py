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
        
        # Define schedule with conflicts (only actions, not value methods)
        m.schedule(
            methods=["increment", "reset", "auto_increment"],
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