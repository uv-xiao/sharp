#!/usr/bin/env python3
"""Simple counter example using Sharp's Python API"""

import sys
sys.path.append('/home/uvxiao/sharp/build/python_packages')

from pysharp import *

@module("Counter", clock="clk", reset="rst")
class Counter(ModuleBuilder):
    def __init__(self):
        super().__init__()
        
        # State
        self.count = self.instance("count", "Register", i32)
        
        # Value method to read count
        @self.value_method("read", timing="combinational")
        def read():
            return self.count.read()
        
        # Action to increment
        @self.action_method("increment", timing="static(1)")
        def increment():
            current = self.count.read()
            self.count.write(current + 1)
        
        # Action to decrement  
        @self.action_method("decrement", timing="static(1)")
        def decrement():
            current = self.count.read()
            self.count.write(current - 1)
            
        # Action to reset
        @self.action_method("reset", timing="static(1)")
        def reset():
            self.count.write(0)
        
        # Define schedule with conflicts
        self.schedule(
            ["increment", "decrement", "reset"],
            conflict_matrix={
                ("increment", "decrement"): ConflictRelation.C,
                ("increment", "reset"): ConflictRelation.C,
                ("decrement", "reset"): ConflictRelation.C,
            }
        )

if __name__ == "__main__":
    # Build and print the module
    counter = Counter()
    print(counter.build())