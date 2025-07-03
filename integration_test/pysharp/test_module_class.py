#!/usr/bin/env python3
# RUN: python %s | FileCheck %s

import sys
sys.path.insert(0, "/home/uvxiao/sharp/build/python_packages")

from sharp import pysharp

# CHECK: === Test 1: Module Class ===
print("=== Test 1: Module Class ===")

class Counter(pysharp.Module):
    # Define state
    count = pysharp.State("count", pysharp.i32)
    
counter = Counter()
# CHECK: Counter module created
print("Counter module created")

# Check builder was created
# CHECK: ModuleBuilder(Counter)
# CHECK:   States: ['count']
print(counter._builder)

# Set conflicts
counter.set_conflict("increment", "decrement", pysharp.ConflictRelation.C)
# CHECK: Conflict set
print("Conflict set")

# CHECK: === Test 2: Module Decorator ===
print("\n=== Test 2: Module Decorator ===")

@pysharp.module("Adder")
class AdderModule:
    # State for result
    result = pysharp.State("result", pysharp.i32)
    
    @pysharp.value_method
    def add(self, a: pysharp.i32, b: pysharp.i32) -> pysharp.i32:
        return a + b
        
    @pysharp.action_method
    def store_result(self, value: pysharp.i32):
        self.result.write(value)
        
# CHECK: Adder
print(AdderModule._module_name)

# Check method decorators
# CHECK: value
print(AdderModule.add._pysharp_method_type)
# CHECK: action
print(AdderModule.store_result._pysharp_method_type)

# CHECK: === Test 3: Complex Module ===
print("\n=== Test 3: Complex Module ===")

@pysharp.module()
class FIFO:
    # States
    data = pysharp.State("data", pysharp.i32)
    valid = pysharp.State("valid", pysharp.i1)
    
    @pysharp.value_method
    def canEnqueue(self) -> pysharp.i1:
        return ~self.valid.read()
        
    @pysharp.value_method
    def canDequeue(self) -> pysharp.i1:
        return self.valid.read()
        
    @pysharp.action_method
    def enqueue(self, value: pysharp.i32):
        self.data.write(value)
        self.valid.write(pysharp.constant(True, pysharp.i1))
        
    @pysharp.action_method
    def dequeue(self) -> pysharp.i32:
        self.valid.write(pysharp.constant(False, pysharp.i1))
        return self.data.read()
        
    @pysharp.rule
    def autoReset(self):
        # Rule that automatically resets when a condition is met
        pass

# CHECK: FIFO
print(FIFO._module_name)

# Create instance
fifo = pysharp.Module("FIFO")
# CHECK: ModuleBuilder(FIFO)
print(fifo._builder)

print("\n✅ All module class tests passed!")