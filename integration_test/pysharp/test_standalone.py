#!/usr/bin/env python3
# Test PySharp functionality without native bindings

import sys
# Import pysharp directly from source
sys.path.insert(0, "/home/uvxiao/sharp/lib/Bindings/Python")

import pysharp

# Test 1: Basic types
print("=== Test 1: Basic Types ===")
print(f"i8: {pysharp.i8}")
print(f"i32: {pysharp.i32}")
print(f"uint(16): {pysharp.uint(16)}")
print(f"sint(32): {pysharp.sint(32)}")

# Test 2: Conflict relations
print("\n=== Test 2: Conflict Relations ===")
print(f"SB={pysharp.ConflictRelation.SB}")
print(f"SA={pysharp.ConflictRelation.SA}")
print(f"C={pysharp.ConflictRelation.C}")
print(f"CF={pysharp.ConflictRelation.CF}")

# Test 3: Signals
print("\n=== Test 3: Signals ===")
a = pysharp.Signal("a", pysharp.i32)
b = pysharp.Signal("b", pysharp.i32)
print(f"Signal a: {a}")
print(f"Signal b: {b}")

# Test 4: Operations
print("\n=== Test 4: Signal Operations ===")
c = a + b
print(f"a + b: {c}")
d = a - 5
print(f"a - 5: {d}")
e = a & b
print(f"a & b: {e}")

# Test 5: Module builder
print("\n=== Test 5: Module Builder ===")
builder = pysharp.ModuleBuilder("Counter")
count_state = builder.add_state("count", pysharp.i32)
get_value = builder.add_value_method("getValue")
increment = builder.add_action_method("increment")
builder.set_conflict("increment", "decrement", pysharp.ConflictRelation.C)
print(builder)

# Test 6: Module class
print("\n=== Test 6: Module Class ===")

@pysharp.module("TestModule")
class TestModule:
    # State
    data = pysharp.State("data", pysharp.i32)
    
    @pysharp.value_method
    def getData(self):
        return self.data.read()
        
    @pysharp.action_method
    def setData(self, value):
        self.data.write(value)

print(f"Module name: {TestModule._module_name}")
print(f"getData method type: {TestModule.getData._pysharp_method_type}")
print(f"setData method type: {TestModule.setData._pysharp_method_type}")

# Test 7: Module instance
print("\n=== Test 7: Module Instance ===")
mod = pysharp.Module("MyModule")
print(f"Module builder: {mod._builder}")

print("\n✅ All standalone tests passed!")