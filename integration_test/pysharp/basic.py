#!/usr/bin/env python3
# RUN: cd %S && python %s | FileCheck %s

import sys
# Import pysharp directly from source
sys.path.insert(0, "/home/uvxiao/sharp/lib/Bindings/Python")

import pysharp

# CHECK: i32
print(pysharp.i32)

# CHECK: uint<16>
print(pysharp.uint(16))

# CHECK: ConflictRelation.C=2
print(f"ConflictRelation.C={pysharp.ConflictRelation.C}")

# CHECK: Signal(x: i32)
x = pysharp.Signal("x", pysharp.i32)
print(x)

# CHECK: Signal((x + 10): i32)
y = x + 10
print(y)

# CHECK: ModuleBuilder(Test)
# CHECK-NEXT:   States: ['data']
# CHECK-NEXT:   Methods: ['read']
builder = pysharp.ModuleBuilder("Test")
builder.add_state("data", pysharp.i32)
builder.add_value_method("read")
print(builder)