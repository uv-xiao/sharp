#!/usr/bin/env python3
# RUN: python %s | FileCheck %s

import sys
sys.path.insert(0, "/home/uvxiao/sharp/build/python_packages")

from sharp import pysharp

# Test basic types
# CHECK: === Test 1: Basic Integer Types ===
print("=== Test 1: Basic Integer Types ===")
# CHECK: i1
print(pysharp.i1)
# CHECK: i8
print(pysharp.i8)
# CHECK: i16
print(pysharp.i16)
# CHECK: i32
print(pysharp.i32)
# CHECK: i64
print(pysharp.i64)

# Test FIRRTL types
# CHECK: === Test 2: FIRRTL Types ===
print("\n=== Test 2: FIRRTL Types ===")
# CHECK: uint<8>
print(pysharp.uint(8))
# CHECK: uint<16>
print(pysharp.uint(16))
# CHECK: sint<32>
print(pysharp.sint(32))

# Test conflict relations
# CHECK: === Test 3: Conflict Relations ===
print("\n=== Test 3: Conflict Relations ===")
# CHECK: SB=0
print(f"SB={pysharp.ConflictRelation.SB}")
# CHECK: SA=1
print(f"SA={pysharp.ConflictRelation.SA}")
# CHECK: C=2
print(f"C={pysharp.ConflictRelation.C}")
# CHECK: CF=3
print(f"CF={pysharp.ConflictRelation.CF}")

# Test signals
# CHECK: === Test 4: Signals ===
print("\n=== Test 4: Signals ===")
a = pysharp.Signal("a", pysharp.i32)
b = pysharp.Signal("b", pysharp.i32)
# CHECK: Signal(a: i32)
print(a)
# CHECK: Signal(b: i32)
print(b)

# Test operations
# CHECK: === Test 5: Signal Operations ===
print("\n=== Test 5: Signal Operations ===")
c = a + b
# CHECK: Signal((a + b): i32)
print(c)
d = a - b
# CHECK: Signal((a - b): i32)
print(d)
e = a & b
# CHECK: Signal((a & b): i32)
print(e)

# Test constants
# CHECK: === Test 6: Constants ===
print("\n=== Test 6: Constants ===")
const1 = pysharp.constant(42, pysharp.i32)
# CHECK: Signal(const_42: i32)
print(const1)
const2 = pysharp.constant(True, pysharp.i1)
# CHECK: Signal(const_True: i1)
print(const2)

print("\n✅ All type tests passed!")