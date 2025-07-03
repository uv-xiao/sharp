#!/usr/bin/env python3
# RUN: python %s | FileCheck %s

import sys
sys.path.insert(0, "/home/uvxiao/sharp/build/python_packages")

from sharp import pysharp

# CHECK: === Test 1: Basic Arithmetic ===
print("=== Test 1: Basic Arithmetic ===")

a = pysharp.Signal("a", pysharp.i32)
b = pysharp.Signal("b", pysharp.i32)
c = pysharp.constant(10, pysharp.i32)

# Test addition
result1 = a + b
# CHECK: Signal((a + b): i32)
print(result1)

result2 = a + 5
# CHECK: Signal((a + 5): i32)
print(result2)

result3 = c + b
# CHECK: Signal((const_10 + b): i32)
print(result3)

# Test subtraction
result4 = a - b
# CHECK: Signal((a - b): i32)
print(result4)

result5 = a - 3
# CHECK: Signal((a - 3): i32)
print(result5)

# CHECK: === Test 2: Bitwise Operations ===
print("\n=== Test 2: Bitwise Operations ===")

x = pysharp.Signal("x", pysharp.i8)
y = pysharp.Signal("y", pysharp.i8)

# Test AND
result6 = x & y
# CHECK: Signal((x & y): i8)
print(result6)

result7 = x & 0xFF
# CHECK: Signal((x & 255): i8)
print(result7)

# Test OR
result8 = x | y
# CHECK: Signal((x | y): i8)
print(result8)

result9 = x | 0x0F
# CHECK: Signal((x | 15): i8)
print(result9)

# CHECK: === Test 3: Chained Operations ===
print("\n=== Test 3: Chained Operations ===")

# Create more complex expressions
p = pysharp.Signal("p", pysharp.i16)
q = pysharp.Signal("q", pysharp.i16)
r = pysharp.Signal("r", pysharp.i16)

# Chained additions
result10 = p + q + r
# CHECK: Signal(((p + q) + r): i16)
print(result10)

# Mixed operations
result11 = (p + q) & r
# CHECK: Signal(((p + q) & r): i16)
print(result11)

# With constants
result12 = (p + 100) - q
# CHECK: Signal(((p + 100) - q): i16)
print(result12)

# CHECK: === Test 4: Type Preservation ===
print("\n=== Test 4: Type Preservation ===")

# Different bit widths
sig8 = pysharp.Signal("sig8", pysharp.i8)
sig16 = pysharp.Signal("sig16", pysharp.i16)
sig32 = pysharp.Signal("sig32", pysharp.i32)

# Operations preserve the type of the left operand
result13 = sig8 + 1
# CHECK: i8
print(result13.type)

result14 = sig16 - 2
# CHECK: i16
print(result14.type)

result15 = sig32 & 0xFFFF
# CHECK: i32
print(result15.type)

print("\n✅ All signal arithmetic tests passed!")