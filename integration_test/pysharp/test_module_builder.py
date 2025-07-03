#!/usr/bin/env python3
# RUN: python %s | FileCheck %s

import sys
sys.path.insert(0, "/home/uvxiao/sharp/build/python_packages")

from sharp import pysharp

# CHECK: === Test 1: Module Builder ===
print("=== Test 1: Module Builder ===")

builder = pysharp.ModuleBuilder("Counter")
# CHECK: ModuleBuilder(Counter)
print(builder)

# Add state
# CHECK: === Test 2: Add State ===
print("\n=== Test 2: Add State ===")
count_state = builder.add_state("count", pysharp.i32)
# CHECK: State(count: i32)
print(count_state)
# CHECK: ModuleBuilder(Counter)
# CHECK:   States: ['count']
print(builder)

# Add methods
# CHECK: === Test 3: Add Methods ===
print("\n=== Test 3: Add Methods ===")
get_value = builder.add_value_method("getValue")
increment = builder.add_action_method("increment")
decrement = builder.add_action_method("decrement")

# CHECK: ModuleBuilder(Counter)
# CHECK:   States: ['count']
# CHECK:   Methods: ['getValue', 'increment', 'decrement']
print(builder)

# Add rule
# CHECK: === Test 4: Add Rule ===
print("\n=== Test 4: Add Rule ===")
auto_inc = builder.add_rule("autoIncrement")

# CHECK: ModuleBuilder(Counter)
# CHECK:   States: ['count']
# CHECK:   Methods: ['getValue', 'increment', 'decrement']
# CHECK:   Rules: ['autoIncrement']
print(builder)

# Set conflicts
# CHECK: === Test 5: Set Conflicts ===
print("\n=== Test 5: Set Conflicts ===")
builder.set_conflict("increment", "decrement", pysharp.ConflictRelation.C)
builder.set_conflict("autoIncrement", "increment", pysharp.ConflictRelation.SB)
# CHECK: Conflicts set successfully
print("Conflicts set successfully")

# Test state operations
# CHECK: === Test 6: State Operations ===
print("\n=== Test 6: State Operations ===")
read_val = count_state.read()
# CHECK: Signal(read_count: i32)
print(read_val)

# Test write operation returns tuple
write_op = count_state.write(read_val)
# CHECK: ('write', State(count: i32), Signal(read_count: i32))
print(write_op)

print("\n✅ All module builder tests passed!")