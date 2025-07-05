#!/usr/bin/env python3
# RUN: %python %s
"""Test PySharp functionality with Sharp integration."""

import sys
import os

# Import Sharp bindings
import sharp
from sharp import ir
from sharp.dialects import txn
print("✓ Successfully imported Sharp")

# Import PySharp
import pysharp
from pysharp import module, value_method, action_method, i32
print("✓ Successfully imported PySharp")

# Create a context and register all dialects
with ir.Context() as ctx:
    # Register Sharp dialects
    sharp.register_sharp_dialects(ctx)
    print("✓ Registered all dialects")
    
    # Test creating a simple Txn module
    module_str = """
    txn.module @TestModule {
        txn.value_method @getValue() -> i32 {
            %c42_i32 = arith.constant 42 : i32
            txn.return %c42_i32 : i32
        }
        txn.schedule []
    }
    """
    
    mlir_module = ir.Module.parse(module_str)
    print("✓ Successfully parsed Txn module")

# Test PySharp decorators
@module
class Counter:
    count = i32
    
    @value_method
    def get_value(self) -> i32:
        return self.count.read()
    
    @action_method
    def increment(self):
        self.count.write(self.count.read() + 1)

print("✓ Successfully created PySharp module")
print("\n✅ All tests passed!")