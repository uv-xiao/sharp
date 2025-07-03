#!/usr/bin/env python3
"""Test PySharp frontend functionality."""

import sys
import os

# Add Sharp to path
sys.path.insert(0, "/home/uvxiao/sharp/build/python_packages")

try:
    # Import PySharp
    import sharp
    from sharp import (
        ConflictRelation, ModuleBuilder,
        i8, i16, i32, i64,
        uint, sint, clock, reset,
        Value, State, constant
    )
    
    print("✓ Successfully imported PySharp frontend")
    
    # Test 1: Basic types
    print("\n=== Test 1: Type System ===")
    print(f"i8: {i8}")
    print(f"i32: {i32}")
    print(f"uint(16): {uint(16)}")
    print(f"sint(32): {sint(32)}")
    print(f"clock: {clock}")
    print(f"reset: {reset}")
    
    # Test 2: Conflict relations
    print("\n=== Test 2: Conflict Relations ===")
    print(f"SB (Sequenced Before): {ConflictRelation.SB}")
    print(f"SA (Sequenced After): {ConflictRelation.SA}")
    print(f"C (Conflict): {ConflictRelation.C}")
    print(f"CF (Conflict-Free): {ConflictRelation.CF}")
    
    # Test 3: Module builder
    print("\n=== Test 3: Module Builder ===")
    builder = ModuleBuilder("Counter")
    
    # Add state
    count_state = builder.add_state("count", i32)
    print(f"Added state: {count_state}")
    
    # Add value method
    get_value = builder.add_value_method("getValue", [], i32)
    print(f"Added value method: {get_value}")
    
    # Add action methods
    increment = builder.add_action_method("increment")
    decrement = builder.add_action_method("decrement")
    print(f"Added action methods: {increment}, {decrement}")
    
    # Set conflicts
    builder.set_conflict("increment", "decrement", ConflictRelation.C)
    
    # Print module
    print(f"\n{builder}")
    
    # Test 4: Values and operations
    print("\n=== Test 4: Values and Operations ===")
    a = Value("a", i32)
    b = Value("b", i32)
    print(f"Created values: {a}, {b}")
    
    # Binary operations
    sum_val = a + b
    diff_val = a - b
    and_val = a & b
    print(f"a + b: {sum_val}")
    print(f"a - b: {diff_val}")
    print(f"a & b: {and_val}")
    
    # Constants
    c1 = constant(1, i32)
    c42 = constant(42, i32)
    print(f"Constants: {c1}, {c42}")
    
    # Test 5: Module decorator (simple test without MLIR)
    print("\n=== Test 5: Module Decorator ===")
    
    @sharp.module("Adder")
    class Adder:
        # State variables
        result = State("result", i32)
        
        # Methods would be defined here in a full implementation
        
    print(f"Created module class with builder: {Adder._module_builder}")
    
    print("\n✅ All PySharp frontend tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)