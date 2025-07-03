#!/usr/bin/env python3
"""Simple test for the Pythonic construction API that works without MLIR bindings."""

import sys
import os

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python_packages/sharp_core'))

# Test that we can at least import the construction module
try:
    # Import just the types and enums which don't depend on MLIR
    # Use edsl_types to avoid the runtime issue with nanobind
    from sharp.edsl_types import (
        HWType, IntType, BoolType, ConflictRelation,
        i1, i8, i16, i32, i64, i128, i256,
        uint, sint, clock, reset
    )
    
    print("✓ Successfully imported Sharp edsl types")
    
    # Test type creation
    custom_type = IntType(width=24)
    print(f"✓ Created custom 24-bit type: {custom_type}")
    
    # Test predefined types
    print(f"✓ i32 type: {i32}")
    print(f"✓ i1 (bool) type: {i1}")
    
    # Test conflict relations
    print(f"✓ ConflictRelation.C = {ConflictRelation.C}")
    print(f"✓ ConflictRelation.SB = {ConflictRelation.SB}")
    
    # Test FIRRTL types
    print("✓ FIRRTL types available:")
    print(f"  - uint<8>: {uint(8)}")
    print(f"  - sint<16>: {sint(16)}")
    print(f"  - clock: {clock}")
    print(f"  - reset: {reset}")
    
    print("\n✅ Basic edsl API import test passed!")
    print("\nNote: Full MLIR integration requires fixing the Python bindings runtime issue.")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the Sharp Python bindings are built and in PYTHONPATH")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)