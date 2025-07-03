#!/usr/bin/env python3
"""Test the Sharp Python bindings."""

import sys
import os

# The bindings should be installed by pixi
try:
    # Import MLIR first
    import mlir
    import mlir.ir as ir
    
    # Import Sharp bindings
    import sharp
    
    # Create a context and register Sharp dialects
    with ir.Context() as ctx:
        sharp._sharp.register_dialects(ctx._CAPIPtr)
        
        # Try to parse some Sharp dialect operations
        module = ir.Module.parse("""
        module {
            sharp.constant 42 : i32
        }
        """, ctx)
        
        print("✓ Successfully created and parsed Sharp module")
        print(module)
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)