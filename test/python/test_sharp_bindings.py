#!/usr/bin/env python3
"""Test the Sharp Python bindings."""

import sys
import os

# The paths are set by the pixi run test-construction command
# No need to manually adjust sys.path

try:
    # Import MLIR first
    import mlir
    import mlir.ir as ir
    
    # Import Sharp bindings
    import sharp
    
    print("✓ Successfully imported Sharp bindings")
    
    # Create a context and register Sharp dialects
    with ir.Context() as ctx:
        sharp._sharp.register_dialects(ctx)
        print("✓ Registered Sharp dialects")
        
        # Test Txn dialect
        txn_module = ir.Module.parse("""
        module {
            txn.module @Counter {
                txn.value_method @getValue() -> i32 {
                    %c42_i32 = arith.constant 42 : i32
                    txn.return %c42_i32 : i32
                }
            }
        }
        """, ctx)
        
        print("\n✓ Successfully parsed Txn module")
        print("\nGenerated Txn MLIR:")
        print(txn_module)
        
        # Test accessing dialects
        from sharp.dialects import txn as txn_dialect
        
        print("\n✓ Successfully imported dialect modules")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
    
print("\n✅ All tests passed!")