#!/usr/bin/env python3
"""Test that Sharp Python bindings can access MLIR and CIRCT dialects."""

import sys
import os

try:
    # Import MLIR first
    import mlir
    import mlir.ir as ir
    from mlir.dialects import arith, scf, index
    
    # Import CIRCT
    import circt
    from circt.dialects import hw, comb, seq, sv, firrtl
    
    # Import Sharp bindings
    import sharp
    
    print("✓ Successfully imported MLIR, CIRCT, and Sharp bindings")
    
    # Create a context 
    with ir.Context() as ctx:
        # Register dialects via Sharp
        sharp._sharp.register_dialects(ctx)
        print("✓ Registered dialects via Sharp")
        
        # Test MLIR dialects
        mlir_module = ir.Module.parse("""
        module {
            func.func @test_mlir(%arg0: i32, %arg1: i32) -> i32 {
                %0 = arith.addi %arg0, %arg1 : i32
                scf.if %0 {
                    %c1 = arith.constant 1 : i32
                    scf.yield
                }
                return %0 : i32
            }
        }
        """, ctx)
        
        print("\n✓ Successfully parsed MLIR module with arith and scf dialects")
        print(mlir_module)
        
        # Test CIRCT dialects
        circt_module = ir.Module.parse("""
        module {
            hw.module @test_hw(%a: i8, %b: i8) -> (out: i8) {
                %0 = comb.add %a, %b : i8
                hw.output %0 : i8
            }
        }
        """, ctx)
        
        print("\n✓ Successfully parsed CIRCT HW module")
        print(circt_module)
        
        # Test Txn dialect
        txn_module = ir.Module.parse("""
        module {
            txn.module @Counter {
                %0 = txn.state @count : i32
                
                txn.value_method @getValue() -> i32 {
                    %v = txn.read %0 : !txn.state<i32>
                    txn.return %v : i32
                }
                
                txn.action_method @increment() {
                    %v = txn.read %0 : !txn.state<i32>
                    %c1 = arith.constant 1 : i32
                    %new = arith.addi %v, %c1 : i32
                    txn.write %0, %new : !txn.state<i32>, i32
                    txn.return
                }
            }
        }
        """, ctx)
        
        print("\n✓ Successfully parsed Txn module using MLIR arith dialect")
        print(txn_module)
        
        # Test that we can use dialect constructors
        from sharp.dialects import txn as txn_dialect
        print("\n✓ Successfully imported txn dialect module")
        
        # Test FIRRTL access
        firrtl_module = ir.Module.parse("""
        firrtl.circuit "Test" {
            firrtl.module @Test(in %a: !firrtl.uint<8>, in %b: !firrtl.uint<8>, out %c: !firrtl.uint<8>) {
                %0 = firrtl.add %a, %b : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
                firrtl.strictconnect %c, %0 : !firrtl.uint<8>
            }
        }
        """, ctx)
        
        print("\n✓ Successfully parsed FIRRTL module")
        print(firrtl_module)
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
    
print("\n✅ All tests passed! Sharp can access MLIR and CIRCT dialects.")