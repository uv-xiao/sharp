#!/usr/bin/env python3
"""Test PySharp functionality with MLIR and CIRCT integration."""

import sys
import os

# Set up Python paths for Sharp, MLIR, and CIRCT
sys.path.insert(0, "/home/uvxiao/sharp/build/python_packages")
sys.path.insert(0, "/home/uvxiao/sharp/.install/unified-build/tools/mlir/python_packages/mlir_core")
sys.path.insert(0, "/home/uvxiao/sharp/.install/unified-build/tools/circt/python_packages/circt_core")

try:
    # Import MLIR
    import mlir.ir as ir
    from mlir.dialects import arith, scf
    print("✓ Successfully imported MLIR")
    
    # Import CIRCT 
    from circt.dialects import comb, seq, sv, firrtl
    print("✓ Successfully imported CIRCT dialects")
    
    # Import Sharp
    import sharp
    from sharp.dialects import txn
    print("✓ Successfully imported Sharp")
    
    # Create a context and register all dialects
    with ir.Context() as ctx:
        # Register Sharp dialects (this will also register MLIR/CIRCT dialects we need)
        sharp._sharp.register_dialects(ctx)
        print("✓ Registered all dialects")
        
        # Test 1: Txn module with MLIR arithmetic
        print("\n=== Test 1: Txn module using MLIR arith dialect ===")
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
        print("✓ Successfully created Txn module with MLIR arith operations")
        print(txn_module)
        
        # Test 2: Txn module with CIRCT combinational logic
        print("\n=== Test 2: Txn module with CIRCT comb operations ===")
        txn_comb_module = ir.Module.parse("""
        module {
            txn.module @LogicUnit {
                txn.value_method @computeAnd(%a: i8, %b: i8) -> i8 {
                    %result = comb.and %a, %b : i8
                    txn.return %result : i8
                }
                
                txn.value_method @computeOr(%a: i8, %b: i8) -> i8 {
                    %result = comb.or %a, %b : i8
                    txn.return %result : i8
                }
            }
        }
        """, ctx)
        print("✓ Successfully created Txn module with CIRCT comb operations")
        print(txn_comb_module)
        
        # Test 3: FIRRTL module
        print("\n=== Test 3: FIRRTL module ===")
        firrtl_module = ir.Module.parse("""
        firrtl.circuit "Adder" {
            firrtl.module @Adder(in %a: !firrtl.uint<8>, in %b: !firrtl.uint<8>, out %sum: !firrtl.uint<8>) {
                %0 = firrtl.add %a, %b : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
                firrtl.strictconnect %sum, %0 : !firrtl.uint<8>
            }
        }
        """, ctx)
        print("✓ Successfully created FIRRTL module")
        print(firrtl_module)
        
        # Test 4: SCF control flow
        print("\n=== Test 4: SCF control flow in Txn ===")
        scf_module = ir.Module.parse("""
        module {
            txn.module @ControlFlow {
                txn.value_method @conditionalAdd(%cond: i1, %a: i32, %b: i32) -> i32 {
                    %result = scf.if %cond -> i32 {
                        %sum = arith.addi %a, %b : i32
                        scf.yield %sum : i32
                    } else {
                        scf.yield %a : i32
                    }
                    txn.return %result : i32
                }
            }
        }
        """, ctx)
        print("✓ Successfully created Txn module with SCF control flow")
        print(scf_module)
        
        print("\n✅ All tests passed! PySharp successfully integrates MLIR and CIRCT dialects.")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)