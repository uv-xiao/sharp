#!/usr/bin/env python3
"""Test the Pythonic construction API for Sharp modules."""

import sys
import os

# Add the build directory to the path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/lib/Bindings/Python'))

try:
    # Import MLIR first from the environment
    import mlir
    import mlir.ir as ir
    import mlir.dialects
    
    # Then import Sharp components
    import sharp
    from sharp.edsl import (
        module, ModuleBuilder, i8, i16, i32, i64, 
        ConflictRelation, Value
    )
    from sharp.dialects import sharp as sharp_dialect
    
    # Register Sharp dialects
    with ir.Context() as ctx:
        sharp.register_sharp_dialects(ctx)
        
except ImportError as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()
    print("\nMake sure the Sharp Python bindings are built and PYTHONPATH includes:")
    print("  - MLIR Python packages")
    print("  - Sharp Python packages")
    sys.exit(1)


def test_simple_module():
    """Test creating a simple module."""
    @module
    def SimpleCounter():
        builder = ModuleBuilder("SimpleCounter")
        
        @builder.value_method(return_type=i32)
        def getValue(b):
            return b.constant(42)
            
        @builder.action_method()
        def reset(b):
            # Empty action
            pass
            
        return builder
        
    # Build the module
    mlir_module = SimpleCounter.build()
    print("Simple module:")
    print(mlir_module)
    assert "txn.module @SimpleCounter" in str(mlir_module)
    assert "txn.value_method @getValue" in str(mlir_module)
    assert "txn.action_method @reset" in str(mlir_module)
    

def test_arithmetic_operations():
    """Test arithmetic operations in methods."""
    @module
    def ArithModule():
        builder = ModuleBuilder("ArithModule")
        
        @builder.value_method(return_type=i32)
        def compute(b, a: i32, b_arg: i32):
            # Test various arithmetic operations
            sum_val = a + b_arg
            doubled = sum_val * 2
            shifted = doubled << 1
            return shifted
            
        @builder.value_method(return_type=i32) 
        def computeWithConstants(b, x: i32):
            # Mix of constants and arguments
            c10 = b.constant(10)
            c5 = b.constant(5)
            result = (x + c10) * c5
            return result
            
        return builder
        
    mlir_module = ArithModule.build()
    print("\nArithmetic module:")
    print(mlir_module)
    assert "arith.addi" in str(mlir_module)
    assert "arith.muli" in str(mlir_module)
    assert "arith.shli" in str(mlir_module)
    

def test_different_types():
    """Test different integer types."""
    @module  
    def TypesModule():
        builder = ModuleBuilder("TypesModule")
        
        @builder.value_method(return_type=i8)
        def getByte(b):
            return b.constant(255, i8)
            
        @builder.value_method(return_type=i16)
        def getShort(b):
            return b.constant(1000, i16)
            
        @builder.value_method(return_type=i64)
        def getLong(b):
            return b.constant(42, i64)
            
        return builder
        
    mlir_module = TypesModule.build()
    print("\nTypes module:")
    print(mlir_module)
    assert "i8" in str(mlir_module)
    assert "i16" in str(mlir_module) 
    assert "i64" in str(mlir_module)
    

def test_rules_and_conflicts():
    """Test rules and conflict matrix."""
    @module
    def ConflictModule():
        builder = ModuleBuilder("ConflictModule")
        
        @builder.rule
        def rule1(b):
            c1 = b.constant(1)
            # Rules typically have side effects
            pass
            
        @builder.rule  
        def rule2(b):
            c2 = b.constant(2)
            pass
            
        @builder.action_method()
        def action(b):
            pass
            
        # Add conflicts
        builder.add_conflict("rule1", "rule2", ConflictRelation.C)
        builder.add_conflict("rule1", "action", ConflictRelation.SB)
        
        return builder
        
    mlir_module = ConflictModule.build()
    print("\nConflict module:")
    print(mlir_module)
    assert "txn.rule @rule1" in str(mlir_module)
    assert "txn.rule @rule2" in str(mlir_module)
    assert "conflict_matrix" in str(mlir_module)
    

def test_comparison_operations():
    """Test comparison and select operations."""
    @module
    def CompareModule():
        builder = ModuleBuilder("CompareModule")
        
        @builder.value_method(return_type=i32)
        def max_value(b, a: i32, b_arg: i32):
            # Compare and select max
            is_greater = a > b_arg
            result = b.select(is_greater, a, b_arg)
            return result
            
        @builder.value_method(return_type=i32)
        def check_equal(b, x: i32, y: i32):
            is_equal = x == y
            # Convert bool to i32
            result = b.select(is_equal, b.constant(1), b.constant(0))
            return result
            
        return builder
        
    mlir_module = CompareModule.build()
    print("\nCompare module:")
    print(mlir_module)
    assert "arith.cmpi" in str(mlir_module)
    assert "arith.select" in str(mlir_module)
    

def test_action_with_return():
    """Test action methods with return values."""
    @module
    def ActionReturnModule():
        builder = ModuleBuilder("ActionReturnModule")
        
        @builder.action_method(return_type=i32)
        def process(b, data: i32):
            # Process and return modified value
            result = data | 1  
            return result
            
        @builder.action_method(return_type=i32)
        def increment(b, value: i32):
            return value + 1
            
        return builder
        
    mlir_module = ActionReturnModule.build()
    print("\nAction return module:")
    print(mlir_module)
    assert "txn.action_method @process" in str(mlir_module)
    assert "txn.return" in str(mlir_module)
    

def test_complex_expressions():
    """Test complex nested expressions."""
    @module
    def ComplexModule():
        builder = ModuleBuilder("ComplexModule")
        
        @builder.value_method(return_type=i32)
        def complex_calc(b, x: i32, y: i32, z: i32):
            # Test operator precedence and nesting
            # result = (x + y) * z - (x & y) ^ z
            sum_xy = x + y
            prod = sum_xy * z
            and_xy = x & y
            xor_val = and_xy ^ z
            result = prod - xor_val
            return result
            
        return builder
        
    mlir_module = ComplexModule.build()
    print("\nComplex module:")
    print(mlir_module)
    # Check that all operations are present
    assert "arith.addi" in str(mlir_module)
    assert "arith.muli" in str(mlir_module)
    assert "arith.andi" in str(mlir_module)
    assert "arith.xori" in str(mlir_module)
    assert "arith.subi" in str(mlir_module)


def run_all_tests():
    """Run all tests."""
    tests = [
        test_simple_module,
        test_arithmetic_operations,
        test_different_types,
        test_rules_and_conflicts,
        test_comparison_operations,
        test_action_with_return,
        test_complex_expressions,
    ]
    
    for test in tests:
        print(f"\n{'='*60}")
        print(f"Running {test.__name__}...")
        print(f"{'='*60}")
        try:
            test()
            print(f"✓ {test.__name__} passed")
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    print("\n✓ All tests passed!")
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)