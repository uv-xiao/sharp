# RUN: %python %s

# Test basic Sharp Python binding imports
import sharp
from sharp import ir

print("✓ Successfully imported Sharp bindings")

# Create a context
ctx = ir.Context()
print("✓ Created MLIR context")

# Register Sharp dialects
sharp.register_sharp_dialects(ctx)
print("✓ Registered Sharp dialects")

# Test that we can access the registered dialects
with ctx:
    # Parse a simple module with txn dialect
    module = ir.Module.parse("""
    txn.module @TestModule {
        txn.value_method @getValue() -> i32 {
            %c42_i32 = arith.constant 42 : i32
            txn.return %c42_i32 : i32
        }
        txn.schedule []
    }
    """)
    print("✓ Successfully parsed txn module")
    
print("✅ All tests passed!")