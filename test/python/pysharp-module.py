# RUN: %python %s

# Test PySharp module creation
import pysharp
from pysharp import module, value_method, action_method, rule
from pysharp import i32, i8

print("✓ Imported PySharp decorators")

# Define a simple module
@module
class Counter:
    count = i32
    enable = i8
    
    @value_method
    def read(self) -> i32:
        return self.count.read()
    
    @action_method  
    def increment(self):
        self.count.write(self.count.read() + 1)
    
    @rule()
    def auto_inc(self):
        # Rules don't have bodies in the current implementation
        pass

print("✓ Successfully defined Counter module")

# Check that the class was processed by the decorator
# The module decorator in the current implementation just returns the class
print(f"✓ Counter class type: {type(Counter).__name__}")

# Check methods
assert hasattr(Counter, 'read')
assert hasattr(Counter, 'increment')
assert hasattr(Counter, 'auto_inc')
print("✓ All methods defined")

print("✅ PySharp module definition test passed!")