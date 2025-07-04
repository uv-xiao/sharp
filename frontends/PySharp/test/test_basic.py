# Test basic PySharp functionality

import pysharp as ps
from pysharp import (
    Module, module, value_method, action_method, rule,
    i32, i8, Clock, Reset, Output,
    Combinational, Static,
    Conflict, ConflictFree
)

@module
class Counter(Module):
    """Simple counter module."""
    
    def __init__(self):
        super().__init__()
        self.ports = [
            Clock(),
            Reset(),
            Output(i32, "count")
        ]
        
        # Define conflicts
        self.conflicts = {
            ("increment", "decrement"): Conflict,
            ("increment", "reset_count"): Conflict,
            ("get_count", "increment"): ConflictFree
        }
    
    @value_method()
    def get_count(self) -> ps.Signal:
        """Get current count value."""
        # In real implementation, this would read from state
        return ps.Const(0, i32)
    
    @action_method(timing=Static(1))
    def increment(self):
        """Increment the counter."""
        # In real implementation: self.count = self.count + 1
        pass
    
    @action_method(timing=Static(1))
    def decrement(self):
        """Decrement the counter."""
        # In real implementation: self.count = self.count - 1
        pass
    
    @action_method()
    def reset_count(self):
        """Reset counter to zero."""
        # In real implementation: self.count = 0
        pass
    
    @rule()
    def auto_increment(self):
        """Automatically increment every cycle."""
        return True  # Always enabled

def test_counter():
    """Test counter module creation."""
    counter = Counter()
    
    # Get MLIR representation
    mlir_module = counter.get_mlir_module()
    print("Counter module MLIR:")
    print(mlir_module)
    
    # Verify it's well-formed
    assert ps.support.verify_module(mlir_module)
    print("\nModule verified successfully!")

if __name__ == "__main__":
    test_counter()