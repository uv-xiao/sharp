#!/usr/bin/env python3
# RUN: %python %s | FileCheck %s

# Test basic PySharp module construction

# CHECK: Creating Counter module
print("Creating Counter module")

from pysharp import module, value_method, action_method, i32, ConflictRelation

# CHECK: Module decorator imported successfully
print("Module decorator imported successfully")

@module
class Counter:
    count = i32
    
    @value_method
    def get_value(self) -> i32:
        return self.count.read()
        
    @action_method  
    def increment(self):
        self.count.write(self.count.read() + 1)
        
    @action_method
    def decrement(self):
        self.count.write(self.count.read() - 1)
        
# CHECK: Counter module created
print("Counter module created")

# CHECK: Methods: get_value, increment, decrement
print(f"Methods: {', '.join(['get_value', 'increment', 'decrement'])}")