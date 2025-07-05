# RUN: %python %s

# Test PySharp imports
import pysharp
from pysharp import i32, i1, ConflictRelation

print("✓ Successfully imported PySharp")

# Check that we have the expected attributes
assert hasattr(pysharp, '__version__')
assert hasattr(pysharp, 'DefaultContext')
assert hasattr(pysharp, 'module')
assert hasattr(pysharp, 'value_method')
assert hasattr(pysharp, 'action_method')

print(f"✓ PySharp version: {pysharp.__version__}")
print(f"✓ Default context: {pysharp.DefaultContext}")

# Test conflict relations
assert ConflictRelation.SequenceBefore == 0
assert ConflictRelation.SequenceAfter == 1
assert ConflictRelation.Conflict == 2
assert ConflictRelation.ConflictFree == 3

print("✓ ConflictRelation enum works correctly")

print("✅ All PySharp import tests passed!")