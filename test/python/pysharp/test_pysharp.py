#!/usr/bin/env python3
# RUN: %python %s

# PySharp unit tests

import sys

# Test importing PySharp type system
from pysharp.types import IntType
from pysharp import i1, i8, i16, i32, i64
print("✓ PySharp types imported successfully")

# Test ConflictRelation enum
from pysharp.common import ConflictRelation
assert ConflictRelation.SequenceBefore == 0
assert ConflictRelation.SequenceAfter == 1
assert ConflictRelation.Conflict == 2
assert ConflictRelation.ConflictFree == 3
print("✓ ConflictRelation enum works correctly")

# Test basic type creation
t1 = IntType(8)
assert str(t1) == "i8"
t2 = IntType(32)
assert str(t2) == "i32"
print("✓ IntType creation works")

# Test predefined types
assert str(i8) == "i8"
assert str(i32) == "i32"
assert str(i64) == "i64"
print("✓ Predefined types work correctly")

print("\n✅ All PySharp unit tests passed!")