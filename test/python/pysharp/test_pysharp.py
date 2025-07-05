#!/usr/bin/env python3
# RUN: %python -m pytest %s -v | FileCheck %s

# PySharp unit tests

import sys
import pytest

sys.path.insert(0, '/home/uvxiao/sharp/frontends/PySharp/src')

# CHECK: test_import_types PASSED
def test_import_types():
    """Test importing PySharp type system"""
    try:
        from pysharp.types import IntType, UIntType, SIntType, BoolType
        # Types should be importable even without MLIR bindings
        assert IntType is not None
        assert BoolType is not None
    except ImportError:
        pytest.skip("PySharp types not available")

# CHECK: test_conflict_relation PASSED
def test_conflict_relation():
    """Test ConflictRelation enum"""
    try:
        from pysharp.common import ConflictRelation
        assert ConflictRelation.SB.value == 0
        assert ConflictRelation.SA.value == 1
        assert ConflictRelation.C.value == 2
        assert ConflictRelation.CF.value == 3
    except ImportError:
        pytest.skip("PySharp common module not available")

# CHECK: test_module_builder PASSED
def test_module_builder():
    """Test ModuleBuilder construction"""
    try:
        from pysharp.builder import ModuleBuilder
        # Should be able to create builder even without MLIR
        builder = ModuleBuilder("TestModule")
        assert builder.name == "TestModule"
    except ImportError:
        pytest.skip("PySharp builder not available")

if __name__ == "__main__":
    # Run tests
    print("Running PySharp unit tests...")