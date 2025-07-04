# PySharp type system following PyCDE patterns

from .sharp import ir
from .sharp.dialects import builtin
from typing import Union, List, Optional
import re

class Type:
    """Base class for PySharp types."""
    
    def __init__(self, mlir_type: ir.Type):
        self._type = mlir_type
    
    @property
    def mlir_type(self) -> ir.Type:
        return self._type
    
    def __str__(self):
        return str(self._type)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self._type})"
    
    def __eq__(self, other):
        if not isinstance(other, Type):
            return False
        return self._type == other._type

class IntType(Type):
    """Integer type with specified width."""
    
    def __init__(self, width: int, signed: bool = True):
        self.width = width
        self.signed = signed
        if signed:
            mlir_type = ir.IntegerType.get_signless(width)
        else:
            mlir_type = ir.IntegerType.get_signless(width)
        super().__init__(mlir_type)
    
    @classmethod
    def from_mlir(cls, mlir_type: ir.Type):
        """Create IntType from MLIR type."""
        if not isinstance(mlir_type, ir.IntegerType):
            raise TypeError(f"Expected IntegerType, got {type(mlir_type)}")
        return cls(mlir_type.width, True)

class UIntType(IntType):
    """Unsigned integer type."""
    
    def __init__(self, width: int):
        super().__init__(width, signed=False)

class SIntType(IntType):
    """Signed integer type."""
    
    def __init__(self, width: int):
        super().__init__(width, signed=True)

class ClockType(Type):
    """Clock type for sequential logic."""
    
    def __init__(self):
        # Use i1 as a placeholder for clock
        super().__init__(ir.IntegerType.get_signless(1))

class ResetType(Type):
    """Reset type for sequential logic."""
    
    def __init__(self):
        # Use i1 for reset
        super().__init__(ir.IntegerType.get_signless(1))

class ArrayType(Type):
    """Array type with element type and size."""
    
    def __init__(self, element_type: Type, size: int):
        self.element_type = element_type
        self.size = size
        # Create MLIR array type
        mlir_type = ir.RankedTensorType.get([size], element_type.mlir_type)
        super().__init__(mlir_type)

class StructType(Type):
    """Struct type with named fields."""
    
    def __init__(self, fields: dict[str, Type]):
        self.fields = fields
        # For now, use a placeholder - would need custom type in practice
        super().__init__(ir.IntegerType.get_signless(32))

# Type utilities
class TypeRegistry:
    """Registry for type construction utilities."""
    
    @staticmethod
    def int_type(width: int, signed: bool = True) -> IntType:
        """Create an integer type."""
        return IntType(width, signed)
    
    @staticmethod
    def uint(width: int) -> UIntType:
        """Create an unsigned integer type."""
        return UIntType(width)
    
    @staticmethod
    def sint(width: int) -> SIntType:
        """Create a signed integer type."""
        return SIntType(width)
    
    @staticmethod
    def clock() -> ClockType:
        """Create a clock type."""
        return ClockType()
    
    @staticmethod
    def reset() -> ResetType:
        """Create a reset type."""
        return ResetType()
    
    @staticmethod
    def array(element_type: Type, size: int) -> ArrayType:
        """Create an array type."""
        return ArrayType(element_type, size)
    
    @staticmethod
    def struct(**fields: Type) -> StructType:
        """Create a struct type."""
        return StructType(fields)

# Export convenient type constructors
types = TypeRegistry()

# Predefined common types
i1 = IntType(1)
i8 = IntType(8)
i16 = IntType(16)
i32 = IntType(32)
i64 = IntType(64)
i128 = IntType(128)
i256 = IntType(256)

# Helper functions for FIRRTL-style types
def uint(width: int) -> UIntType:
    """Create FIRRTL-style uint type."""
    return UIntType(width)

def sint(width: int) -> SIntType:
    """Create FIRRTL-style sint type."""
    return SIntType(width)

# Type parsing utilities
def parse_type(type_str: str) -> Type:
    """Parse a type string into a Type object."""
    # Handle basic integer types
    if m := re.match(r'i(\d+)', type_str):
        return IntType(int(m.group(1)))
    elif m := re.match(r'uint<(\d+)>', type_str):
        return UIntType(int(m.group(1)))
    elif m := re.match(r'sint<(\d+)>', type_str):
        return SIntType(int(m.group(1)))
    else:
        raise ValueError(f"Unknown type format: {type_str}")

__all__ = [
    'Type', 'IntType', 'UIntType', 'SIntType', 'ClockType', 'ResetType',
    'ArrayType', 'StructType', 'types', 'i1', 'i8', 'i16', 'i32', 'i64',
    'i128', 'i256', 'uint', 'sint', 'parse_type'
]