#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""PySharp: Pythonic frontend for Sharp hardware description."""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Union, Callable
from enum import IntEnum
import inspect

# Import Sharp dialect bindings
# These will be available after building with proper paths
try:
    from . import _mlir_libs
    _sharp = _mlir_libs._sharp
    
    # Check if we can access ir module
    if hasattr(_mlir_libs, '_mlir'):
        from ._mlir_libs._mlir import ir
        from ._mlir_libs._mlir.dialects import arith, scf, index
    else:
        # Fallback - try to import from the build output structure
        from .sharp import ir
        from .sharp.dialects import arith, scf, index
    
    # Import Sharp dialects
    from .dialects import txn
    
    # Try to import CIRCT dialects
    try:
        from .sharp.dialects import firrtl, comb, seq, sv
    except ImportError:
        # CIRCT dialects not available
        firrtl = comb = seq = sv = None
        
    _BINDINGS_AVAILABLE = True
except ImportError:
    _BINDINGS_AVAILABLE = False
    ir = arith = scf = index = txn = None
    firrtl = comb = seq = sv = None
    _sharp = None


class ConflictRelation(IntEnum):
    """Conflict relation between actions in a module."""
    SB = 0  # Sequenced Before
    SA = 1  # Sequenced After  
    C = 2   # Conflict
    CF = 3  # Conflict-Free


# Type definitions
class Type:
    """Base class for hardware types."""
    def __init__(self):
        self._mlir_type = None
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
    @property
    def mlir_type(self):
        """Get the MLIR type representation."""
        return self._mlir_type


class IntType(Type):
    """Integer type with specified bit width."""
    def __init__(self, width: int):
        super().__init__()
        self.width = width
        if _BINDINGS_AVAILABLE and ir:
            self._mlir_type = ir.IntegerType.get_signless(width)
        
    def __repr__(self):
        return f"i{self.width}"


# Predefined types
i1 = IntType(1)
i8 = IntType(8)
i16 = IntType(16)  
i32 = IntType(32)
i64 = IntType(64)
i128 = IntType(128)
i256 = IntType(256)


class UIntType(Type):
    """FIRRTL unsigned integer type."""
    def __init__(self, width: int):
        super().__init__()
        self.width = width
        if _BINDINGS_AVAILABLE and firrtl:
            self._mlir_type = firrtl.UIntType.get(width)
        
    def __repr__(self):
        return f"uint<{self.width}>"


class SIntType(Type):
    """FIRRTL signed integer type."""
    def __init__(self, width: int):
        super().__init__()
        self.width = width
        if _BINDINGS_AVAILABLE and firrtl:
            self._mlir_type = firrtl.SIntType.get(width)
        
    def __repr__(self):
        return f"sint<{self.width}>"


def uint(width: int) -> UIntType:
    return UIntType(width)


def sint(width: int) -> SIntType:
    return SIntType(width)


# Signal/Value classes
class Signal:
    """Represents a signal/value in the hardware design."""
    
    def __init__(self, name: str, type: Type, mlir_value=None):
        self.name = name
        self.type = type
        self._mlir_value = mlir_value
        
    def __repr__(self):
        return f"Signal({self.name}: {self.type})"
    
    # Arithmetic operations
    def __add__(self, other):
        if isinstance(other, (int, Signal)):
            return BinaryOp("+", self, other, self.type)
        return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, (int, Signal)):
            return BinaryOp("-", self, other, self.type)
        return NotImplemented
    
    def __and__(self, other):
        if isinstance(other, (int, Signal)):
            return BinaryOp("&", self, other, self.type)
        return NotImplemented
    
    def __or__(self, other):
        if isinstance(other, (int, Signal)):
            return BinaryOp("|", self, other, self.type)
        return NotImplemented


class BinaryOp(Signal):
    """Binary operation on signals."""
    def __init__(self, op: str, lhs: Signal, rhs: Union[Signal, int], result_type: Type):
        super().__init__(f"({lhs.name} {op} {self._rhs_name(rhs)})", result_type)
        self.op = op
        self.lhs = lhs
        self.rhs = rhs
        
    def _rhs_name(self, rhs):
        if isinstance(rhs, Signal):
            return rhs.name
        return str(rhs)


class Constant(Signal):
    """Constant signal."""
    def __init__(self, value: Union[int, bool], type: Type):
        super().__init__(f"const_{value}", type)
        self.value = value
        
        if _BINDINGS_AVAILABLE and ir and arith:
            with ir.InsertionPoint.current:
                const_op = arith.ConstantOp(type.mlir_type, value)
                self._mlir_value = const_op.result


# Module components
class State:
    """State variable in a module."""
    def __init__(self, name: str, type: Type, initial_value=None):
        self.name = name
        self.type = type
        self.initial_value = initial_value
        self._state_op = None
        
    def __repr__(self):
        return f"State({self.name}: {self.type})"
    
    def read(self) -> Signal:
        """Read the state value."""
        return Signal(f"read_{self.name}", self.type)
        
    def write(self, value: Signal):
        """Write to the state."""
        return ("write", self, value)


class Port:
    """Module port."""
    def __init__(self, name: str, type: Type, direction: str):
        self.name = name
        self.type = type
        self.direction = direction
        
    def __repr__(self):
        return f"Port({self.name}: {self.type}, {self.direction})"


class Method:
    """Base class for methods."""
    def __init__(self, name: str, func: Optional[Callable] = None):
        self.name = name
        self.func = func
        self.params = []
        self.returns = None
        self._method_op = None
        
        # Extract signature if function provided
        if func:
            sig = inspect.signature(func)
            for param_name, param in sig.parameters.items():
                if param_name != 'self' and param.annotation != inspect.Parameter.empty:
                    self.params.append((param_name, param.annotation))
            if sig.return_annotation != inspect.Signature.empty:
                self.returns = sig.return_annotation


class ValueMethod(Method):
    """Value method - combinational, no side effects."""
    pass


class ActionMethod(Method):
    """Action method - can modify state."""  
    pass


class Rule:
    """Rule in a module."""
    def __init__(self, name: str, func: Optional[Callable] = None):
        self.name = name
        self.func = func
        self._rule_op = None


# Module builder
class ModuleBuilder:
    """Builder for creating hardware modules."""
    
    def __init__(self, name: str):
        self.name = name
        self.states: List[State] = []
        self.methods: List[Method] = []
        self.rules: List[Rule] = []
        self.conflict_matrix: Dict[tuple, ConflictRelation] = {}
        self._mlir_module = None
        self._txn_module = None
        
    def add_state(self, name: str, type: Type, initial_value=None) -> State:
        """Add a state variable."""
        state = State(name, type, initial_value)
        self.states.append(state)
        return state
        
    def add_value_method(self, name: str, func: Optional[Callable] = None) -> ValueMethod:
        """Add a value method."""
        method = ValueMethod(name, func)
        self.methods.append(method)
        return method
        
    def add_action_method(self, name: str, func: Optional[Callable] = None) -> ActionMethod:
        """Add an action method."""
        method = ActionMethod(name, func)
        self.methods.append(method)
        return method
        
    def add_rule(self, name: str, func: Optional[Callable] = None) -> Rule:
        """Add a rule."""
        rule = Rule(name, func)
        self.rules.append(rule)
        return rule
        
    def set_conflict(self, action1: str, action2: str, relation: ConflictRelation):
        """Set conflict relation between two actions."""
        self.conflict_matrix[(action1, action2)] = relation
        
    def build(self):
        """Build the MLIR representation."""
        if not _BINDINGS_AVAILABLE:
            raise RuntimeError("MLIR bindings not available")
            
        with ir.Context() as ctx:
            # Register dialects
            if _sharp:
                _sharp.register_dialects(ctx)
                
            self._mlir_module = ir.Module.create()
            
            with ir.InsertionPoint(self._mlir_module.body):
                # Create txn.module
                self._txn_module = txn.ModuleOp(name=self.name)
                
                # Add states, methods, rules
                # This would be implemented with proper MLIR construction
                
        return self._mlir_module
        
    def __repr__(self):
        parts = [f"ModuleBuilder({self.name})"]
        if self.states:
            parts.append(f"  States: {[s.name for s in self.states]}")
        if self.methods:
            parts.append(f"  Methods: {[m.name for m in self.methods]}")  
        if self.rules:
            parts.append(f"  Rules: {[r.name for r in self.rules]}")
        return "\n".join(parts)


# Module class and decorator
class Module:
    """Base class for hardware modules."""
    
    def __init__(self, name: Optional[str] = None):
        self._name = name or self.__class__.__name__
        self._builder = ModuleBuilder(self._name)
        self._process_attributes()
        
    def _process_attributes(self):
        """Process class attributes to build the module."""
        for name, value in self.__class__.__dict__.items():
            if isinstance(value, State):
                # Clone state for this instance
                state = State(value.name, value.type, value.initial_value)
                setattr(self, name, state)
                self._builder.add_state(state.name, state.type, state.initial_value)
                
    def set_conflict(self, action1: str, action2: str, relation: ConflictRelation):
        """Set conflict relation between actions."""
        self._builder.set_conflict(action1, action2, relation)
        
    def build(self):
        """Build the MLIR module."""
        return self._builder.build()


def module(name: Optional[str] = None):
    """Decorator to create a hardware module from a class."""
    def decorator(cls):
        # Process methods and rules
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value) and hasattr(attr_value, '_pysharp_method_type'):
                # This is a decorated method
                pass
                
        # Add module name
        if name:
            cls._module_name = name
        else:
            cls._module_name = cls.__name__
            
        return cls
    return decorator


def value_method(func):
    """Decorator for value methods."""
    func._pysharp_method_type = 'value'
    return func


def action_method(func):
    """Decorator for action methods."""
    func._pysharp_method_type = 'action'
    return func


def rule(func):
    """Decorator for rules."""
    func._pysharp_method_type = 'rule'
    return func


# Convenience functions
def constant(value: Union[int, bool], type: Type) -> Constant:
    """Create a constant signal."""
    return Constant(value, type)