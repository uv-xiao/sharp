# PySharp module definition following PyCDE patterns

from typing import Dict, List, Optional, Callable, Any, Type as PyType
from functools import wraps
from sharp import ir
from sharp.dialects import txn
from .types import Type, IntType
from .signals import Signal
from .common import (ConflictRelation, Timing, Combinational, 
                     PortDirection, Port)
from .builder import ModuleBuilder

class MethodDecorator:
    """Base class for method decorators."""
    
    def __init__(self, func: Callable):
        self.func = func
        self.timing = Combinational
        self.always_ready = False
        self.always_enabled = False

class ValueMethodDecorator(MethodDecorator):
    """Decorator for value methods."""
    pass

class ActionMethodDecorator(MethodDecorator):
    """Decorator for action methods."""
    pass

class RuleDecorator(MethodDecorator):
    """Decorator for rules."""
    
    def __init__(self, func: Callable, guard: Optional[Callable] = None):
        super().__init__(func)
        self.guard = guard

def value_method(timing: Timing = Combinational, 
                always_ready: bool = False):
    """Decorator for value methods."""
    def decorator(func):
        dec = ValueMethodDecorator(func)
        dec.timing = timing
        dec.always_ready = always_ready
        return dec
    return decorator

def action_method(timing: Timing = Combinational,
                 always_ready: bool = False,
                 always_enabled: bool = False):
    """Decorator for action methods."""
    def decorator(func):
        dec = ActionMethodDecorator(func)
        dec.timing = timing
        dec.always_ready = always_ready
        dec.always_enabled = always_enabled
        return dec
    return decorator

def rule(guard: Optional[Callable] = None):
    """Decorator for rules."""
    def decorator(func):
        return RuleDecorator(func, guard)
    return decorator

class ModuleMeta(type):
    """Metaclass for PySharp modules."""
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Collect methods and rules
        value_methods = {}
        action_methods = {}
        rules = {}
        
        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, ValueMethodDecorator):
                value_methods[attr_name] = attr_value
            elif isinstance(attr_value, ActionMethodDecorator):
                action_methods[attr_name] = attr_value
            elif isinstance(attr_value, RuleDecorator):
                rules[attr_name] = attr_value
        
        # Store in class
        namespace['_value_methods'] = value_methods
        namespace['_action_methods'] = action_methods
        namespace['_rules'] = rules
        
        return super().__new__(mcs, name, bases, namespace)

class Module(metaclass=ModuleMeta):
    """Base class for PySharp modules."""
    
    def __init__(self, **params):
        self.params = params
        self._builder = None
        self._built = False
        
        # Module info
        self.name = self.__class__.__name__
        self.ports: List[Port] = []
        self.conflicts: Dict[tuple, ConflictRelation] = {}
        
        # Build the module
        self._build()
    
    def _build(self):
        """Build the MLIR representation of this module."""
        if self._built:
            return
        
        # Create module builder
        self._builder = ModuleBuilder(self.name)
        
        # Add ports
        for port in self.ports:
            self._builder.add_port(port)
        
        # Add value methods
        for name, method_dec in self._value_methods.items():
            self._builder.add_value_method(
                name, method_dec.func, method_dec.timing,
                method_dec.always_ready
            )
        
        # Add action methods
        for name, method_dec in self._action_methods.items():
            self._builder.add_action_method(
                name, method_dec.func, method_dec.timing,
                method_dec.always_ready, method_dec.always_enabled
            )
        
        # Add rules
        for name, rule_dec in self._rules.items():
            self._builder.add_rule(name, rule_dec.func, rule_dec.guard)
        
        # Set conflicts
        self._builder.set_conflicts(self.conflicts)
        
        # Build the module
        self._builder.build()
        self._built = True
    
    def get_mlir_module(self) -> ir.Module:
        """Get the MLIR module."""
        if not self._built:
            self._build()
        return self._builder.get_mlir_module()
    
    def __str__(self):
        """Get string representation of the module."""
        return str(self.get_mlir_module())

# Module decorator for simpler syntax
def module(cls: PyType[Module]) -> PyType[Module]:
    """Decorator to mark a class as a PySharp module."""
    # The metaclass handles most of the work
    return cls

# Example usage pattern:
# @module
# class Counter(Module):
#     def __init__(self):
#         super().__init__()
#         self.ports = [
#             Clock(),
#             Reset(),
#             Output(i32, "count")
#         ]
#         
#     @value_method()
#     def get_count(self) -> Signal:
#         return self.count
#         
#     @action_method(timing=Static(1))
#     def increment(self):
#         self.count = self.count + 1
#         
#     @rule()
#     def auto_increment(self):
#         return True  # Always fire

__all__ = [
    'Module', 'module',
    'value_method', 'action_method', 'rule',
    'ValueMethodDecorator', 'ActionMethodDecorator', 'RuleDecorator'
]