"""Pythonic construction API for Sharp modules.

This module provides a high-level Pythonic API for constructing Sharp Txn modules,
similar to CIRCT's PyCDE. It allows defining hardware modules using Python decorators
and functions.
"""

from typing import Dict, List, Optional, Union, Callable, Any, Tuple
from dataclasses import dataclass
from enum import IntEnum
import functools
import inspect

try:
    from . import _sharp
    from . import ir
    from . import dialects
    from .dialects import arith
    from .dialects import sharp as sharp_dialect
except ImportError:
    # For development/testing without full build
    import _sharp
    import ir
    import dialects
    from dialects import arith
    from dialects import sharp as sharp_dialect


class ConflictRelation(IntEnum):
    """Conflict relations between actions."""
    SB = 0  # Sequenced Before
    SA = 1  # Sequenced After
    C = 2   # Conflict
    CF = 3  # Conflict if Fired


@dataclass
class HWType:
    """Base class for hardware types."""
    pass


@dataclass
class IntType(HWType):
    """Integer type with specified bit width."""
    width: int
    
    def to_mlir(self, ctx) -> ir.Type:
        """Convert to MLIR integer type."""
        return ir.IntegerType.get_signless(self.width, ctx)


@dataclass
class BoolType(HWType):
    """Boolean type (1-bit integer)."""
    
    def to_mlir(self, ctx) -> ir.Type:
        """Convert to MLIR i1 type."""
        return ir.IntegerType.get_signless(1, ctx)


# Type aliases for convenience
i1 = BoolType()
i8 = IntType(8)
i16 = IntType(16)
i32 = IntType(32)
i64 = IntType(64)
i128 = IntType(128)
i256 = IntType(256)


class MethodBuilder:
    """Builder for method bodies."""
    
    def __init__(self, method_op, entry_block):
        self.method_op = method_op
        self.entry_block = entry_block
        self.ip = ir.InsertionPoint(entry_block)
        self.values = {}  # Map from Python values to MLIR values
        
    def __enter__(self):
        self.ip.__enter__()
        return self
        
    def __exit__(self, *args):
        self.ip.__exit__(*args)
        
    def constant(self, value: Union[int, bool], type: HWType = None) -> 'Value':
        """Create a constant value."""
        if type is None:
            if isinstance(value, bool):
                type = i1
            else:
                # Default to i32 for integers
                type = i32
                
        mlir_type = type.to_mlir(self.method_op.context)
        
        if isinstance(value, bool):
            value = 1 if value else 0
            
        const_op = arith.ConstantOp(mlir_type, value)
        return Value(const_op.result, self)
        
    def add(self, lhs: 'Value', rhs: 'Value') -> 'Value':
        """Add two values."""
        add_op = arith.AddIOp(lhs.mlir_value, rhs.mlir_value)
        return Value(add_op.result, self)
        
    def sub(self, lhs: 'Value', rhs: 'Value') -> 'Value':
        """Subtract two values."""
        sub_op = arith.SubIOp(lhs.mlir_value, rhs.mlir_value)
        return Value(sub_op.result, self)
        
    def mul(self, lhs: 'Value', rhs: 'Value') -> 'Value':
        """Multiply two values."""
        mul_op = arith.MulIOp(lhs.mlir_value, rhs.mlir_value)
        return Value(mul_op.result, self)
        
    def and_(self, lhs: 'Value', rhs: 'Value') -> 'Value':
        """Bitwise AND two values."""
        and_op = arith.AndIOp(lhs.mlir_value, rhs.mlir_value)
        return Value(and_op.result, self)
        
    def or_(self, lhs: 'Value', rhs: 'Value') -> 'Value':
        """Bitwise OR two values."""
        or_op = arith.OrIOp(lhs.mlir_value, rhs.mlir_value)
        return Value(or_op.result, self)
        
    def xor(self, lhs: 'Value', rhs: 'Value') -> 'Value':
        """Bitwise XOR two values."""
        xor_op = arith.XOrIOp(lhs.mlir_value, rhs.mlir_value)
        return Value(xor_op.result, self)
        
    def shl(self, value: 'Value', amount: 'Value') -> 'Value':
        """Shift left."""
        shl_op = arith.ShLIOp(value.mlir_value, amount.mlir_value)
        return Value(shl_op.result, self)
        
    def shr(self, value: 'Value', amount: 'Value') -> 'Value':
        """Logical shift right."""
        shr_op = arith.ShRUIOp(value.mlir_value, amount.mlir_value)
        return Value(shr_op.result, self)
        
    def select(self, cond: 'Value', true_val: 'Value', false_val: 'Value') -> 'Value':
        """Select between two values based on condition."""
        select_op = arith.SelectOp(cond.mlir_value, true_val.mlir_value, false_val.mlir_value)
        return Value(select_op.result, self)
        
    def cmp_eq(self, lhs: 'Value', rhs: 'Value') -> 'Value':
        """Compare equal."""
        cmp_op = arith.CmpIOp(arith.CmpIPredicate.eq, lhs.mlir_value, rhs.mlir_value)
        return Value(cmp_op.result, self)
        
    def cmp_ne(self, lhs: 'Value', rhs: 'Value') -> 'Value':
        """Compare not equal."""
        cmp_op = arith.CmpIOp(arith.CmpIPredicate.ne, lhs.mlir_value, rhs.mlir_value)
        return Value(cmp_op.result, self)
        
    def cmp_lt(self, lhs: 'Value', rhs: 'Value') -> 'Value':
        """Compare less than (signed)."""
        cmp_op = arith.CmpIOp(arith.CmpIPredicate.slt, lhs.mlir_value, rhs.mlir_value)
        return Value(cmp_op.result, self)
        
    def cmp_gt(self, lhs: 'Value', rhs: 'Value') -> 'Value':
        """Compare greater than (signed)."""
        cmp_op = arith.CmpIOp(arith.CmpIPredicate.sgt, lhs.mlir_value, rhs.mlir_value)
        return Value(cmp_op.result, self)
        
    def return_(self, value: Optional['Value'] = None):
        """Return from the method."""
        if value is None:
            sharp_dialect.ReturnOp([])
        else:
            sharp_dialect.ReturnOp([value.mlir_value])


class Value:
    """Wrapper for MLIR values with operator overloading."""
    
    def __init__(self, mlir_value, builder: MethodBuilder):
        self.mlir_value = mlir_value
        self.builder = builder
        
    def __add__(self, other):
        if isinstance(other, int):
            other = self.builder.constant(other)
        return self.builder.add(self, other)
        
    def __sub__(self, other):
        if isinstance(other, int):
            other = self.builder.constant(other)
        return self.builder.sub(self, other)
        
    def __mul__(self, other):
        if isinstance(other, int):
            other = self.builder.constant(other)
        return self.builder.mul(self, other)
        
    def __and__(self, other):
        if isinstance(other, (int, bool)):
            other = self.builder.constant(other)
        return self.builder.and_(self, other)
        
    def __or__(self, other):
        if isinstance(other, (int, bool)):
            other = self.builder.constant(other)
        return self.builder.or_(self, other)
        
    def __xor__(self, other):
        if isinstance(other, (int, bool)):
            other = self.builder.constant(other)
        return self.builder.xor(self, other)
        
    def __lshift__(self, other):
        if isinstance(other, int):
            other = self.builder.constant(other)
        return self.builder.shl(self, other)
        
    def __rshift__(self, other):
        if isinstance(other, int):
            other = self.builder.constant(other)
        return self.builder.shr(self, other)
        
    def __eq__(self, other):
        if isinstance(other, int):
            other = self.builder.constant(other)
        return self.builder.cmp_eq(self, other)
        
    def __ne__(self, other):
        if isinstance(other, int):
            other = self.builder.constant(other)
        return self.builder.cmp_ne(self, other)
        
    def __lt__(self, other):
        if isinstance(other, int):
            other = self.builder.constant(other)
        return self.builder.cmp_lt(self, other)
        
    def __gt__(self, other):
        if isinstance(other, int):
            other = self.builder.constant(other)
        return self.builder.cmp_gt(self, other)


class ModuleBuilder:
    """Builder for Sharp modules."""
    
    def __init__(self, name: str, context: Optional[ir.Context] = None):
        self.name = name
        self.context = context or ir.Context()
        self.methods = []
        self.rules = []
        self.instances = []
        self.conflict_matrix = {}
        self.module_op = None
        
    def value_method(self, func: Callable = None, *, return_type: HWType = i32):
        """Decorator for value methods."""
        def decorator(f):
            # Get parameter types from function signature
            sig = inspect.signature(f)
            params = []
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                param_type = param.annotation if param.annotation != inspect.Parameter.empty else i32
                params.append((param_name, param_type))
            
            self.methods.append({
                'name': f.__name__,
                'type': 'value',
                'func': f,
                'params': params,
                'return_type': return_type
            })
            return f
            
        if func is None:
            return decorator
        return decorator(func)
        
    def action_method(self, func: Callable = None, *, return_type: Optional[HWType] = None):
        """Decorator for action methods."""
        def decorator(f):
            # Get parameter types from function signature
            sig = inspect.signature(f)
            params = []
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                param_type = param.annotation if param.annotation != inspect.Parameter.empty else i32
                params.append((param_name, param_type))
                
            self.methods.append({
                'name': f.__name__,
                'type': 'action',
                'func': f,
                'params': params,
                'return_type': return_type
            })
            return f
            
        if func is None:
            return decorator
        return decorator(func)
        
    def rule(self, func: Callable):
        """Decorator for rules."""
        self.rules.append({
            'name': func.__name__,
            'func': func
        })
        return func
        
    def add_conflict(self, action1: str, action2: str, relation: ConflictRelation):
        """Add a conflict relationship between two actions."""
        key = f"{action1},{action2}"
        self.conflict_matrix[key] = relation
        
    def build(self) -> ir.Module:
        """Build the MLIR module."""
        with self.context:
            # Load dialects
            self.context.allow_unregistered_dialects = True
            dialects.register_dialects(self.context)
            
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                # Create the txn.module
                self.module_op = sharp_dialect.ModuleOp(self.name)
                
                # Build methods and rules inside the module
                with ir.InsertionPoint(self.module_op.body):
                    # Build value methods
                    for method_info in self.methods:
                        if method_info['type'] == 'value':
                            self._build_value_method(method_info)
                        else:
                            self._build_action_method(method_info)
                            
                    # Build rules
                    for rule_info in self.rules:
                        self._build_rule(rule_info)
                        
                    # Build schedule
                    self._build_schedule()
                    
            return module
            
    def _build_value_method(self, method_info):
        """Build a value method operation."""
        # Build function type
        param_types = [ptype.to_mlir(self.context) for _, ptype in method_info['params']]
        return_type = method_info['return_type'].to_mlir(self.context)
        func_type = ir.FunctionType.get(param_types, [return_type])
        
        # Create method operation
        method_op = sharp_dialect.ValueMethodOp(
            method_info['name'],
            func_type
        )
        
        # Build method body
        entry_block = method_op.body.blocks.append(*param_types)
        builder = MethodBuilder(method_op, entry_block)
        
        # Create argument values
        args = {}
        for i, (param_name, _) in enumerate(method_info['params']):
            args[param_name] = Value(entry_block.arguments[i], builder)
            
        # Call the Python function with the builder
        with builder:
            result = method_info['func'](builder, **args)
            if result is not None:
                builder.return_(result)
            else:
                # For value methods, we need to return something
                builder.return_(builder.constant(0, method_info['return_type']))
                
    def _build_action_method(self, method_info):
        """Build an action method operation."""
        # Build function type
        param_types = [ptype.to_mlir(self.context) for _, ptype in method_info['params']]
        return_types = []
        if method_info['return_type'] is not None:
            return_types.append(method_info['return_type'].to_mlir(self.context))
        func_type = ir.FunctionType.get(param_types, return_types)
        
        # Create method operation
        method_op = sharp_dialect.ActionMethodOp(
            method_info['name'],
            func_type
        )
        
        # Build method body
        entry_block = method_op.body.blocks.append(*param_types)
        builder = MethodBuilder(method_op, entry_block)
        
        # Create argument values
        args = {}
        for i, (param_name, _) in enumerate(method_info['params']):
            args[param_name] = Value(entry_block.arguments[i], builder)
            
        # Call the Python function with the builder
        with builder:
            result = method_info['func'](builder, **args)
            if result is not None:
                builder.return_(result)
            else:
                builder.return_()
                
    def _build_rule(self, rule_info):
        """Build a rule operation."""
        # Create rule operation
        rule_op = sharp_dialect.RuleOp(rule_info['name'])
        
        # Build rule body
        entry_block = rule_op.body.blocks.append()
        builder = MethodBuilder(rule_op, entry_block)
        
        # Call the Python function with the builder
        with builder:
            rule_info['func'](builder)
            builder.return_()
            
    def _build_schedule(self):
        """Build the schedule operation."""
        # Collect all action names
        actions = []
        for method in self.methods:
            actions.append(method['name'])
        for rule in self.rules:
            actions.append(rule['name'])
            
        # Build conflict matrix attribute
        conflict_dict = {}
        for key, value in self.conflict_matrix.items():
            conflict_dict[key] = ir.IntegerAttr.get(
                ir.IntegerType.get_signless(32, self.context),
                int(value)
            )
            
        conflict_attr = ir.DictAttr.get(conflict_dict, self.context)
        
        # Create schedule operation
        sharp_dialect.ScheduleOp(actions, conflict_attr)


def module(func: Callable) -> ModuleBuilder:
    """Decorator to create a Sharp module.
    
    Example:
        @sharp.module
        def Counter():
            builder = ModuleBuilder("Counter")
            
            @builder.value_method(return_type=i32)
            def getValue(b):
                return b.constant(42)
                
            @builder.action_method()
            def reset(b):
                pass
                
            return builder
    """
    # Call the function to get the builder
    builder = func()
    return builder