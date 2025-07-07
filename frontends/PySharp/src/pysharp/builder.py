# PySharp module builder - constructs MLIR representation

from typing import Dict, List, Optional, Callable, Any, Tuple
from sharp import ir
from sharp.dialects import txn
from .types import Type, IntType
from .signals import Signal, Const
from .common import (ConflictRelation, Timing, Port, PortDirection,
                     SequenceBefore, SequenceAfter, Conflict, ConflictFree)

class ModuleBuilder:
    """Builder for constructing txn.module operations."""
    
    def __init__(self, name: str):
        self.name = name
        self.ports: List[Port] = []
        self.value_methods: List[Dict[str, Any]] = []
        self.action_methods: List[Dict[str, Any]] = []
        self.rules: List[Dict[str, Any]] = []
        self.conflicts: Dict[Tuple[str, str], ConflictRelation] = {}
        self.module_op = None
        
    def add_port(self, port: Port):
        """Add a port to the module."""
        self.ports.append(port)
    
    def add_value_method(self, name: str, func: Callable, 
                        timing: Timing, always_ready: bool = False):
        """Add a value method."""
        self.value_methods.append({
            'name': name,
            'func': func,
            'timing': timing,
            'always_ready': always_ready
        })
    
    def add_action_method(self, name: str, func: Callable,
                         timing: Timing, always_ready: bool = False,
                         always_enabled: bool = False):
        """Add an action method."""
        self.action_methods.append({
            'name': name,
            'func': func,
            'timing': timing,
            'always_ready': always_ready,
            'always_enabled': always_enabled
        })
    
    def add_rule(self, name: str, func: Callable, 
                guard: Optional[Callable] = None):
        """Add a rule."""
        self.rules.append({
            'name': name,
            'func': func,
            'guard': guard
        })
    
    def set_conflicts(self, conflicts: Dict[Any, ConflictRelation]):
        """Set conflict relationships."""
        # Convert to proper format
        for key, value in conflicts.items():
            if isinstance(key, tuple) and len(key) == 2:
                self.conflicts[key] = value
    
    def build(self) -> ir.Module:
        """Build the MLIR module."""
        with ir.InsertionPoint(ir.Module.current):
            # Create the txn.module operation
            attrs = {'moduleName': ir.StringAttr.get(self.name)}
            self.module_op = txn.ModuleOp(self.name, attributes=attrs)
            
            with ir.InsertionPoint(self.module_op.body):
                # Build value methods
                for method_info in self.value_methods:
                    self._build_value_method(method_info)
                
                # Build action methods
                for method_info in self.action_methods:
                    self._build_action_method(method_info)
                
                # Build rules
                for rule_info in self.rules:
                    self._build_rule(rule_info)
                
                # Build schedule
                self._build_schedule()
        
        return self.module_op
    
    def _build_value_method(self, method_info: Dict[str, Any]):
        """Build a value method."""
        name = method_info['name']
        func = method_info['func']
        timing = method_info['timing']
        
        # Get function signature from Python function
        import inspect
        sig = inspect.signature(func)
        
        # Build argument types (skip 'self')
        arg_types = []
        arg_names = []
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            # For now, assume i32 for untyped parameters
            arg_types.append(ir.IntegerType.get_signless(32))
            arg_names.append(param_name)
        
        # Return type - assume i32 for now
        result_type = ir.IntegerType.get_signless(32)
        
        # Create attributes
        attrs = {
            'timing': ir.StringAttr.get(str(timing))
        }
        if method_info['always_ready']:
            attrs['always_ready'] = ir.UnitAttr.get()
        
        # Create the value method operation
        method_op = txn.ValueMethodOp(
            name, 
            arg_types,
            [result_type],
            attributes=attrs
        )
        
        # Build method body
        with ir.InsertionPoint(method_op.body):
            # Create block arguments
            entry_block = method_op.body.blocks[0]
            
            # Simple implementation - just return a constant
            # In a real implementation, we'd translate the Python function
            result = arith.ConstantOp(
                result_type,
                ir.IntegerAttr.get(result_type, 0)
            )
            txn.ReturnOp([result])
    
    def _build_action_method(self, method_info: Dict[str, Any]):
        """Build an action method."""
        name = method_info['name']
        func = method_info['func']
        timing = method_info['timing']
        
        # Get function signature
        import inspect
        sig = inspect.signature(func)
        
        # Build argument types
        arg_types = []
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            arg_types.append(ir.IntegerType.get_signless(32))
        
        # Create attributes
        attrs = {
            'timing': ir.StringAttr.get(str(timing))
        }
        if method_info['always_ready']:
            attrs['always_ready'] = ir.UnitAttr.get()
        if method_info['always_enabled']:
            attrs['always_enabled'] = ir.UnitAttr.get()
        
        # Create the action method operation
        method_op = txn.ActionMethodOp(name, arg_types, attributes=attrs)
        
        # Build method body
        with ir.InsertionPoint(method_op.body):
            # Simple implementation - just yield
            txn.YieldOp([])
    
    def _build_rule(self, rule_info: Dict[str, Any]):
        """Build a rule."""
        name = rule_info['name']
        func = rule_info['func']
        guard = rule_info['guard']
        
        # Create the rule operation
        rule_op = txn.RuleOp(name)
        
        # Build rule body
        with ir.InsertionPoint(rule_op.body):
            # Create guard condition
            if guard:
                # Call guard function - simplified
                guard_val = arith.ConstantOp(
                    ir.IntegerType.get_signless(1),
                    ir.IntegerAttr.get(ir.IntegerType.get_signless(1), 1)
                )
            else:
                # Default to always true
                guard_val = arith.ConstantOp(
                    ir.IntegerType.get_signless(1),
                    ir.IntegerAttr.get(ir.IntegerType.get_signless(1), 1)
                )
            
            # Rule body - simplified
            # In a real implementation, we'd translate the Python function
            
            # Yield the guard value
            txn.YieldOp([guard_val])
    
    def _build_schedule(self):
        """Build the schedule operation."""
        # Collect all schedulable names (only actions: action methods and rules)
        # According to Sharp's execution model, value methods are not scheduled
        schedulable_names = []
        schedulable_names.extend(m['name'] for m in self.action_methods)
        schedulable_names.extend(r['name'] for r in self.rules)
        
        # Build conflict matrix dictionary
        conflict_dict = {}
        for (name1, name2), relation in self.conflicts.items():
            key = f"{name1},{name2}"
            conflict_dict[key] = ir.IntegerAttr.get(
                ir.IntegerType.get_signless(32), 
                int(relation)
            )
        
        # Create schedule operation
        schedule_attrs = {}
        if conflict_dict:
            schedule_attrs['conflict_matrix'] = ir.DictAttr.get(conflict_dict)
        
        schedule_op = txn.ScheduleOp(
            schedulable_names,
            attributes=schedule_attrs
        )
    
    def get_mlir_module(self) -> ir.Module:
        """Get the built MLIR module."""
        return self.module_op

__all__ = ['ModuleBuilder']