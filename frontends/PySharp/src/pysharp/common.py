# PySharp common definitions

from enum import Enum, IntEnum
from typing import Optional, Union, List, Dict, Any
from .types import Type, IntType, ClockType, ResetType

class ConflictRelation(IntEnum):
    """Conflict relations between actions in Sharp.
    
    According to Sharp's execution model, schedules are predetermined,
    so only Conflict and ConflictFree relations are meaningful.
    """
    SequenceBefore = 0  # SB - first action must execute before second
    SequenceAfter = 1   # SA - first action must execute after second
    Conflict = 2        # C - actions conflict and cannot execute together
    ConflictFree = 3    # CF - actions can execute in any order

# Convenient aliases
SequenceBefore = ConflictRelation.SequenceBefore
SequenceAfter = ConflictRelation.SequenceAfter
Conflict = ConflictRelation.Conflict
ConflictFree = ConflictRelation.ConflictFree

# Short aliases
SB = SequenceBefore
SA = SequenceAfter
C = Conflict
CF = ConflictFree

class PortDirection(Enum):
    """Port direction for module interfaces."""
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"

class Port:
    """Represents a module port."""
    
    def __init__(self, name: str, type: Type, direction: PortDirection):
        self.name = name
        self.type = type
        self.direction = direction
    
    def __repr__(self):
        return f"Port({self.name}, {self.type}, {self.direction})"

# Helper functions for port creation
def Input(type: Type, name: Optional[str] = None) -> Port:
    """Create an input port."""
    if name is None:
        name = "input"
    return Port(name, type, PortDirection.INPUT)

def Output(type: Type, name: Optional[str] = None) -> Port:
    """Create an output port."""
    if name is None:
        name = "output"
    return Port(name, type, PortDirection.OUTPUT)

def Clock(name: str = "clock") -> Port:
    """Create a clock input port."""
    return Port(name, ClockType(), PortDirection.INPUT)

def Reset(name: str = "reset") -> Port:
    """Create a reset input port."""
    return Port(name, ResetType(), PortDirection.INPUT)

class Timing:
    """Timing specification for methods and rules."""
    
    def __init__(self, kind: str, cycles: Optional[int] = None):
        self.kind = kind
        self.cycles = cycles
    
    def __str__(self):
        if self.kind == "static" and self.cycles is not None:
            return f"static({self.cycles})"
        return self.kind
    
    @classmethod
    def combinational(cls):
        """Create combinational timing."""
        return cls("combinational")
    
    @classmethod
    def static(cls, cycles: int):
        """Create static timing with specified cycles."""
        return cls("static", cycles)
    
    @classmethod
    def dynamic(cls):
        """Create dynamic timing."""
        return cls("dynamic")

# Timing presets
Combinational = Timing.combinational()
Dynamic = Timing.dynamic()

def Static(cycles: int) -> Timing:
    """Create static timing with specified cycles."""
    return Timing.static(cycles)

class MethodAttribute:
    """Base class for method attributes."""
    pass

class AlwaysReady(MethodAttribute):
    """Method is always ready."""
    pass

class AlwaysEnabled(MethodAttribute):
    """Method is always enabled."""
    pass

__all__ = [
    'ConflictRelation', 'Conflict', 'ConflictFree',
    'C', 'CF',
    'PortDirection', 'Port', 'Input', 'Output', 'Clock', 'Reset',
    'Timing', 'Combinational', 'Dynamic', 'Static',
    'MethodAttribute', 'AlwaysReady', 'AlwaysEnabled'
]