# PySharp - Pythonic hardware description for Sharp
# Following PyCDE pattern of importing from bundled MLIR/CIRCT

from .sharp import ir
from . import sharp
import atexit

# Push a default context onto the context stack at import time
DefaultContext = ir.Context()
DefaultContext.__enter__()
sharp.register_dialects(DefaultContext)
DefaultContext.allow_unregistered_dialects = True

@atexit.register
def __exit_ctxt():
  DefaultContext.__exit__(None, None, None)

# Core imports
from .common import (Clock, Reset, Input, Output, ConflictRelation,
                     SequenceBefore, SequenceAfter, Conflict, ConflictFree)
from .module import (Module, module, value_method, action_method, rule)
from .types import (types, i1, i8, i16, i32, i64, i128, uint, sint)
from .signals import (Signal, Const, Wire, Reg)
from .builder import (ModuleBuilder)

# Set default location
DefaultLocation = ir.Location.unknown()
DefaultLocation.__enter__()

@atexit.register
def __exit_loc():
  DefaultLocation.__exit__(None, None, None)

# Version info
__version__ = "0.1.0"