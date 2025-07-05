# Custom __init__.py for PySharp's bundled sharp module
# This replaces the standard sharp __init__.py to work in the nested structure

# The MLIR/CIRCT modules are provided by the build system
# Import them to make them available as sharp.ir, sharp.passmanager, etc.
from . import ir
from . import passmanager
from . import rewrite

# Import the native Sharp extension from the nested location
from ._mlir_libs import _sharp

# Function to register all dialects
def register_sharp_dialects(context):
    """Register Sharp, MLIR, and CIRCT dialects on a Context."""
    _sharp.register_dialects(context)

# Re-export everything for convenience
__all__ = ['ir', 'passmanager', 'rewrite', 'register_sharp_dialects']