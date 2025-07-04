# Sharp MLIR/CIRCT bindings wrapper
# This module imports from the installed Sharp Python bindings

try:
    # Try to import from the bundled _mlir_libs
    from . import _mlir_libs
except ImportError:
    # Fallback to system-installed bindings
    import _sharp as _mlir_libs

# Re-export core MLIR functionality
from ._mlir_libs import ir
from ._mlir_libs import passmanager

# Import and register dialects
def register_dialects(context):
    """Register all required dialects with the context."""
    # Register MLIR dialects
    context.load_all_available_dialects()
    
    # Sharp's txn dialect should be automatically registered
    # CIRCT dialects should also be available

# Re-export dialect modules
from . import dialects

__all__ = ['ir', 'passmanager', 'register_dialects', 'dialects']