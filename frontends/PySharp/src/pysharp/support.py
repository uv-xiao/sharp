# PySharp support utilities

from typing import Any, Optional, Union, List
from sharp import ir, passmanager
import sys
import io

def emit_mlir(module: ir.Module, file: Optional[Union[str, io.IOBase]] = None):
    """Emit MLIR representation of a module."""
    if file is None:
        print(module)
    elif isinstance(file, str):
        with open(file, 'w') as f:
            f.write(str(module))
    else:
        file.write(str(module))

def emit_verilog(module: ir.Module, file: Optional[Union[str, io.IOBase]] = None):
    """Emit Verilog from a module using the txn-export-verilog pipeline."""
    pm = passmanager.PassManager.parse("builtin.module(txn-export-verilog)")
    pm.run(module.operation)
    
    # The Verilog would be in the transformed module
    # This is a simplified version - real implementation would extract the SV
    emit_mlir(module, file)

def run_passes(module: ir.Module, pipeline: str) -> ir.Module:
    """Run a pass pipeline on a module."""
    pm = passmanager.PassManager.parse(pipeline)
    pm.run(module.operation)
    return module

def verify_module(module: ir.Module) -> bool:
    """Verify that a module is well-formed."""
    try:
        module.operation.verify()
        return True
    except Exception as e:
        print(f"Verification failed: {e}", file=sys.stderr)
        return False

class Context:
    """Context manager for PySharp operations."""
    
    def __init__(self):
        self.modules: List[ir.Module] = []
    
    def add_module(self, module: ir.Module):
        """Add a module to the context."""
        self.modules.append(module)
    
    def get_modules(self) -> List[ir.Module]:
        """Get all modules in the context."""
        return self.modules
    
    def emit_all(self, file: Optional[Union[str, io.IOBase]] = None):
        """Emit all modules."""
        for module in self.modules:
            emit_mlir(module, file)

# Global context
_global_context = Context()

def get_context() -> Context:
    """Get the global PySharp context."""
    return _global_context

__all__ = [
    'emit_mlir', 'emit_verilog', 'run_passes', 'verify_module',
    'Context', 'get_context'
]