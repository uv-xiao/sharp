# Sharp dialect bindings
# Import dialects from the bundled bindings

from ... import sharp as _root

# Import Sharp's txn dialect
try:
    from _root._mlir_libs.dialects import txn
except ImportError:
    # Fallback
    import sharp.dialects.txn as txn

# Import MLIR dialects we need
try:
    from _root._mlir_libs.dialects import (
        arith,
        builtin,
        func,
        scf,
    )
except ImportError:
    # Fallback imports
    import mlir.dialects.arith as arith
    import mlir.dialects.builtin as builtin
    import mlir.dialects.func as func
    import mlir.dialects.scf as scf

# Import CIRCT dialects we need
try:
    from _root._mlir_libs.dialects import (
        hw,
        comb,
        seq,
        sv,
        firrtl,
    )
except ImportError:
    # Fallback imports
    import circt.dialects.hw as hw
    import circt.dialects.comb as comb
    import circt.dialects.seq as seq
    import circt.dialects.sv as sv
    import circt.dialects.firrtl as firrtl

__all__ = [
    'txn', 'arith', 'builtin', 'func', 'scf',
    'hw', 'comb', 'seq', 'sv', 'firrtl'
]