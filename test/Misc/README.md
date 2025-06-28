# Miscellaneous Tests

This directory contains experimental tests and trials for various MLIR features and dialect interactions.

## Tests

### dialects-coexist.mlir
Demonstrates that FIRRTL, hw, comb, and seq dialects can coexist in the same IR module. This test shows:

1. **hw.module** - Hardware modules using the HW dialect
2. **comb operations** - Combinational logic (add, sub, mux, etc.)
3. **seq operations** - Sequential logic (registers with clock)
4. **firrtl.circuit** - FIRRTL circuits and modules
5. **Mixed usage** - All dialects can be used in the same compilation unit

Key findings:
- FIRRTL uses its own type system (`!firrtl.uint<N>`, `!firrtl.clock`)
- HW/Comb/Seq use standard integer types (`iN`) and `!seq.clock`
- Both can coexist without conflicts
- Conversion between the type systems would require specific lowering passes

### Usage
```bash
# Run a test
sharp-opt test/Misc/dialects-coexist.mlir

# Verify with FileCheck
sharp-opt test/Misc/dialects-coexist.mlir | FileCheck test/Misc/dialects-coexist.mlir
```