# Sharp Development Status

This document tracks the implementation progress of features outlined in PLAN.md.

## Overview

Sharp is implementing transaction-based hardware description with conflict matrix support and FIRRTL/Verilog generation capabilities.

## Feature Status

### ✅ Completed
- Basic Sharp Core dialect with constant operations
- Sharp Txn dialect with modules, methods, rules, and scheduling
- MLIR infrastructure setup with CIRCT integration
- Build system with Pixi package manager
- Testing infrastructure with lit/FileCheck
- **Conflict Matrix (CM) in modules** (2025-06-29)
  - Added CM dictionary attribute to modules for SB, SA, C, CF relations
  - Supports action-to-action conflict specifications
- **Timing attributes for rules/methods** (2025-06-29)
  - Added timing string attribute: "combinational" | "static(n)" | "dynamic"
  - Integrated into rule and method operations

### 🚧 In Progress
- **Conflict Matrix Inference Pass**
  - [ ] Implement analysis pass to complete CM based on inference rules
  - [ ] Support querying submodule CM for parent module inference

### 📋 Planned
  
- **FIRRTL Translation**
  - Create FIRRTL primitive modules (Register, Wire) in `lib/Dialect/Txn/firrtl_primitives/`
  - Implement txn-to-FIRRTL conversion pass
  - Extend Koika-style translation with method support
  - Generate will-fire (wf) logic considering conflict matrix
  
- **Verilog Export**
  - Add sharp-opt command-line options for Verilog generation
  - Integration with CIRCT's export-verilog infrastructure

### 🚫 Known Limitations
- Python bindings have runtime issues
- Multi-cycle operations not yet supported in translation
- Nonsynthesizable primitives will fail translation

## Implementation Notes

### Conflict Matrix Inference Rules
1. Any action conflicts (C) with itself
2. Actions that are both SA and SB conflict (C)
3. Actions calling same action method of same instance conflict (C)
4. Conflict propagation through method calls:
   - m0 SA m1 => a0 SA a1
   - m0 SB m1 => a0 SB a1
   - m0 C m1 => a0 C a1
5. Default to conflict-free (CF) if relationship cannot be determined

### Translation Architecture
- Bottom-up translation: submodules before parent modules
- Will-fire logic must consider:
  - Conflict matrix relationships
  - Register read/write conflicts
  - Multiple calls to same action method (always conflicts)

## Next Steps
1. Implement conflict matrix attributes in TxnOps.td
2. Add timing attributes to rule and method operations
3. Create FIRRTL primitive module templates
4. Design conflict matrix inference algorithm
5. Implement basic txn-to-FIRRTL conversion