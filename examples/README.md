# Sharp Examples

This directory contains examples demonstrating various features of the Sharp compiler.

## Basic Examples

### simple_alu.mlir
A simple ALU without state, demonstrating:
- Value methods with multiple arguments
- Arithmetic operations (add, sub, mul)
- Bitwise operations (and, or, xor)
- Comparison operations (eq, gt)

### verilog_export_example.mlir
Traffic light controller showing:
- Action methods with conflict matrix
- Verilog export pipeline (Txn → FIRRTL → HW → Verilog)
- Ready/enable signal generation

### jit_simulation_example.mlir
Simple accumulator demonstrating:
- JIT compilation to LLVM IR
- Fast simulation mode
- Rules that always fire

## Analysis Examples

### analysis_example.mlir
Demonstrates analysis passes:
- Conflict matrix inference
- Schedule completion
- Reachability analysis for conditional calls

### most_dynamic_example.mlir
Shows advanced will-fire generation:
- Primitive-level conflict detection
- Most-dynamic mode for fine-grained scheduling
- Conditional primitive calls

## Running Examples

### Basic parsing and verification:
```bash
sharp-opt <example.mlir>
```

### Run analysis passes:
```bash
# Infer missing conflict relationships
sharp-opt <example.mlir> -sharp-infer-conflict-matrix

# Complete partial schedules
sharp-opt <example.mlir> -sharp-action-scheduling

# Add reachability conditions
sharp-opt <example.mlir> -sharp-reachability-analysis
```

### Generate hardware:
```bash
# Convert to FIRRTL
sharp-opt <example.mlir> --convert-txn-to-firrtl

# Export to Verilog
sharp-opt <example.mlir> -txn-export-verilog
```

### Simulate:
```bash
# JIT mode (fast)
sharp-opt <example.mlir> -sharp-simulate=mode=jit

# Transaction-level simulation
sharp-opt <example.mlir> -sharp-simulate=mode=tl
```

## Tutorial Examples

The `sharp-tutorial/` directory contains a comprehensive 8-chapter tutorial with progressively complex examples covering all aspects of Sharp.