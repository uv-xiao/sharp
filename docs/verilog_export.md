# Verilog Export

## Overview

Sharp provides integrated Verilog export through CIRCT's infrastructure:
1. **Txn to FIRRTL** - Transaction modules to FIRRTL
2. **FIRRTL to HW** - FIRRTL to CIRCT's HW dialect
3. **HW to Verilog** - Export synthesizable Verilog

## Command-Line Usage

### Basic Export
```bash
sharp-opt --txn-export-verilog input.mlir -o output.v
```

### Step-by-Step
```bash
# Txn to FIRRTL only
sharp-opt --convert-txn-to-firrtl input.mlir -o output.fir.mlir

# Txn to HW dialects
sharp-opt --lower-to-hw input.mlir -o output.hw.mlir

# Complete pipeline
sharp-opt --txn-export-verilog input.mlir -o output.v
```

### Split Export
```bash
# Each module in separate file
sharp-opt --export-split-verilog --split-verilog-dir ./output/ input.mlir
```

## Options
- `--txn-export-verilog` - Single file export
- `--export-split-verilog` - Multiple file export
- `--split-verilog-dir <dir>` - Output directory
- `--lower-to-hw` - Stop at HW dialect
- `--convert-txn-to-firrtl` - Stop at FIRRTL

## Example

Input:
```mlir
txn.module @Counter {
  %reg = txn.instance @count of @Register<i32>
  
  txn.rule @increment {
    %val = txn.call @count::@read() : () -> i32
    %one = arith.constant 1 : i32
    %next = arith.addi %val, %one : i32
    txn.call @count::@write(%next) : (i32) -> ()
    txn.yield
  }
  
  txn.schedule [@increment]
}
```

Export:
```bash
sharp-opt --txn-export-verilog counter.mlir -o counter.v
```

Generates:
- Clock/reset inputs
- Method interfaces
- Scheduled logic
- Primitive instances

## Build Integration

Makefile:
```makefile
%.v: %.mlir
	sharp-opt --txn-export-verilog $< -o $@

%.fir.mlir: %.mlir
	sharp-opt --convert-txn-to-firrtl $< -o $@
```

CMake:
```cmake
add_custom_command(
  OUTPUT ${OUTPUT_FILE}.v
  COMMAND sharp-opt --txn-export-verilog ${INPUT_FILE} -o ${OUTPUT_FILE}.v
  DEPENDS ${INPUT_FILE}
)
```