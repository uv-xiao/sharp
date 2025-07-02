# Verilog Export

Sharp provides integrated Verilog export functionality through CIRCT's export-verilog infrastructure. This allows you to compile Txn modules all the way to synthesizable Verilog.

## Overview

The Verilog export pipeline consists of:
1. **Txn to FIRRTL conversion** - Converts transaction-based modules to FIRRTL
2. **FIRRTL to HW lowering** - Lowers FIRRTL to CIRCT's HW dialect
3. **Verilog emission** - Exports HW dialect to Verilog

## Command-Line Usage

### Basic Export

To export a Txn module to Verilog:

```bash
sharp-opt --txn-export-verilog input.mlir -o output.v
```

This automatically runs the complete lowering pipeline.

### Step-by-Step Lowering

You can also run the pipeline step-by-step:

```bash
# Convert Txn to FIRRTL only
sharp-opt --convert-txn-to-firrtl input.mlir -o output.fir.mlir

# Convert Txn to HW dialects (includes FIRRTL step)
sharp-opt --lower-to-hw input.mlir -o output.hw.mlir

# Export to Verilog (includes all previous steps)
sharp-opt --txn-export-verilog input.mlir -o output.v
```

### Split Verilog Export

For larger designs, you can export each module to a separate file:

```bash
sharp-opt --export-split-verilog --split-verilog-dir ./output/ input.mlir
```

This creates one `.v` file per module in the specified directory.

## Options

- `--txn-export-verilog` - Export to a single Verilog file (includes full lowering pipeline)
- `--export-split-verilog` - Export each module to a separate file
- `--split-verilog-dir <dir>` - Directory for split Verilog output (default: `./`)
- `--lower-to-hw` - Stop after lowering to HW dialect
- `--convert-txn-to-firrtl` - Stop after Txn to FIRRTL conversion
- `--timing` - Display execution time of each pass

## Example

Given a simple counter module:

```mlir
txn.module @Counter {
  %reg = txn.instance @count of @Register<i32> : \!txn.module<"Register">
  
  txn.rule @increment {
    %val = txn.call @count::@read() : () -> i32
    %c1 = arith.constant 1 : i32
    %new_val = arith.addi %val, %c1 : i32
    txn.call @count::@write(%new_val) : (i32) -> ()
    txn.return
  }
  
  txn.value_method @get() -> i32 {
    %val = txn.call @count::@read() : () -> i32
    txn.return %val : i32
  }
  
  txn.schedule [@increment] {}
}
```

Export to Verilog:

```bash
sharp-opt --txn-export-verilog counter.mlir -o counter.v
```

This generates synthesizable Verilog with:
- Clock and reset inputs
- Method interfaces (enable/ready signals)
- Proper scheduling based on conflict matrix
- Instantiated primitives (Register, Wire, etc.)

## Integration with Build Systems

The `sharp-opt` tool can be integrated into standard hardware build flows:

```makefile
# Makefile example
%.v: %.mlir
	sharp-opt --txn-export-verilog $< -o $@

# With split output
%.verilog/: %.mlir
	mkdir -p $@
	sharp-opt --export-split-verilog --split-verilog-dir $@ $<
```

## Limitations

- Empty action bodies may generate empty `when` blocks in FIRRTL that cause issues. Ensure all actions have some operations.
- The pipeline requires complete schedules. Use `--sharp-action-scheduling` pass if needed.
- Verilog export preserves the module hierarchy from Txn, including primitive instantiations.

## Advanced Usage

### Custom Lowering Options

CIRCT provides various lowering options that can be configured:

```bash
sharp-opt --txn-export-verilog \
  --lowering-options="emitBindComments,disallowLocalVariables" \
  input.mlir -o output.v
```

See CIRCT documentation for full lowering options.

### Debugging

To debug the lowering pipeline:

```bash
# See FIRRTL output
sharp-opt --convert-txn-to-firrtl input.mlir | less

# See HW dialect output  
sharp-opt --lower-to-hw input.mlir | less

# Time each pass
sharp-opt --txn-export-verilog --timing input.mlir -o output.v
```
EOF < /dev/null