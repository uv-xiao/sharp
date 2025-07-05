# Chapter 5: Translation to FIRRTL and Verilog

## Overview

Sharp modules can be translated to standard hardware description languages for synthesis and implementation. This chapter covers:
- Txn to FIRRTL conversion
- FIRRTL to Verilog generation
- Handling primitives in translation
- Verification of translated designs

## Translation Pipeline

```
Txn Module → FIRRTL → Verilog
           ↓         ↓
      (Simulation) (Synthesis)
```

## Txn to FIRRTL Translation

The `--convert-txn-to-firrtl` pass converts transaction-level modules to FIRRTL:
- Methods become module ports
- Conflict checking generates ready/valid signals
- Primitives instantiate FIRRTL implementations
- Schedule determines arbitration logic

## Example: Counter Translation

Let's translate a simple counter:

### counter_hw.mlir

```mlir
// Hardware counter for translation
txn.module @HardwareCounter {
  %count = txn.instance @count of @Register<i32> : !txn.module<"Register">
  
  // Enable signal (input)
  txn.value_method @getCount() -> i32 {
    %val = txn.call @count::@read() : () -> i32
    txn.return %val : i32
  }
  
  // Increment action
  txn.action_method @increment() {
    %current = txn.call @count::@read() : () -> i32
    %one = arith.constant 1 : i32
    %next = arith.addi %current, %one : i32
    txn.call @count::@write(%next) : (i32) -> ()
    txn.yield
  }
  
  // Decrement action
  txn.action_method @decrement() {
    %current = txn.call @count::@read() : () -> i32
    %one = arith.constant 1 : i32
    %next = arith.subi %current, %one : i32
    txn.call @count::@write(%next) : (i32) -> ()
    txn.yield
  }
  
  // Reset action
  txn.action_method @reset() {
    %zero = arith.constant 0 : i32
    txn.call @count::@write(%zero) : (i32) -> ()
    txn.yield
  }
  
  txn.schedule [@getCount, @increment, @decrement, @reset] {
    conflict_matrix = {
      // Value method doesn't conflict
      "getCount,getCount" = 3 : i32,       // CF
      "getCount,increment" = 3 : i32,      // CF
      "getCount,decrement" = 3 : i32,      // CF
      "getCount,reset" = 3 : i32,          // CF
      
      // Actions conflict with each other
      "increment,increment" = 2 : i32,     // C
      "increment,decrement" = 2 : i32,     // C
      "increment,reset" = 2 : i32,         // C
      "decrement,decrement" = 2 : i32,     // C
      "decrement,reset" = 2 : i32,         // C
      "reset,reset" = 2 : i32              // C
    }
  }
}
```

### Translating to FIRRTL

```bash
sharp-opt counter_hw.mlir --convert-txn-to-firrtl > counter.fir
```

The output FIRRTL includes:
- Module with clock and reset ports
- Method ports (enable/ready/data)
- Conflict resolution logic
- Register primitive instances

## FIRRTL to Verilog

Use CIRCT's export pipeline:

```bash
sharp-opt counter_hw.mlir --txn-export-verilog -o counter.v
```

This runs the complete pipeline:
1. Txn → FIRRTL
2. FIRRTL → HW dialect
3. HW → Verilog

### Generated Verilog Structure

```verilog
module HardwareCounter(
  input clock,
  input reset,
  
  // getCount method
  output [31:0] getCount_data,
  input getCount_enable,
  output getCount_ready,
  
  // increment method
  input increment_enable,
  output increment_ready,
  
  // decrement method
  input decrement_enable,
  output decrement_ready,
  
  // reset method
  input reset_enable,
  output reset_ready
);
  // Implementation...
endmodule
```

## Complex Example with Multiple Primitives

### datapath.mlir

```mlir
// Datapath with FIFO and processing
txn.module @Datapath {
  %input_fifo = txn.instance @input_fifo of @FIFO<i32> : !txn.module<"FIFO">
  %output_reg = txn.instance @output_reg of @Register<i32> : !txn.module<"Register">
  %status = txn.instance @status of @Register<i1> : !txn.module<"Register">
  
  // Input data
  txn.action_method @pushData(%data: i32) {
    txn.call @input_fifo::@enqueue(%data) : (i32) -> ()
    txn.yield
  }
  
  // Process and store
  txn.action_method @process() {
    // Check if data available
    %empty = txn.call @input_fifo::@isEmpty() : () -> i1
    %true = arith.constant true
    %has_data = arith.xori %empty, %true : i1
    
    // Get data and process
    %data = txn.call @input_fifo::@dequeue() : () -> i32
    %two = arith.constant 2 : i32
    %processed = arith.muli %data, %two : i32
    
    // Store result
    txn.call @output_reg::@write(%processed) : (i32) -> ()
    txn.call @status::@write(%has_data) : (i1) -> ()
    txn.yield
  }
  
  // Read output
  txn.value_method @getOutput() -> i32 {
    %val = txn.call @output_reg::@read() : () -> i32
    txn.return %val : i32
  }
  
  // Check status
  txn.value_method @isReady() -> i1 {
    %val = txn.call @status::@read() : () -> i1
    txn.return %val : i1
  }
  
  txn.schedule [@pushData, @process, @getOutput, @isReady]
}
```

## Translation Script

### run.sh

```bash
#!/bin/bash

SHARP_ROOT="../../.."
SHARP_OPT="$SHARP_ROOT/build/bin/sharp-opt"

echo "=== Chapter 5: Translation to Hardware ==="
echo ""

echo "1. Translating simple counter to FIRRTL:"
echo "----------------------------------------"
$SHARP_OPT counter_hw.mlir --convert-txn-to-firrtl > counter.fir 2>&1
if [ $? -eq 0 ]; then
    echo "✅ FIRRTL generation successful"
    echo "Generated $(wc -l < counter.fir) lines of FIRRTL"
else
    echo "❌ FIRRTL generation failed"
fi
echo ""

echo "2. Translating counter to Verilog:"
echo "----------------------------------------"
$SHARP_OPT counter_hw.mlir --txn-export-verilog -o counter.v 2>&1
if [ -f counter.v ]; then
    echo "✅ Verilog generation successful"
    echo "Generated $(wc -l < counter.v) lines of Verilog"
    echo ""
    echo "Module interface:"
    grep -E "(module|input|output)" counter.v | head -10
else
    echo "❌ Verilog generation failed"
fi
echo ""

echo "3. Translating complex datapath:"
echo "----------------------------------------"
$SHARP_OPT datapath.mlir --convert-txn-to-firrtl > datapath.fir 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Complex module translation successful"
    # Note: FIFO translation requires full primitive implementation
else
    echo "⚠️  Translation incomplete (FIFO primitive needs FIRRTL impl)"
fi
echo ""

echo "4. Checking FIRRTL output structure:"
echo "----------------------------------------"
if [ -f counter.fir ]; then
    echo "FIRRTL circuit structure:"
    grep -E "(circuit|module|input|output)" counter.fir | head -15
fi
```

## Understanding Translation

### Method Port Mapping
Each Txn method becomes hardware ports:
- **Value methods**: data output + enable input + ready output
- **Action methods**: enable input + ready output + optional data

### Conflict Resolution
The conflict matrix generates:
- Ready signals based on current state
- Arbitration when multiple methods enabled
- Proper sequencing for SB/SA relations

### Primitive Handling
Primitives are replaced with their FIRRTL implementations:
- Register → FIRRTL RegInit
- Wire → FIRRTL wire
- FIFO → Custom FIRRTL module

## Verification

After translation, verify correctness:
1. Check port connectivity
2. Verify timing behavior
3. Ensure conflict resolution
4. Test with simulation

## Exercises

1. **Add handshaking**: Modify a method to use req/ack protocol
2. **Parameterize width**: Make the counter width configurable
3. **Add pipeline stages**: Create a multi-stage processing datapath
4. **Custom primitive**: Define and translate a new primitive type

## Advanced Topics

### Optimization During Translation
- Constant propagation
- Dead code elimination
- State minimization
- Resource sharing

### Backend-Specific Features
Different backends support:
- ASIC vs FPGA optimizations
- Vendor-specific primitives
- Timing constraints
- Power optimization

## Key Takeaways

- Translation preserves transaction semantics in RTL
- Conflict matrices become hardware arbiters
- Methods map to well-defined port interfaces
- Primitives need backend implementations
- Verification ensures correctness

## Next Chapter

Chapter 6 explores simulation modes:
- Transaction-level simulation
- RTL simulation with Arcilator
- Hybrid TL/RTL simulation
- Performance comparison