# Chapter 5: Translation to FIRRTL and Verilog

## Overview

Sharp modules can be translated to standard hardware description languages for synthesis and implementation. This chapter covers:
- Txn to FIRRTL conversion
- FIRRTL to Verilog generation
- Handling primitives in translation
- Verification of translated designs

## Txn to FIRRTL Translation

The `--convert-txn-to-firrtl` pass converts transaction-level modules to FIRRTL:
- Methods become module ports
- Conflict checking generates ready/valid signals
- Primitives instantiate FIRRTL implementations
- Schedule determines arbitration logic


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
  %input_fifo = txn.instance @input_fifo of @FIFO<i32> : index
  %output_reg = txn.instance @output_reg of @Register<i32> : index
  %status = txn.instance @status of @Register<i1> : index
  
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

## Timing Modes

Sharp supports two timing modes for will-fire logic generation, each with different trade-offs:

### Static Mode (`--convert-txn-to-firrtl=will-fire-mode=static`)

**Characteristics:**
- Conservative will-fire logic generation
- Fixed compile-time priority scheme 
- Rigid priority-encoder chains
- Predictable but can cause starvation

**Use Cases:**
- Maximum safety and predictability
- Simple designs
- Early development and prototyping
- When deterministic behavior is critical

**Generated Logic:** Most conservative, clear priority relationships

### Dynamic Mode (`--convert-txn-to-firrtl=will-fire-mode=dynamic`)

**Characteristics:**
- Balanced scheduling with fairness considerations
- More sophisticated arbitration logic
- Prevents starvation through dynamic arbitration
- Balanced hardware complexity

**Use Cases:**
- Production designs requiring fairness
- Multi-action modules with balanced priorities
- When both performance and fairness matter
- General-purpose default mode

**Generated Logic:** More complex but fairer arbitration


### Timing Mode Comparison

Run all timing modes with:
```bash
./run_translation.sh     # Runs all modes
./run.sh                 # Convenience wrapper for all modes
```

Individual timing modes:
```bash
./run_translation.sh static        # Conservative, predictable
./run_translation.sh dynamic       # Balanced, fair
```

**Complexity Comparison (for counter example):**
- Static: 93 lines FIRRTL, 15 will-fire signals, 26 logic gates
- Dynamic: 108 lines FIRRTL, 18 will-fire signals, 30 logic gates

Dynamic mode provides a good balance between scheduling flexibility and hardware complexity.

## Module Nesting and Hierarchical Design

Sharp supports hierarchical module composition where modules can instantiate other custom modules (not just primitives). This enables:

### Nested Module Example

```mlir
// Inner module: SimpleAdder
txn.module @SimpleAdder {
  %result = txn.instance @result of @Register<i32> : index
  
  txn.action_method @add(%a: i32, %b: i32) { ... }
  txn.value_method @getResult() -> i32 { ... }
  txn.action_method @reset() { ... }
  
  txn.schedule [@add, @reset] { ... }
}

// Outer module: DualProcessor  
txn.module @DualProcessor {
  %adder1 = txn.instance @adder1 of @SimpleAdder : index
  %adder2 = txn.instance @adder2 of @SimpleAdder : index
  
  txn.action_method @processA(%x: i32, %y: i32) {
    txn.call @adder1::@add(%x, %y) : (i32, i32) -> ()
    txn.yield
  }
  // ...
}
```

### Timing Mode Impact on Nested Modules

**Complexity comparison for nested modules:**
- Static: 156 lines FIRRTL (most conservative)
- Dynamic: 166 lines FIRRTL (balanced arbitration)

The timing mode differences become more pronounced with nested hierarchies, as each module's scheduling decisions interact with the parent module's timing mode.

### Design Benefits

1. **Modularity**: Reusable components with encapsulated behavior
2. **Scalability**: Complex systems built from simpler verified modules  
3. **Timing Isolation**: Each module has independent scheduling constraints
4. **Testing**: Individual modules can be verified separately

### Current Limitations

- Value method calls to child modules require careful handling
- Action methods with return values from child modules need proper port mapping
- Complex nested schedules may require timing mode tuning

## Key Takeaways

- Translation preserves transaction semantics in RTL
- Conflict matrices become hardware arbiters
- Methods map to well-defined port interfaces
- Primitives need backend implementations
- Two timing modes offer different performance/complexity trade-offs
- Choose timing mode based on your priority requirements
- Verification ensures correctness across all modes

## Next Chapter

Chapter 6 explores simulation modes:
- Transaction-level simulation
- RTL simulation with Arcilator
- Hybrid TL/RTL simulation
- Performance comparison