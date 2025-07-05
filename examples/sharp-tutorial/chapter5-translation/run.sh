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