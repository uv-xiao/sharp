#!/bin/bash

SHARP_ROOT="../../.."
SHARP_OPT="$SHARP_ROOT/build/bin/sharp-opt"

echo "=== Chapter 5: Translation with Dynamic Timing Mode ==="
echo ""

echo "Dynamic Timing Mode:"
echo "- Precise will-fire logic generation using actual dependencies"
echo "- Analyzes method call patterns to optimize conflict detection"
echo "- Balanced between hardware efficiency and analysis complexity"
echo ""

echo "1. Translating counter with dynamic timing to FIRRTL:"
echo "-----------------------------------------------------"
$SHARP_OPT counter_hw.mlir --convert-txn-to-firrtl=will-fire-mode=dynamic > counter_dynamic.fir 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Dynamic FIRRTL generation successful"
    echo "Generated $(wc -l < counter_dynamic.fir) lines of FIRRTL"
else
    echo "❌ Dynamic FIRRTL generation failed"
    cat counter_dynamic.fir
fi
echo ""

echo "2. Translating counter with dynamic timing to Verilog using firtool:"
echo "----------------------------------------------------------------"
# First generate FIRRTL
$SHARP_OPT counter_hw.mlir --convert-txn-to-firrtl=will-fire-mode=dynamic > counter_dynamic.fir 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ FIRRTL generation successful"
    # Then use firtool to convert to Verilog
    $SHARP_ROOT/.install/unified/bin/firtool counter_dynamic.fir --format=mlir --verilog -o counter_dynamic.v 2>counter_dynamic_verilog.log
    if [ -f counter_dynamic.v ]; then
        echo "   ✅ Dynamic Verilog generation successful"
        echo "   Generated $(wc -l < counter_dynamic.v) lines of Verilog"
        echo ""
        echo "   Module interface:"
        grep -E "(module|input|output)" counter_dynamic.v | head -10
        rm counter_dynamic_verilog.log
    else
        echo "   ❌ Dynamic Verilog generation failed"
        echo "   firtool errors:"
        cat counter_dynamic_verilog.log
    fi
else
    echo "   ❌ FIRRTL generation failed"
    cat counter_dynamic.fir
fi
echo ""

echo "3. Analyzing dynamic timing FIRRTL structure:"
echo "---------------------------------------------"
if [ -f counter_dynamic.fir ]; then
    echo "Dynamic timing features:"
    echo "- Will-fire signals: $(grep -c "_wf" counter_dynamic.fir)"
    echo "- Ready signals: $(grep -c "RDY" counter_dynamic.fir)"
    echo "- Enable signals: $(grep -c "EN" counter_dynamic.fir)"
    echo "- When blocks: $(grep -c "firrtl.when" counter_dynamic.fir)"
    echo ""
    echo "First few will-fire signal definitions:"
    grep "_wf.*=" counter_dynamic.fir | head -5
fi
echo ""

echo "4. Testing nested modules with dynamic timing:"
echo "----------------------------------------------"
$SHARP_OPT nested_modules.mlir --convert-txn-to-firrtl=will-fire-mode=dynamic > nested_dynamic.fir 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Dynamic nested modules translation successful"
    echo "Generated $(wc -l < nested_dynamic.fir) lines of FIRRTL"
else
    echo "❌ Dynamic nested modules translation failed"
fi

echo ""
echo "5. Testing datapath with action method return values:"
echo "-----------------------------------------------------"
$SHARP_OPT datapath.mlir --convert-txn-to-firrtl=will-fire-mode=dynamic > datapath_dynamic.fir 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Dynamic datapath translation successful!"
    echo "Generated $(wc -l < datapath_dynamic.fir) lines of FIRRTL"
    echo "🎉 Action methods with return values from child modules now working!"
else
    echo "❌ Dynamic datapath translation failed"
    cat datapath_dynamic.fir
fi