#!/bin/bash

SHARP_ROOT="../../.."
SHARP_OPT="$SHARP_ROOT/build/bin/sharp-opt"

echo "=== Chapter 5: Translation with Most-Dynamic Timing Mode ==="
echo ""

echo "Most-Dynamic Timing Mode:"
echo "- Primitive-level tracking for maximum optimization"
echo "- Individual primitive method calls tracked separately"
echo "- Highest hardware efficiency but most complex analysis"
echo ""

echo "1. Translating counter with most-dynamic timing to FIRRTL:"
echo "----------------------------------------------------------"
$SHARP_OPT counter_hw.mlir --convert-txn-to-firrtl=will-fire-mode=most-dynamic > counter_most_dynamic.fir 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Most-Dynamic FIRRTL generation successful"
    echo "Generated $(wc -l < counter_most_dynamic.fir) lines of FIRRTL"
else
    echo "❌ Most-Dynamic FIRRTL generation failed"
    cat counter_most_dynamic.fir
fi
echo ""

echo "2. Translating counter with most-dynamic timing to Verilog using firtool:"
echo "------------------------------------------------------------------------"
# First generate FIRRTL
$SHARP_OPT counter_hw.mlir --convert-txn-to-firrtl=will-fire-mode=most-dynamic > counter_most_dynamic.fir 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ FIRRTL generation successful"
    # Then use firtool to convert to Verilog
    $SHARP_ROOT/.install/unified/bin/firtool counter_most_dynamic.fir --format=mlir --verilog -o counter_most_dynamic.v 2>counter_most_dynamic_verilog.log
    if [ -f counter_most_dynamic.v ]; then
        echo "   ✅ Most-Dynamic Verilog generation successful"
        echo "   Generated $(wc -l < counter_most_dynamic.v) lines of Verilog"
        echo ""
        echo "   Module interface:"
        grep -E "(module|input|output)" counter_most_dynamic.v | head -10
        rm counter_most_dynamic_verilog.log
    else
        echo "   ❌ Most-Dynamic Verilog generation failed"
        echo "   firtool errors:"
        cat counter_most_dynamic_verilog.log
    fi
else
    echo "   ❌ FIRRTL generation failed"
    cat counter_most_dynamic.fir
fi
echo ""

echo "3. Analyzing most-dynamic timing FIRRTL structure:"
echo "--------------------------------------------------"
if [ -f counter_most_dynamic.fir ]; then
    echo "Most-dynamic timing features:"
    echo "- Will-fire signals: $(grep -c "_wf" counter_most_dynamic.fir)"
    echo "- Ready signals: $(grep -c "RDY" counter_most_dynamic.fir)"
    echo "- Enable signals: $(grep -c "EN" counter_most_dynamic.fir)"
    echo "- When blocks: $(grep -c "firrtl.when" counter_most_dynamic.fir)"
    echo ""
    echo "First few will-fire signal definitions:"
    grep "_wf.*=" counter_most_dynamic.fir | head -5
fi
echo ""

echo "4. Testing nested modules with most-dynamic timing:"
echo "---------------------------------------------------"
$SHARP_OPT nested_modules.mlir --convert-txn-to-firrtl=will-fire-mode=most-dynamic > nested_most_dynamic.fir 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Most-Dynamic nested modules translation successful"
    echo "Generated $(wc -l < nested_most_dynamic.fir) lines of FIRRTL"
else
    echo "❌ Most-Dynamic nested modules translation failed"
fi

echo ""
echo "5. Testing datapath with action method return values:"
echo "-----------------------------------------------------"
$SHARP_OPT datapath.mlir --convert-txn-to-firrtl=will-fire-mode=most-dynamic > datapath_most_dynamic.fir 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Most-Dynamic datapath translation successful!"
    echo "Generated $(wc -l < datapath_most_dynamic.fir) lines of FIRRTL"
    echo "🎉 Action methods with return values from child modules now working!"
else
    echo "❌ Most-Dynamic datapath translation failed"
    cat datapath_most_dynamic.fir
fi