#!/bin/bash

SHARP_ROOT="../../.."
SHARP_OPT="$SHARP_ROOT/build/bin/sharp-opt"

echo "=== Chapter 5: Translation with Static Timing Mode ==="
echo ""

echo "Static Timing Mode:"
echo "- Conservative will-fire logic generation"
echo "- All actions conservatively marked as conflicting unless proven safe"
echo "- Minimal hardware optimization but maximum correctness"
echo ""

echo "1. Translating counter with static timing to FIRRTL:"
echo "----------------------------------------------------"
$SHARP_OPT counter_hw.mlir --convert-txn-to-firrtl=will-fire-mode=static > counter_static.fir 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Static FIRRTL generation successful"
    echo "Generated $(wc -l < counter_static.fir) lines of FIRRTL"
else
    echo "❌ Static FIRRTL generation failed"
    cat counter_static.fir
fi
echo ""

echo "2. Translating counter with static timing to Verilog using firtool:"
echo "----------------------------------------------------------------"
# First generate FIRRTL
$SHARP_OPT counter_hw.mlir --convert-txn-to-firrtl=will-fire-mode=static > counter_static.fir 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ FIRRTL generation successful"
    # Then use firtool to convert to Verilog
    $SHARP_ROOT/.install/unified/bin/firtool counter_static.fir --format=mlir --verilog -o counter_static.v 2>counter_static_verilog.log
    if [ -f counter_static.v ]; then
        echo "   ✅ Static Verilog generation successful"
        echo "   Generated $(wc -l < counter_static.v) lines of Verilog"
        echo ""
        echo "   Module interface:"
        grep -E "(module|input|output)" counter_static.v | head -10
        rm counter_static_verilog.log
    else
        echo "   ❌ Static Verilog generation failed"
        echo "   firtool errors:"
        cat counter_static_verilog.log
    fi
else
    echo "   ❌ FIRRTL generation failed"
    cat counter_static.fir
fi
echo ""

echo "3. Analyzing static timing FIRRTL structure:"
echo "--------------------------------------------"
if [ -f counter_static.fir ]; then
    echo "Static timing features:"
    echo "- Will-fire signals: $(grep -c "_wf" counter_static.fir)"
    echo "- Ready signals: $(grep -c "RDY" counter_static.fir)"
    echo "- Enable signals: $(grep -c "EN" counter_static.fir)"
    echo "- When blocks: $(grep -c "firrtl.when" counter_static.fir)"
    echo ""
    echo "First few will-fire signal definitions:"
    grep "_wf.*=" counter_static.fir | head -5
fi
echo ""

echo "4. Testing nested modules with static timing:"
echo "---------------------------------------------"
$SHARP_OPT nested_modules.mlir --convert-txn-to-firrtl=will-fire-mode=static > nested_static.fir 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Static nested modules translation successful"
    echo "Generated $(wc -l < nested_static.fir) lines of FIRRTL"
else
    echo "❌ Static nested modules translation failed"
fi

echo ""
echo "5. Testing datapath with action method return values:"
echo "-----------------------------------------------------"
$SHARP_OPT datapath.mlir --convert-txn-to-firrtl=will-fire-mode=static > datapath_static.fir 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Static datapath translation successful!"
    echo "Generated $(wc -l < datapath_static.fir) lines of FIRRTL"
    echo "🎉 Action methods with return values from child modules now working!"
    echo ""
    echo "Datapath features:"
    echo "- SimpleFifo module with dequeue return value"
    echo "- Hierarchical data flow between modules"
    echo "- Action methods that return data to parent"
else
    echo "❌ Static datapath translation failed"
    cat datapath_static.fir
fi