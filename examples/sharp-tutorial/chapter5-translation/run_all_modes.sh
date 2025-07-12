#!/bin/bash

SHARP_ROOT="../../.."
SHARP_OPT="$SHARP_ROOT/build/bin/sharp-opt"

echo "=== Chapter 5: Comparison of All Timing Modes ==="
echo ""

echo "Running all timing modes for comparison..."
echo ""

# Run each timing mode
echo "========================================="
./run_static.sh
echo ""

echo "========================================="
./run_dynamic.sh
echo ""

echo "========================================="
./run_most_dynamic.sh
echo ""

echo "========================================="
echo "=== Timing Mode Comparison Summary ==="
echo ""

# Compare generated file sizes
echo "Generated FIRRTL file sizes:"
echo "Counter example:"
if [ -f counter_static.fir ]; then
    echo "  Static:       $(wc -l < counter_static.fir) lines"
fi
if [ -f counter_dynamic.fir ]; then
    echo "  Dynamic:      $(wc -l < counter_dynamic.fir) lines"
fi
if [ -f counter_most_dynamic.fir ]; then
    echo "  Most-Dynamic: $(wc -l < counter_most_dynamic.fir) lines"
fi
echo ""
echo "Nested modules example:"
if [ -f nested_static.fir ]; then
    echo "  Static:       $(wc -l < nested_static.fir) lines"
fi
if [ -f nested_dynamic.fir ]; then
    echo "  Dynamic:      $(wc -l < nested_dynamic.fir) lines"
fi
if [ -f nested_most_dynamic.fir ]; then
    echo "  Most-Dynamic: $(wc -l < nested_most_dynamic.fir) lines"
fi
echo ""
echo "Datapath example (action methods with return values):"
if [ -f datapath_static.fir ]; then
    echo "  Static:       $(wc -l < datapath_static.fir) lines"
fi
if [ -f datapath_dynamic.fir ]; then
    echo "  Dynamic:      $(wc -l < datapath_dynamic.fir) lines"
fi
if [ -f datapath_most_dynamic.fir ]; then
    echo "  Most-Dynamic: $(wc -l < datapath_most_dynamic.fir) lines"
fi
echo ""

echo "Generated Verilog file sizes:"
if [ -f counter_static.v ]; then
    echo "Static:       $(wc -l < counter_static.v) lines"
fi
if [ -f counter_dynamic.v ]; then
    echo "Dynamic:      $(wc -l < counter_dynamic.v) lines"
fi
if [ -f counter_most_dynamic.v ]; then
    echo "Most-Dynamic: $(wc -l < counter_most_dynamic.v) lines"
fi
echo ""

# Compare will-fire signal counts
echo "Will-fire signal comparison:"
if [ -f counter_static.fir ]; then
    echo "Static:       $(grep -c "_wf" counter_static.fir) will-fire signals"
fi
if [ -f counter_dynamic.fir ]; then
    echo "Dynamic:      $(grep -c "_wf" counter_dynamic.fir) will-fire signals"
fi
if [ -f counter_most_dynamic.fir ]; then
    echo "Most-Dynamic: $(grep -c "_wf" counter_most_dynamic.fir) will-fire signals"
fi
echo ""

echo "Hardware complexity comparison:"
if [ -f counter_static.fir ]; then
    echo "Static:       $(grep -c "firrtl.when" counter_static.fir) when blocks, $(grep -c "firrtl.and\|firrtl.or\|firrtl.not" counter_static.fir) logic gates"
fi
if [ -f counter_dynamic.fir ]; then
    echo "Dynamic:      $(grep -c "firrtl.when" counter_dynamic.fir) when blocks, $(grep -c "firrtl.and\|firrtl.or\|firrtl.not" counter_dynamic.fir) logic gates"
fi
if [ -f counter_most_dynamic.fir ]; then
    echo "Most-Dynamic: $(grep -c "firrtl.when" counter_most_dynamic.fir) when blocks, $(grep -c "firrtl.and\|firrtl.or\|firrtl.not" counter_most_dynamic.fir) logic gates"
fi
echo ""

echo "Use cases for each timing mode:"
echo "- Static: Maximum safety, simple designs, early development"
echo "- Dynamic: Balanced optimization, production designs"
echo "- Most-Dynamic: Maximum performance, complex designs with many primitives"