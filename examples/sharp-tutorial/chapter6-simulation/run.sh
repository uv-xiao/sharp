#!/bin/bash

SHARP_ROOT="../../.."
SHARP_OPT="$SHARP_ROOT/build/bin/sharp-opt"

echo "=== Chapter 6: Simulation Modes ==="
echo ""

echo "1. Transaction-Level Simulation:"
echo "----------------------------------------"
$SHARP_ROOT/tools/generate-workspace.sh pipeline.mlir pipeline_tl
cd pipeline_tl && mkdir -p build && cd build && cmake .. > /dev/null 2>&1 && make > /dev/null 2>&1
if [ -f Pipeline_sim ]; then
    echo "✅ TL simulation built"
    echo "Running 5 cycles:"
    ./Pipeline_sim --cycles 5 --stats
else
    echo "❌ TL simulation build failed"
fi
cd ../../..
echo ""

echo "2. RTL Preparation (Arcilator):"
echo "----------------------------------------"
if $SHARP_OPT counter_rtl.mlir --sharp-arcilator > counter_arc.mlir 2>&1; then
    echo "✅ Arc conversion successful"
    echo "Generated $(wc -l < counter_arc.mlir) lines of Arc IR"
else
    echo "❌ Arc conversion failed"
fi
echo ""

echo "3. JIT Compilation Test:"
echo "----------------------------------------"
# JIT has limited support currently
echo "Testing JIT mode availability..."
if $SHARP_OPT counter_rtl.mlir --sharp-simulate="mode=jit" 2>&1 | grep -q "error"; then
    echo "⚠️  JIT mode has limited support"
else
    echo "✅ JIT mode available"
fi
echo ""

echo "4. Performance Comparison Setup:"
echo "----------------------------------------"
$SHARP_ROOT/tools/generate-workspace.sh perf_test.mlir perf_sim
if [ -d perf_sim ]; then
    echo "✅ Performance test module ready"
    echo "Build with: cd perf_sim && mkdir build && cd build && cmake .. && make"
    echo "Run with: ./PerfTest_sim --cycles 1000000 --stats"
else
    echo "❌ Performance test setup failed"
fi