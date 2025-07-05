#!/bin/bash

SHARP_ROOT="../../.."
SHARP_OPT="$SHARP_ROOT/build/bin/sharp-opt"

echo "=== Chapter 7: Python Frontend ==="
echo ""

echo "1. Simple Counter Generation:"
echo "----------------------------------------"
python3 counter.py > counter_generated.mlir
if [ -f counter_generated.mlir ]; then
    echo "✅ Counter modules generated"
    echo "Validating generated MLIR..."
    if $SHARP_OPT counter_generated.mlir > /dev/null 2>&1; then
        echo "✅ Generated MLIR is valid"
    else
        echo "❌ Generated MLIR has errors"
    fi
else
    echo "❌ Counter generation failed"
fi
echo ""

echo "2. Pipeline Generator:"
echo "----------------------------------------"
python3 pipeline_gen.py
if [ -f pipeline_3x32.mlir ] && [ -f pipeline_5x16.mlir ]; then
    echo "✅ Pipeline variants generated"
    echo "  - pipeline_3x32.mlir (3 stages, 32-bit)"
    echo "  - pipeline_5x16.mlir (5 stages, 16-bit)"
else
    echo "❌ Pipeline generation failed"
fi
echo ""

echo "3. Systolic Array Generation:"
echo "----------------------------------------"
python3 matrix_mult.py > systolic_arrays.log
if [ $? -eq 0 ]; then
    echo "✅ Systolic arrays generated"
    cat systolic_arrays.log
else
    echo "❌ Systolic array generation failed"
fi
echo ""

echo "4. Advanced Features Demo:"
echo "----------------------------------------"
python3 advanced_features.py
if [ $? -eq 0 ]; then
    echo "✅ Advanced modules generated"
    echo "  - Parameterized FIFOs"
    echo "  - FFT stages"
    echo "  - Complex arithmetic"
else
    echo "❌ Advanced feature generation failed"
fi
echo ""

echo "5. Integration Example:"
echo "----------------------------------------"
# Show how Python-generated modules can be simulated
if [ -f counter_generated.mlir ]; then
    $SHARP_ROOT/tools/generate-workspace.sh counter_generated.mlir counter_sim
    if [ -d counter_sim ]; then
        echo "✅ Python-generated module ready for simulation"
        echo "Build with: cd counter_sim && mkdir build && cd build && cmake .. && make"
    fi
fi