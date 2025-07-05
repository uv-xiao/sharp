#!/bin/bash

SHARP_ROOT="../../.."
SHARP_OPT="$SHARP_ROOT/build/bin/sharp-opt"

echo "=== Chapter 4: Analysis Passes ==="
echo ""

echo "1. Testing conflict matrix inference:"
echo "----------------------------------------"
$SHARP_OPT complex_module.mlir --sharp-infer-conflict-matrix 2>&1 | grep -A 20 "conflict_matrix"
echo ""

echo "2. Testing combinational loop detection:"
echo "----------------------------------------"
if $SHARP_OPT loop_example.mlir --sharp-detect-combinational-loops 2>&1 | grep -q "error"; then
    echo "✅ Loop detected as expected"
else
    echo "❌ Loop detection failed"
fi
echo ""

echo "3. Testing pre-synthesis check on valid module:"
echo "----------------------------------------"
if $SHARP_OPT complex_module.mlir --sharp-pre-synthesis-check 2>&1 | grep -q "error"; then
    echo "❌ Unexpected synthesis errors"
else
    echo "✅ Module is synthesizable"
fi
echo ""

echo "4. Testing complete analysis pipeline:"
echo "----------------------------------------"
$SHARP_OPT complex_module.mlir --sharp-infer-conflict-matrix --sharp-pre-synthesis-check > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Valid module passes all checks"
else
    echo "❌ Analysis failed"
fi