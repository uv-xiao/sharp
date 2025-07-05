#!/bin/bash

SHARP_ROOT="../../.."
SHARP_OPT="$SHARP_ROOT/build/bin/sharp-opt"

echo "=== Chapter 1: Basic Concepts ==="
echo ""

echo "1. Parsing and verifying toggle.mlir:"
$SHARP_OPT toggle.mlir > /dev/null && echo "✅ Parse successful" || echo "❌ Parse failed"
echo ""

echo "2. Running conflict matrix inference:"
echo "----------------------------------------"
$SHARP_OPT toggle.mlir --sharp-infer-conflict-matrix | grep -A 20 "schedule"
echo ""

echo "3. Generating simulation workspace:"
$SHARP_ROOT/tools/generate-workspace.sh toggle.mlir toggle_sim

echo ""
echo "4. Building simulation:"
cd toggle_sim && mkdir -p build && cd build && cmake .. > /dev/null 2>&1 && make > /dev/null 2>&1
if [ -f Toggle_sim ]; then
    echo "✅ Build successful"
    echo ""
    echo "5. Running simulation:"
    echo "----------------------------------------"
    ./Toggle_sim --cycles 5 --verbose --stats
else
    echo "❌ Build failed"
fi