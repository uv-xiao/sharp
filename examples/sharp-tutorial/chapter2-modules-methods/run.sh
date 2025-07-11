#!/bin/bash

SHARP_ROOT="../../.."
SHARP_OPT="$SHARP_ROOT/build/bin/sharp-opt"

echo "=== Chapter 2: Modules and Methods ==="
echo ""

echo "1. Parsing and verifying counter.mlir:"
$SHARP_OPT counter.mlir > /dev/null && echo "✅ Parse successful" || echo "❌ Parse failed"
echo ""

echo "2. Running conflict matrix inference:"
echo "----------------------------------------"
echo "..."
$SHARP_OPT counter.mlir --sharp-primitive-gen --sharp-infer-conflict-matrix | grep -A 20 "schedule"
echo ""