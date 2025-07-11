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
echo "..."
$SHARP_OPT toggle.mlir --sharp-primitive-gen --sharp-infer-conflict-matrix | grep -A 20 "schedule"
echo ""