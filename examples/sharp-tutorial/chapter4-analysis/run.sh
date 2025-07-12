#!/bin/bash

SHARP_ROOT="../../.."
SHARP_OPT="$SHARP_ROOT/build/bin/sharp-opt"

echo "=== Chapter 4: Analysis Passes ==="
echo ""

echo "1. Testing primitive generation (required first):"
echo "----------------------------------------"
if $SHARP_OPT complex_module.mlir --sharp-primitive-gen > /dev/null 2>&1; then
    echo "✅ Primitive generation completed successfully"
else
    echo "❌ Primitive generation failed"
fi
echo ""

echo "2. Testing conflict matrix inference (requires primitive-gen):"
echo "----------------------------------------"
$SHARP_OPT complex_module.mlir --sharp-primitive-gen --sharp-infer-conflict-matrix 2>&1 | grep -A 20 "conflict_matrix"
echo ""

echo "3. Testing reachability analysis (requires primitive-gen):"
echo "----------------------------------------"
echo "Running reachability analysis to track conditional execution..."
$SHARP_OPT complex_module.mlir --sharp-primitive-gen --sharp-reachability-analysis 2>&1 | grep -A 5 -B 2 "if.*then" | head -15
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ Reachability analysis completed successfully"
else
    echo "❌ Reachability analysis failed"
fi
echo ""

echo "4. Testing general semantic validation (requires conflict-matrix + reachability):"
echo "----------------------------------------"
if $SHARP_OPT complex_module.mlir --sharp-primitive-gen --sharp-infer-conflict-matrix --sharp-reachability-analysis --sharp-general-check > /dev/null 2>&1; then
    echo "✅ Module passes general semantic validation"
else
    echo "❌ General semantic validation failed"
fi
echo ""

echo "5. Testing pre-synthesis check (requires general-check):"
echo "----------------------------------------"
if $SHARP_OPT complex_module.mlir --sharp-primitive-gen --sharp-infer-conflict-matrix --sharp-reachability-analysis --sharp-general-check --sharp-pre-synthesis-check 2>&1 | grep -q "error"; then
    echo "❌ Unexpected synthesis errors"
else
    echo "✅ Valid module is synthesizable"
fi
echo ""

echo "6. Testing dependency enforcement (should fail):"
echo "----------------------------------------"
echo "Testing general-check without dependencies (should fail):"
if $SHARP_OPT complex_module.mlir --sharp-general-check 2>&1 | grep -q "missing dependency"; then
    echo "✅ Dependency enforcement working correctly"
else
    echo "❌ Dependency enforcement failed"
fi
echo ""

echo "7. Testing pre-synthesis violations:"
echo "----------------------------------------"
echo "Testing disallowed operation violations:"
if $SHARP_OPT non_synthesizable.mlir --sharp-primitive-gen --sharp-infer-conflict-matrix --sharp-reachability-analysis --sharp-general-check --sharp-pre-synthesis-check 2>&1 | grep -q "disallowed operation"; then
    echo "✅ Non-synthesizable modules detected correctly"
else
    echo "❌ Failed to detect non-synthesizable modules"
fi
echo ""

echo "8. Testing schedule completeness validation:"
echo "----------------------------------------"
echo "Testing incomplete schedule detection (should fail):"
if $SHARP_OPT incomplete_schedule.mlir --sharp-primitive-gen --sharp-infer-conflict-matrix --sharp-reachability-analysis --sharp-general-check 2>&1 | grep -q "missing.*action"; then
    echo "✅ Incomplete schedule correctly detected by general-check"
else
    echo "❌ Failed to detect incomplete schedule"
fi

echo "Testing action-scheduling pass to fix incomplete schedule:"
if $SHARP_OPT incomplete_schedule.mlir --sharp-primitive-gen --sharp-infer-conflict-matrix --sharp-reachability-analysis --sharp-action-scheduling --sharp-general-check > /dev/null 2>&1; then
    echo "✅ Action-scheduling successfully fixed incomplete schedule"
else
    echo "❌ Action-scheduling failed to fix incomplete schedule"
fi
echo ""

echo "9. Testing complete analysis pipeline (correct order):"
echo "----------------------------------------"
$SHARP_OPT complex_module.mlir --sharp-primitive-gen --sharp-infer-conflict-matrix --sharp-reachability-analysis --sharp-general-check --sharp-pre-synthesis-check > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Valid module passes all analysis checks in correct order"
else
    echo "❌ Analysis pipeline failed"
fi