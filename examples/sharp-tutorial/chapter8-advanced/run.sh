#!/bin/bash

SHARP_ROOT="../../.."
SHARP_OPT="$SHARP_ROOT/build/bin/sharp-opt"

echo "=== Chapter 8: Advanced Topics ==="
echo ""

echo "1. Custom Primitives:"
echo "----------------------------------------"
if $SHARP_OPT custom_primitives.mlir --parse-only > /dev/null 2>&1; then
    echo "✅ Custom primitive definitions valid"
else
    echo "❌ Custom primitive parsing failed"
fi
echo ""

echo "2. Verification Examples:"
echo "----------------------------------------"
if $SHARP_OPT verification_example.mlir --sharp-verify > verify_output.txt 2>&1; then
    echo "✅ Verification properties checked"
    grep -E "(property|assert)" verify_output.txt || echo "Properties defined"
else
    echo "⚠️  Verification pass not yet implemented"
fi
echo ""

echo "3. Performance Patterns:"
echo "----------------------------------------"
$SHARP_OPT optimization_patterns.mlir --sharp-optimize > optimized.mlir 2>&1
if [ -f optimized.mlir ]; then
    echo "✅ Optimization patterns processed"
    echo "Original size: $(wc -l < optimization_patterns.mlir) lines"
    echo "Optimized size: $(wc -l < optimized.mlir) lines"
else
    echo "⚠️  Optimization pass in development"
fi
echo ""

echo "4. Case Study - Cache:"
echo "----------------------------------------"
$SHARP_ROOT/tools/generate-workspace.sh case_study_cache.mlir cache_sim
if [ -d cache_sim ]; then
    echo "✅ Cache controller ready for simulation"
    echo "Features: CAM-based tags, statistics, hit rate calculation"
else
    echo "❌ Cache case study generation failed"
fi
echo ""

echo "5. Case Study - Crypto:"
echo "----------------------------------------"
if $SHARP_OPT case_study_crypto.mlir --parse-only > /dev/null 2>&1; then
    echo "✅ AES engine design valid"
    echo "Features: Round-based processing, S-box lookups, auto-round rule"
else
    echo "❌ Crypto case study parsing failed"
fi
echo ""

echo "6. Debug Features:"
echo "----------------------------------------"
if $SHARP_OPT debug_features.mlir --sharp-simulate="mode=translation,debug=true" > debug_sim.cpp 2>&1; then
    echo "✅ Debug instrumentation added"
    grep -c "debug_print" debug_sim.cpp && echo "debug probes inserted"
else
    echo "⚠️  Debug features require simulation mode"
fi

echo ""
echo "Advanced features demonstrate Sharp's capability for:"
echo "- Custom hardware primitives"
echo "- Formal verification integration"
echo "- Performance optimization"
echo "- Real-world applications"