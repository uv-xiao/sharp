#!/bin/bash
# Test PySharp basic functionality

set -e

echo "Testing PySharp basic functionality..."

# Run the basic test and check output
python /home/uvxiao/sharp/integration_test/pysharp/basic.py > output.txt

# Check expected outputs
if grep -q "i32" output.txt && \
   grep -q "uint<16>" output.txt && \
   grep -q "ConflictRelation.C=2" output.txt && \
   grep -q "Signal(x: i32)" output.txt && \
   grep -q "Signal((x + 10): i32)" output.txt && \
   grep -q "ModuleBuilder(Test)" output.txt; then
    echo "✅ Basic test passed!"
    rm output.txt
    exit 0
else
    echo "❌ Basic test failed!"
    echo "Output was:"
    cat output.txt
    rm output.txt
    exit 1
fi