#!/bin/bash
# Script to run PySharp integration tests

set -e

echo "Running PySharp integration tests..."
echo "================================="

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run each test
for test in $SCRIPT_DIR/test_*.py; do
    echo "Running $(basename $test)..."
    python $test
    echo ""
done

echo "================================="
echo "All PySharp integration tests completed!"