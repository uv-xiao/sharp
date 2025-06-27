#!/bin/bash
set -e

echo "Testing Sharp Python bindings..."

# Use the Pixi environment's Python
PYTHON_CMD="${PIXI_PROJECT_ROOT}/.pixi/envs/default/bin/python"

# Set up Python path
export PYTHONPATH="${PIXI_PROJECT_ROOT}/.install/unified/python_packages/mlir_core:${PIXI_PROJECT_ROOT}/build/python_packages/sharp_core:$PYTHONPATH"

# Run the test
$PYTHON_CMD test_sharp.py