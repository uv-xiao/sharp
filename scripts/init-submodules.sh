#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Initializing submodules..."

cd "$PROJECT_ROOT"

# Initialize CIRCT submodule (without recursion first)
echo "Initializing CIRCT submodule..."
git submodule update --init circt

# Now handle CIRCT's submodules with proper shallow cloning
cd "$PROJECT_ROOT/circt"

# Check if CIRCT has .gitmodules
if [ -f .gitmodules ]; then
    echo "Initializing CIRCT's LLVM submodule with shallow clone..."
    
    # Initialize LLVM submodule with shallow clone
    # This respects the shallow setting in CIRCT's .gitmodules
    git submodule update --init --depth 1 llvm
    
    # If there are other submodules in CIRCT that need to be initialized
    # they can be added here with appropriate depth settings
fi

echo "Submodules initialized successfully!"

# Show status
echo ""
echo "Submodule status:"
cd "$PROJECT_ROOT"
git submodule status --recursive