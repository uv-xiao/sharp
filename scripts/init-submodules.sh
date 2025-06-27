#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

INITIALIZED_SOMETHING=false

# Check if CIRCT submodule is already initialized
if [ -d "circt/.git" ] && [ -f "circt/CMakeLists.txt" ]; then
    : # CIRCT already initialized, do nothing
else
    echo "Initializing CIRCT submodule..."
    git submodule update --init circt
    INITIALIZED_SOMETHING=true
fi

# Check if LLVM submodule within CIRCT is already initialized
if [ -d "circt/llvm/.git" ] && [ -f "circt/llvm/CMakeLists.txt" ]; then
    : # LLVM already initialized, do nothing
else
    # Now handle CIRCT's submodules with proper shallow cloning
    cd "$PROJECT_ROOT/circt"
    
    # Check if CIRCT has .gitmodules
    if [ -f .gitmodules ]; then
        echo "Initializing CIRCT's LLVM submodule with shallow clone..."
        
        # Initialize LLVM submodule with shallow clone
        # This respects the shallow setting in CIRCT's .gitmodules
        git submodule update --init --depth 1 llvm
        INITIALIZED_SOMETHING=true
        
        # If there are other submodules in CIRCT that need to be initialized
        # they can be added here with appropriate depth settings
    fi
fi

# Only show status if something was initialized
if [ "$INITIALIZED_SOMETHING" = true ]; then
    echo "Submodules initialized successfully!"
    # Show status
    echo ""
    echo "Submodule status:"
    cd "$PROJECT_ROOT"
    git submodule status --recursive
fi