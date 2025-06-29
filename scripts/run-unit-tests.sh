#!/bin/bash
# Check if unit tests are available and run them

# Check if gtest libraries exist
if [ -f "$PIXI_PROJECT_ROOT/.install/unified/lib/libllvm_gtest.a" ] || [ -f "$PIXI_PROJECT_ROOT/.install/unified/lib/libllvm_gtest_main.a" ]; then
    echo "Running unit tests..."
    # Use the check-sharp-unit target which builds and runs tests
    cmake --build build --target check-sharp-unit
else
    echo "Unit tests not available - LLVM needs to be rebuilt with gtest support"
    echo ""
    echo "The build system has been updated to include gtest support."
    echo "To enable unit tests, run:"
    echo "  pixi run build    # This will automatically rebuild with gtest"
    echo ""
    echo "Note: The first rebuild will take some time as it needs to recompile LLVM with test support."
    echo ""
    echo "In the meantime, you can use:"
    echo "  pixi run test-lit    # Run lit tests (comprehensive coverage)"
fi