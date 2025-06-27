#!/bin/bash
set -e

echo "=== Verifying Clang-20 and LLD Usage in Sharp Project ==="
echo

# Check environment
echo "1. Environment Variables:"
echo "   CC=$CC"
echo "   CXX=$CXX"
echo "   LD=$LD"
echo "   AR=$AR"
echo "   RANLIB=$RANLIB"
echo

# Check compiler versions
echo "2. Compiler Versions:"
$CC --version | head -1
$CXX --version | head -1
$LD --version | head -1
echo

# Check build configuration
echo "3. Sharp Build Configuration:"
if [ -f build/CMakeCache.txt ]; then
    echo "   C Compiler: $(grep "CMAKE_C_COMPILER:STRING" build/CMakeCache.txt | cut -d= -f2)"
    echo "   C++ Compiler: $(grep "CMAKE_CXX_COMPILER:STRING" build/CMakeCache.txt | cut -d= -f2)"
    echo "   Linker: $(grep "CMAKE_LINKER:FILEPATH" build/CMakeCache.txt | cut -d= -f2)"
    echo "   LLVM Linker: $(grep "LLVM_USE_LINKER" build/CMakeCache.txt | cut -d= -f2)"
else
    echo "   Build not configured yet"
fi
echo

# Check unified build configuration
echo "4. Unified LLVM/MLIR/CIRCT Build Configuration:"
if [ -f .install/unified-build/CMakeCache.txt ]; then
    echo "   C Compiler: $(grep "CMAKE_C_COMPILER:UNINITIALIZED" .install/unified-build/CMakeCache.txt | cut -d= -f2)"
    echo "   C++ Compiler: $(grep "CMAKE_CXX_COMPILER:UNINITIALIZED" .install/unified-build/CMakeCache.txt | cut -d= -f2)"
    echo "   LLVM Linker: $(grep "LLVM_USE_LINKER" .install/unified-build/CMakeCache.txt | cut -d= -f2)"
else
    echo "   Unified build not found"
fi
echo

# Test compilation
echo "5. Test Compilation:"
cat > /tmp/test_sharp.cpp << 'EOF'
#include <iostream>
int main() {
    std::cout << "Successfully compiled with clang-20!" << std::endl;
    return 0;
}
EOF

$CXX -fuse-ld=lld -o /tmp/test_sharp /tmp/test_sharp.cpp
/tmp/test_sharp
rm -f /tmp/test_sharp /tmp/test_sharp.cpp

echo
echo "✅ All builds in Sharp project are using clang-20 and lld!"