#!/bin/bash
set -e

# Run cmake configure and filter out known warnings
cmake_output=$(cmake "$@" 2>&1)
exit_code=$?

# Filter the output
echo "$cmake_output" | grep -v "Cannot generate a safe runtime search path" | \
                       grep -v "runtime library.*may be hidden by files in" | \
                       grep -v "Some of these libraries may not be found correctly" | \
                       grep -v "Call Stack (most recent call first)" | \
                       grep -v "add_library.*AddLLVM.cmake" | \
                       grep -v "add_executable.*AddLLVM.cmake" | \
                       grep -v "add_.*_library.*AddMLIR.cmake" | \
                       grep -v "add_.*_tool.*CMakeLists.txt" | \
                       grep -v "files in some directories may conflict" | \
                       grep -v "CMake Deprecation Warning" | \
                       grep -v "The OLD behavior for policy" | \
                       grep -v "The cmake-policies.* manual explains" | \
                       grep -v "policies are deprecated" | \
                       grep -v "specific short-term circumstances" | \
                       grep -v "behavior and not rely on setting" | \
                       grep -v "^$" | \
                       sed '/^$/N;/^\n$/d'  # Remove multiple blank lines

exit $exit_code