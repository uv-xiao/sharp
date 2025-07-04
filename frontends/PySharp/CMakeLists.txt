# ===- CMakeLists.txt - PySharp CMake file -----------------------*- cmake -*-===//
#
# PySharp frontend following PyCDE pattern
#
# ===--------------------------------------------------------------------------===//

add_subdirectory(src)

# Integration tests
if(SHARP_BINDINGS_PYTHON_ENABLED AND SHARP_INCLUDE_TESTS)
  add_subdirectory(test)
  add_subdirectory(integration_test EXCLUDE_FROM_ALL)
endif()