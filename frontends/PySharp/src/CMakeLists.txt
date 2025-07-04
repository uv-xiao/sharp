# ===- CMakeLists.txt - PySharp sources --------------------------*- cmake -*-===//
#
# PySharp Python sources following PyCDE pattern
#
# ===-----------------------------------------------------------------------===//

include(AddMLIRPython)

# Set the package prefix for Sharp's Python bindings
# This makes the bindings available as pysharp.sharp.* (similar to pycde.circt.*)
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=pysharp.sharp.")

# Declare PySharp Python sources
declare_mlir_python_sources(PySharpSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
  pysharp/__init__.py
  pysharp/types.py
  pysharp/module.py
  pysharp/signals.py
  pysharp/common.py
  pysharp/builder.py
  pysharp/dialects/__init__.py
  pysharp/dialects/txn.py
  pysharp/dialects/arith.py
  pysharp/dialects/comb.py
  pysharp/dialects/hw.py
  pysharp/dialects/seq.py
  pysharp/support.py
)

################################################################################
# Build composite binaries
################################################################################
set(PYSHARP_PYTHON_PACKAGE_DIR "${SHARP_PYTHON_PACKAGES_DIR}/pysharp/")

# Bundle our own, self-contained CAPI library with all of our deps
add_mlir_python_common_capi_library(PySharp_SharpPythonCAPI
  INSTALL_COMPONENT SharpBindingsPythonModules
  INSTALL_DESTINATION python_packages/pysharp/sharp/_mlir_libs
  OUTPUT_DIRECTORY "${PYSHARP_PYTHON_PACKAGE_DIR}/pysharp/sharp/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_SOURCES
    MLIRPythonSources.Core
    SharpBindingsPythonExtension
)

# Create Python modules for Sharp bindings (this creates the .sharp import)
add_mlir_python_modules(PySharp_SharpPythonModules
  ROOT_PREFIX "${PYSHARP_PYTHON_PACKAGE_DIR}/pysharp/sharp"
  INSTALL_PREFIX "python_packages/pysharp/sharp"
  DECLARED_SOURCES
    MLIRPythonSources.Core
    SharpBindingsPythonExtension
    SharpBindingsPythonSources
    CIRCTBindingsPythonSources
  COMMON_CAPI_LINK_LIBS
    PySharp_SharpPythonCAPI
)

# Create main PySharp Python module
add_mlir_python_modules(PySharp
  ROOT_PREFIX "${PYSHARP_PYTHON_PACKAGE_DIR}/"
  INSTALL_PREFIX "python_packages/"
  DECLARED_SOURCES
    PySharpSources
  COMMON_CAPI_LINK_LIBS
    PySharp_SharpPythonCAPI
)

# Install the CAPI library
install(TARGETS PySharp_SharpPythonCAPI
  DESTINATION python_packages/pysharp/sharp/_mlir_libs
  RUNTIME_DEPENDENCIES
    PRE_EXCLUDE_REGEXES ".*"
    PRE_INCLUDE_REGEXES ".*zlib.*"
  COMPONENT PySharp
)

# Add dependencies
add_dependencies(PySharp PySharp_SharpPythonModules)
add_dependencies(install-PySharp install-PySharp_SharpPythonModules)