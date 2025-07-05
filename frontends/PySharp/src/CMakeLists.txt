# ===- CMakeLists.txt - PySharp sources --------------------------*- cmake -*-===//
#
# PySharp Python sources following PyCDE pattern
#
# ===-----------------------------------------------------------------------===//

include(AddMLIRPython)

# Don't set a package prefix - the bindings will be available directly
# add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=pysharp.sharp.")

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
# Build PySharp as a pure Python package
################################################################################
set(PYSHARP_PYTHON_PACKAGE_DIR "${SHARP_PYTHON_PACKAGES_DIR}/pysharp/")

# Create main PySharp Python module (pure Python, no native extensions)
add_mlir_python_modules(PySharp
  ROOT_PREFIX "${PYSHARP_PYTHON_PACKAGE_DIR}/"
  INSTALL_PREFIX "python_packages/"
  DECLARED_SOURCES
    PySharpSources
)

# PySharp depends on Sharp bindings being available
add_dependencies(PySharp SharpPythonModules)
add_dependencies(install-PySharp install-SharpPythonModules)