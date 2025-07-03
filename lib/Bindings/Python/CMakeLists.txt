################################################################################
# Set up Python binding tools
################################################################################

include(AddMLIRPython)

# Don't set package prefix - we'll handle the structure ourselves

################################################################################
# Declare Python sources
################################################################################

declare_mlir_python_sources(SharpBindingsPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    __init__.py
    pysharp.py
)

################################################################################
# Declare native Python extension
################################################################################

declare_mlir_python_extension(SharpBindingsPythonExtension
  MODULE_NAME _sharp
  ADD_TO_PARENT SharpBindingsPythonSources
  SOURCES
    SharpModule.cpp
    SharpModules.h
  EMBED_CAPI_LINK_LIBS
    SharpCAPICore
    MLIRCAPIIR
    MLIRCAPITransforms
    # MLIR dialects
    MLIRCAPISCF
    MLIRCAPISMT
    MLIRCAPIIndex
    MLIRCAPIArith
    # CIRCT dialects
    CIRCTCAPIFIRRTL
    CIRCTCAPIComb
    CIRCTCAPIHWArith
    CIRCTCAPISeq
    CIRCTCAPISV
  PRIVATE_LINK_LIBS
    LLVMSupport
  PYTHON_BINDINGS_LIBRARY
    nanobind
)

add_dependencies(SharpBindingsPythonExtension sharp-headers)

################################################################################
# Declare dialect-specific bindings.
################################################################################

# Ensure the build directory for generated Python files exists.
file(MAKE_DIRECTORY ${SHARP_BINARY_DIR}/lib/Bindings/Python/sharp/dialects)

declare_mlir_python_sources(SharpBindingsPythonSources.Dialects
  ADD_TO_PARENT SharpBindingsPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharpBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/TxnOps.td
  SOURCES
    dialects/__init__.py
    dialects/txn.py
  DIALECT_NAME txn)

################################################################################
# Build composite binaries - Just build the Sharp extension
################################################################################

# Build Sharp Python modules
add_mlir_python_modules(SharpPythonModules
  ROOT_PREFIX "${SHARP_PYTHON_PACKAGES_DIR}/sharp"
  INSTALL_PREFIX "python_packages/sharp"
  DECLARED_SOURCES
    SharpBindingsPythonSources
  COMMON_CAPI_LINK_LIBS
    SharpBindingsPythonExtension
    SharpCAPICore
)