################################################################################
# Set up Python binding tools
################################################################################

include(AddMLIRPython)

# Set package prefix for Sharp bindings (top-level sharp package)
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=sharp.")

################################################################################
# Declare Python sources
################################################################################

declare_mlir_python_sources(SharpBindingsPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    __init__.py
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

# Bundle MLIR and Sharp into a common CAPI library
add_mlir_python_common_capi_library(SharpPythonCAPI
  INSTALL_COMPONENT SharpBindingsPythonModules
  INSTALL_DESTINATION python_packages/sharp/_mlir_libs
  OUTPUT_DIRECTORY "${SHARP_PYTHON_PACKAGES_DIR}/sharp/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../.."
  DECLARED_SOURCES
    MLIRPythonSources.Core
    SharpBindingsPythonExtension
)

# Build Sharp Python modules
add_mlir_python_modules(SharpPythonModules
  ROOT_PREFIX "${SHARP_PYTHON_PACKAGES_DIR}/sharp"
  INSTALL_PREFIX "python_packages/sharp"
  DECLARED_SOURCES
    MLIRPythonSources.Core
    SharpBindingsPythonSources
    SharpBindingsPythonExtension
  COMMON_CAPI_LINK_LIBS
    SharpPythonCAPI
)