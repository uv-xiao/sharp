################################################################################
# Set up Python binding tools
################################################################################

include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=sharp.")

################################################################################
# Declare native Python extension
################################################################################

set(PYTHON_BINDINGS_SOURCES
  SharpModule.cpp
  # CoreModule.cpp
  # Headers must be included explicitly so they are installed.
  SharpModules.h
)

set(PYTHON_BINDINGS_LINK_LIBS
  SharpCAPICore
  MLIRCAPIIR
  # needed for mlirFrozenRewritePatternSetDestroy
  # but not the actual passes
  MLIRCAPITransforms
)

declare_mlir_python_extension(SharpBindingsPythonExtension
  MODULE_NAME _sharp
  SOURCES
    ${PYTHON_BINDINGS_SOURCES}
  EMBED_CAPI_LINK_LIBS
    ${PYTHON_BINDINGS_LINK_LIBS}
  PRIVATE_LINK_LIBS
    LLVMSupport
  PYTHON_BINDINGS_LIBRARY
    nanobind
)

add_dependencies(SharpBindingsPythonExtension sharp-headers)

################################################################################
# Declare Python sources
################################################################################

declare_mlir_python_sources(SharpBindingsPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    __init__.py
)

declare_mlir_python_sources(SharpBindingsPythonSources.Support
  ADD_TO_PARENT SharpBindingsPythonSources
  SOURCES
    support.py
)

################################################################################
# Declare dialect-specific bindings.
################################################################################

# Ensure the build directory for generated Python files exists. Ninja is able to
# generate this, but make does not and the build fails.
file(MAKE_DIRECTORY ${SHARP_BINARY_DIR}/lib/Bindings/Python/sharp/dialects)

declare_mlir_python_sources(SharpBindingsPythonSources.Dialects
  ADD_TO_PARENT SharpBindingsPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharpBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/SharpOps.td
  SOURCES
    dialects/sharp.py
  DIALECT_NAME sharp)

################################################################################
# Build composite binaries
################################################################################

# Bundle our own, self-contained CAPI library with all of our deps.
add_mlir_python_common_capi_library(SharpBindingsPythonCAPI
  INSTALL_COMPONENT SharpPythonModules
  INSTALL_DESTINATION python_packages/sharp_core/sharp/_mlir_libs
  OUTPUT_DIRECTORY "${SHARP_PYTHON_PACKAGES_DIR}/sharp_core/sharp/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MLIRPythonSources.Core
    SharpBindingsPythonExtension
)

# Bundle the Sharp python sources into our package.
add_mlir_python_modules(SharpPythonModules
  ROOT_PREFIX "${SHARP_PYTHON_PACKAGES_DIR}/sharp_core/sharp"
  INSTALL_PREFIX "python_packages/sharp_core/sharp"
  DECLARED_SOURCES
    MLIRPythonSources.Core
    SharpBindingsPythonExtension
    SharpBindingsPythonSources
  COMMON_CAPI_LINK_LIBS
    SharpBindingsPythonCAPI
)