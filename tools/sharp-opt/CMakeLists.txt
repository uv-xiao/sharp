get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# Build sharp-opt executable
add_llvm_executable(sharp-opt
  sharp-opt.cpp
)

# Add to default build targets
set_target_properties(sharp-opt PROPERTIES FOLDER "Tools")

# Link required libraries
target_link_libraries(sharp-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  SharpCore
  MLIROptLib
  MLIRSupport
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
)

# Make it part of the default build
add_custom_target(sharp-opt-build ALL DEPENDS sharp-opt)