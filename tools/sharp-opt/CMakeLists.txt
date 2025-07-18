get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# Build sharp-opt executable
add_llvm_executable(sharp-opt
  sharp-opt.cpp
)

# Add to default build targets
set_target_properties(sharp-opt PROPERTIES FOLDER "Tools")

# Get CIRCT libraries
get_property(circt_dialect_libs GLOBAL PROPERTY CIRCT_DIALECT_LIBS)
get_property(circt_conversion_libs GLOBAL PROPERTY CIRCT_CONVERSION_LIBS)

# Link required libraries
target_link_libraries(sharp-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${circt_dialect_libs}
  ${circt_conversion_libs}
  SharpTxn
  SharpAnalysis
  SharpConversionPasses
  SharpSimulationPasses
  SharpSimulationDialect
  MLIROptLib
  MLIRSupport
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRExecutionEngine
  CIRCTExportVerilog
  CIRCTFIRRTLToHW
  CIRCTSeqToSV
  CIRCTHWToSV
  CIRCTFIRRTLTransforms
  CIRCTSeqTransforms
  CIRCTSVTransforms
  CIRCTHWTransforms
  CIRCTArc
  CIRCTArcToLLVM
  CIRCTSupport
  LLVMX86CodeGen
  LLVMX86AsmParser
  LLVMX86Desc
  LLVMX86Info
)

# Make it part of the default build
add_custom_target(sharp-opt-build ALL DEPENDS sharp-opt)