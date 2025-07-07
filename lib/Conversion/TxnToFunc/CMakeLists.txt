add_mlir_library(SharpTxnToFunc
  TxnToFuncPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${SHARP_MAIN_INCLUDE_DIR}/sharp/Conversion/TxnToFunc

  DEPENDS
  SharpConversionPassesIncGen

  LINK_LIBS PUBLIC
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRTransformUtils
  SharpTxn
  )

target_include_directories(obj.SharpTxnToFunc
  PUBLIC
  $<BUILD_INTERFACE:${SHARP_MAIN_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${SHARP_INCLUDE_DIR}>
  )