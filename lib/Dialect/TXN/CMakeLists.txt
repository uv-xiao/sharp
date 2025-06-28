add_mlir_dialect_library(SharpTXN
  TXNDialect.cpp
  TXNOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${SHARP_MAIN_INCLUDE_DIR}/sharp/Dialect/TXN

  DEPENDS
  MLIRTXNIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  CIRCTComb
  CIRCTHW
  CIRCTSeq
  )