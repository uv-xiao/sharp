add_mlir_dialect_library(SharpTxn
  TxnDialect.cpp
  TxnOps.cpp
  TxnAttrs.cpp

  ADDITIONAL_HEADER_DIRS
  ${SHARP_MAIN_INCLUDE_DIR}/sharp/Dialect/Txn

  DEPENDS
  MLIRTxnIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  CIRCTComb
  CIRCTHW
  CIRCTSeq
  )