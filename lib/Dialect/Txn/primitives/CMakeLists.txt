add_mlir_dialect_library(SharpTxnPrimitives
  Register.cpp
  Wire.cpp

  ADDITIONAL_HEADER_DIRS
  ${SHARP_MAIN_INCLUDE_DIR}/sharp/Dialect/Txn

  DEPENDS
  MLIRTxnIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  SharpTxn
  CIRCTFIRRTL
  )