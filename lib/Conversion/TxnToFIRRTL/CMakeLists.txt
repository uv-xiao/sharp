add_mlir_library(SharpTxnToFIRRTL
  TxnToFIRRTLPass.cpp

  DEPENDS
  SharpConversionPassesIncGen
  
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  MLIRArithDialect
  SharpTxn
  CIRCTFIRRTL
  )