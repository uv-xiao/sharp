add_mlir_library(SharpTxnToFIRRTL
  TxnToFIRRTLPass.cpp
  LowerOpToFIRRTLPass.cpp
  TranslateTxnToFIRRTLPass.cpp

  DEPENDS
  SharpConversionPassesIncGen
  
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  MLIRArithDialect
  SharpTxn
  SharpTxnPrimitives
  SharpAnalysis
  CIRCTFIRRTL
  )