add_subdirectory(TxnToFIRRTL)

add_mlir_library(SharpConversionPasses
  PassDetail.cpp

  DEPENDS
  SharpConversionPassesIncGen

  LINK_LIBS PUBLIC
  SharpTxnToFIRRTL
  )