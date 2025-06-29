add_mlir_dialect_library(SharpAnalysis
  ConflictMatrixInference.cpp

  ADDITIONAL_HEADER_DIRS
  ${SHARP_MAIN_INCLUDE_DIR}/sharp/Analysis

  DEPENDS
  SharpAnalysisPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  SharpTxn
  )