add_mlir_dialect_library(SharpAnalysis
  ConflictMatrixInference.cpp
  PreSynthesisCheck.cpp
  ReachabilityAnalysis.cpp
  CombinationalLoopDetection.cpp
  MethodAttributeValidation.cpp
  ActionScheduling.cpp
  ScheduleValidation.cpp
  ValueMethodConflictCheck.cpp
  ActionCallValidation.cpp
  CollectPrimitiveActions.cpp
  InlineFunctions.cpp

  ADDITIONAL_HEADER_DIRS
  ${SHARP_MAIN_INCLUDE_DIR}/sharp/Analysis

  DEPENDS
  SharpAnalysisPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  SharpTxn
  MLIRArithDialect
  )