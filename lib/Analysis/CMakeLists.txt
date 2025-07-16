add_mlir_dialect_library(SharpAnalysis
  AnalysisError.cpp
  ConflictDebugger.cpp
  ScheduleDebugger.cpp
  GeneralCheck.cpp
  ConflictMatrixInference.cpp
  PreSynthesisCheck.cpp
  ReachabilityAnalysis.cpp
  # Individual passes now consolidated:
  # MethodAttributeValidation.cpp - merged into PreSynthesisCheck
  # ScheduleValidation.cpp - merged into GeneralCheck
  # ValueMethodConflictCheck.cpp - merged into GeneralCheck
  # ActionCallValidation.cpp - merged into GeneralCheck
  ActionScheduling.cpp
  CollectPrimitiveActions.cpp
  InlineFunctions.cpp
  PrimitiveGen.cpp

  ADDITIONAL_HEADER_DIRS
  ${SHARP_MAIN_INCLUDE_DIR}/sharp/Analysis

  DEPENDS
  SharpAnalysisPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  SharpTxn
  SharpTxnPrimitives
  MLIRArithDialect
  )