set(LLVM_OPTIONAL_SOURCES
  TxnSimulatePass.cpp
  ConcurrentSimulationPass.cpp
  ArcilatorIntegrationPass.cpp
  HybridSimulationPass.cpp
  PassRegistration.cpp
)

# Pass declarations are already generated in include/sharp/Simulation/CMakeLists.txt

add_mlir_library(SharpSimulationPasses
  TxnSimulatePass.cpp
  ConcurrentSimulationPass.cpp
  ArcilatorIntegrationPass.cpp
  HybridSimulationPass.cpp
  PassRegistration.cpp

  DEPENDS
  SharpSimulationPassesIncGen

  LINK_LIBS PUBLIC
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  MLIRExecutionEngine
  MLIRFuncToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRArithToLLVM
  MLIRTargetLLVM
  MLIRToLLVMIRTranslationRegistration
  SharpTxn
  SharpTxnToFunc
  SharpConversionPasses
  SharpSimulationDialect
  SharpSimulationCore
  SharpSimulationTL
  SharpSimulationConcurrent
  SharpSimulationHybrid
  CIRCTArc
  CIRCTArcTransforms
  CIRCTArcToLLVM
  CIRCTConvertToArcs
  CIRCTFIRRTLToHW
  CIRCTFIRRTL
  CIRCTHW
  CIRCTSeq
  CIRCTComb
)