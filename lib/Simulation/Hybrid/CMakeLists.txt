# Hybrid simulation bridge components

add_mlir_library(SharpSimulationHybrid
  HybridBridge.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${SHARP_MAIN_INCLUDE_DIR}/sharp/Simulation
  
  LINK_LIBS PUBLIC
  SharpSimulationCore
  MLIRIR
  MLIRSupport
  LLVMSupport
)