add_mlir_library(SharpSimulationTL
  SpecPrimitives.cpp

  ADDITIONAL_HEADER_DIRS
  ${SHARP_MAIN_INCLUDE_DIR}/sharp/Simulation


  LINK_LIBS PUBLIC
  SharpSimulationCore
  MLIRIR
  MLIRSupport
  )