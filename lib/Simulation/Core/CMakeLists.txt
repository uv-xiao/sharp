add_mlir_library(SharpSimulationCore
  Event.cpp
  SimModule.cpp
  Simulator.cpp

  ADDITIONAL_HEADER_DIRS
  ${SHARP_MAIN_INCLUDE_DIR}/sharp/Simulation

  DEPENDS
  SharpSimulationIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  )