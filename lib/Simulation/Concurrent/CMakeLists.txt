add_mlir_library(SharpSimulationConcurrent
  Context.cpp
  ConcurrentSimulator.cpp

  ADDITIONAL_HEADER_DIRS
  ${SHARP_MAIN_INCLUDE_DIR}/sharp/Simulation/Concurrent

  DEPENDS
  SharpSimulationCore

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  SharpSimulationCore
  )

target_include_directories(obj.SharpSimulationConcurrent
  PUBLIC
  $<BUILD_INTERFACE:${SHARP_MAIN_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${SHARP_INCLUDE_DIR}>
  )