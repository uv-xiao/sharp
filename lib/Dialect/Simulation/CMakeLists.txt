add_mlir_dialect_library(SharpSimulationDialect
  SimulationDialect.cpp
  SimulationOps.cpp
  
  DEPENDS
  SharpSimulationOpsIncGen
  
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)