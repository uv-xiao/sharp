# Generate simulation dialect declarations
set(LLVM_TARGET_DEFINITIONS SimulationOps.td)
mlir_tablegen(SimulationOps.h.inc -gen-op-decls)
mlir_tablegen(SimulationOps.cpp.inc -gen-op-defs)
mlir_tablegen(SimulationDialect.h.inc -gen-dialect-decls)
mlir_tablegen(SimulationDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(SharpSimulationOpsIncGen)

# Generate pass declarations
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name SharpSimulation)
add_public_tablegen_target(SharpSimulationPassesIncGen)