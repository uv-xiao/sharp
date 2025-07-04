//===- SimulationOps.td - Sharp Simulation Operations ---------------------===//
//
// Defines operations for Sharp's simulation framework.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_SIMULATION_OPS
#define SHARP_SIMULATION_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Sharp_SimulationDialect : Dialect {
  let name = "sharp.sim";
  let summary = "Sharp simulation dialect";
  let description = [{
    This dialect provides operations for configuring and controlling
    Sharp's multi-level simulation framework.
  }];
  
  let cppNamespace = "::sharp::simulation";
}

class Sim_Op<string mnemonic, list<Trait> traits = []> :
    Op<Sharp_SimulationDialect, mnemonic, traits>;

// TODO: Add operations later once the basic infrastructure is working

#endif // SHARP_SIMULATION_OPS