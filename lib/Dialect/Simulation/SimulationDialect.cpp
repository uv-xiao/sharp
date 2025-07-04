//===- SimulationDialect.cpp - Sharp Simulation Dialect ------------------===//
//
// This file implements the Sharp Simulation dialect.
//
//===----------------------------------------------------------------------===//

#include "sharp/Simulation/SimulationOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace sharp::simulation;

//===----------------------------------------------------------------------===//
// Simulation Dialect
//===----------------------------------------------------------------------===//

#include "sharp/Simulation/SimulationDialect.cpp.inc"

void SharpSimulationDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "sharp/Simulation/SimulationOps.cpp.inc"
      >();
}