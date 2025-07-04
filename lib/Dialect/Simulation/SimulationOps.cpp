//===- SimulationOps.cpp - Sharp Simulation Operations -------------------===//
//
// This file implements the operations for the Sharp Simulation dialect.
//
//===----------------------------------------------------------------------===//

#include "sharp/Simulation/SimulationOps.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace sharp::simulation;

//===----------------------------------------------------------------------===//
// SimulationOps
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "sharp/Simulation/SimulationOps.cpp.inc"