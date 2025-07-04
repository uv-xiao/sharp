//===- SimulationOps.h - Sharp Simulation Operations -------------*- C++ -*-===//
//
// This file defines the operations for the Sharp Simulation dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_SIMULATION_SIMULATIONOPS_H
#define SHARP_SIMULATION_SIMULATIONOPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "sharp/Simulation/SimulationDialect.h"

#define GET_OP_CLASSES
#include "sharp/Simulation/SimulationOps.h.inc"

#endif // SHARP_SIMULATION_SIMULATIONOPS_H