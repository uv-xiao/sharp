//===- InitAllDialects.h - Sharp Dialect Registration ----------*- C++ -*-===//
//
// Part of the Sharp Project
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to register all Sharp dialects.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_INITALL_DIALECTS_H
#define SHARP_INITALL_DIALECTS_H

#include "sharp/Dialect/Txn/TxnDialect.h"
#include "sharp/Simulation/SimulationDialect.h"
#include "mlir/IR/Dialect.h"

namespace sharp {

inline void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<sharp::txn::TxnDialect>();
  registry.insert<sharp::simulation::SharpSimulationDialect>();
}

} // namespace sharp

#endif // SHARP_INITALL_DIALECTS_H