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

#include "sharp/Dialect/Core/CoreDialect.h"
#include "mlir/IR/Dialect.h"

namespace sharp {

inline void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<sharp::core::CoreDialect>();
}

} // namespace sharp

#endif // SHARP_INITALL_DIALECTS_H