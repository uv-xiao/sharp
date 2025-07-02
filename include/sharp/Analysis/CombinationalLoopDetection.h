//===- CombinationalLoopDetection.h - Detect combinational loops -*- C++ -*-===//
//
// Part of the Sharp Project
//
//===----------------------------------------------------------------------===//
//
// This file declares the combinational loop detection analysis pass for the
// Txn dialect. It detects cycles in combinational logic paths.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_ANALYSIS_COMBINATIONALLOOPDETECTION_H
#define SHARP_ANALYSIS_COMBINATIONALLOOPDETECTION_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace sharp {

/// Create a pass to detect combinational loops in Txn modules.
std::unique_ptr<Pass> createCombinationalLoopDetectionPass();

/// Register the pass
void registerCombinationalLoopDetectionPass();

} // namespace sharp
} // namespace mlir

#endif // SHARP_ANALYSIS_COMBINATIONALLOOPDETECTION_H