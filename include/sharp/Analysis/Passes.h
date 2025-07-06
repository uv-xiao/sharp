//===- Passes.h - Sharp Analysis passes ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Sharp analysis passes.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_ANALYSIS_PASSES_H
#define SHARP_ANALYSIS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;

namespace sharp {

/// Create a pass to infer and complete conflict matrices for txn modules.
std::unique_ptr<mlir::Pass> createConflictMatrixInferencePass();

/// Create a pass to check for non-synthesizable constructs.
std::unique_ptr<mlir::Pass> createPreSynthesisCheckPass();

/// Create a pass to compute reachability conditions for method calls.
std::unique_ptr<mlir::Pass> createReachabilityAnalysisPass();

/// Create a pass to detect combinational loops in txn modules.
std::unique_ptr<mlir::Pass> createCombinationalLoopDetectionPass();

/// Create a pass to validate method attributes for FIRRTL translation.
std::unique_ptr<mlir::Pass> createMethodAttributeValidationPass();

/// Create a pass to complete partial schedules to minimize conflicts.
std::unique_ptr<mlir::Pass> createActionSchedulingPass();

/// Create a pass to validate that schedules only contain actions.
std::unique_ptr<mlir::Pass> createScheduleValidationPass();

/// Create a pass to check that value methods are conflict-free with all actions.
std::unique_ptr<mlir::Pass> createValueMethodConflictCheckPass();

/// Create a pass to validate that actions do not call other actions in the same module.
std::unique_ptr<mlir::Pass> createActionCallValidationPass();

/// Generate the code for registering analysis passes.
#define GEN_PASS_REGISTRATION
#include "sharp/Analysis/Passes.h.inc"

} // namespace sharp
} // namespace mlir

#endif // SHARP_ANALYSIS_PASSES_H