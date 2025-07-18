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

/// Create a unified pass for general semantic validation of Sharp Txn modules.
std::unique_ptr<mlir::Pass> createGeneralCheckPass();

/// Create a pass to infer and complete conflict matrices for txn modules.
std::unique_ptr<mlir::Pass> createConflictMatrixInferencePass();

/// Create a comprehensive pass to check for non-synthesizable constructs and method attributes.
std::unique_ptr<mlir::Pass> createPreSynthesisCheckPass();

/// Create a pass to compute reachability conditions for method calls.
std::unique_ptr<mlir::Pass> createReachabilityAnalysisPass();

/// Create a pass to complete partial schedules to minimize conflicts.
std::unique_ptr<mlir::Pass> createActionSchedulingPass();

// Consolidated passes - individual functions removed:
// createMethodAttributeValidationPass() - functionality moved to PreSynthesisCheck
// createScheduleValidationPass() - functionality moved to GeneralCheck
// createValueMethodConflictCheckPass() - functionality moved to GeneralCheck
// createActionCallValidationPass() - functionality moved to GeneralCheck

/// Create a pass to collect primitive action calls for each action.
std::unique_ptr<mlir::Pass> createCollectPrimitiveActionsPass();

/// Create a pass to inline txn.func calls within txn modules.
std::unique_ptr<mlir::Pass> createInlineFunctionsPass();

/// Create a pass to generate missing primitive definitions.
std::unique_ptr<mlir::Pass> createPrimitiveGenPass();

/// Generate the code for registering analysis passes.
#define GEN_PASS_REGISTRATION
#include "sharp/Analysis/Passes.h.inc"

} // namespace sharp
} // namespace mlir

#endif // SHARP_ANALYSIS_PASSES_H