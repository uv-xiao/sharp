//===- ScheduleValidation.cpp - Validate schedule operations -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the schedule validation pass for Sharp Txn modules.
// The pass ensures that schedules only contain actions (rules and action methods),
// not value methods, in accordance with Sharp's execution model.
//
//===----------------------------------------------------------------------===//

#include "sharp/Dialect/Txn/TxnOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sharp-schedule-validation"

namespace mlir {
namespace sharp {

using ::sharp::txn::ModuleOp;
using ::sharp::txn::ValueMethodOp;
using ::sharp::txn::ActionMethodOp;
using ::sharp::txn::RuleOp;
using ::sharp::txn::ScheduleOp;
namespace txn = ::sharp::txn;

#define GEN_PASS_DEF_SCHEDULEVALIDATION
#include "sharp/Analysis/Passes.h.inc"

namespace {

class ScheduleValidationPass
    : public impl::ScheduleValidationBase<ScheduleValidationPass> {
public:
  void runOnOperation() override;

private:
  /// Validate all schedules in a module
  LogicalResult validateModule(txn::ModuleOp module);
  
  /// Validate a single schedule operation
  LogicalResult validateSchedule(txn::ScheduleOp schedule, txn::ModuleOp module);
};

void ScheduleValidationPass::runOnOperation() {
  auto moduleOp = getOperation();
  
  // Process each txn module
  moduleOp.walk([&](txn::ModuleOp txnModule) {
    if (failed(validateModule(txnModule))) {
      signalPassFailure();
    }
  });
}

LogicalResult ScheduleValidationPass::validateModule(txn::ModuleOp module) {
  LLVM_DEBUG(llvm::dbgs() << "Validating schedules in module: " 
             << module.getSymName() << "\n");
  
  // Find the schedule operation
  auto scheduleOp = module.getSchedule();
  if (!scheduleOp) {
    // No schedule is valid (module has no actions)
    return success();
  }
  
  return validateSchedule(scheduleOp, module);
}

LogicalResult ScheduleValidationPass::validateSchedule(txn::ScheduleOp schedule,
                                                       txn::ModuleOp module) {
  auto actionNames = schedule.getActionNames();
  bool hasErrors = false;
  
  for (const auto& actionRef : actionNames) {
    auto actionName = actionRef.cast<FlatSymbolRefAttr>().getValue();
    
    // Find the operation with this name
    Operation *op = nullptr;
    module.walk([&](Operation *candidate) {
      if (auto symbolOp = dyn_cast<SymbolOpInterface>(candidate)) {
        if (symbolOp.getNameAttr() == actionName) {
          op = candidate;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    
    if (!op) {
      schedule.emitError() << "scheduled action '" << actionName 
                          << "' not found in module";
      hasErrors = true;
      continue;
    }
    
    // Check if it's a value method (which should not be scheduled)
    if (isa<ValueMethodOp>(op)) {
      schedule.emitError() << "value method '" << actionName 
                          << "' cannot be in schedule - only actions "
                          << "(rules and action methods) are schedulable";
      hasErrors = true;
      continue;
    }
    
    // Verify it's either a rule or action method
    if (!isa<RuleOp>(op) && !isa<ActionMethodOp>(op)) {
      schedule.emitError() << "scheduled item '" << actionName 
                          << "' is neither a rule nor an action method";
      hasErrors = true;
    }
  }
  
  return hasErrors ? failure() : success();
}

} // namespace

std::unique_ptr<mlir::Pass> createScheduleValidationPass() {
  return std::make_unique<ScheduleValidationPass>();
}

} // namespace sharp
} // namespace mlir