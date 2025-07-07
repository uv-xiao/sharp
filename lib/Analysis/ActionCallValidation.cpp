//===- ActionCallValidation.cpp - Validate action call restrictions ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the action call validation pass for Sharp Txn modules.
// The pass ensures that actions (rules and action methods) do not call other
// actions within the same module, as required by Sharp's execution model.
//
//===----------------------------------------------------------------------===//

#include "sharp/Dialect/Txn/TxnOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "sharp-action-call-validation"

namespace mlir {
namespace sharp {

using ::sharp::txn::ModuleOp;
using ::sharp::txn::ValueMethodOp;
using ::sharp::txn::ActionMethodOp;
using ::sharp::txn::RuleOp;
using ::sharp::txn::CallOp;
using ::sharp::txn::InstanceOp;
namespace txn = ::sharp::txn;

#define GEN_PASS_DEF_ACTIONCALLVALIDATION
#include "sharp/Analysis/Passes.h.inc"

namespace {

class ActionCallValidationPass
    : public impl::ActionCallValidationBase<ActionCallValidationPass> {
public:
  void runOnOperation() override;

private:
  /// Validate all actions in a module
  LogicalResult validateModule(txn::ModuleOp module);
  
  /// Validate calls within an action
  LogicalResult validateAction(Operation *action, txn::ModuleOp module);
  
  /// Check if a call is to an action in the same module
  bool isLocalActionCall(CallOp call, txn::ModuleOp module);
  
  /// Get the target method name from a call (handling both direct and instance calls)
  std::optional<StringRef> getCallTargetName(CallOp call);
};

void ActionCallValidationPass::runOnOperation() {
  auto moduleOp = getOperation();
  
  // Process each txn module
  moduleOp.walk([&](txn::ModuleOp txnModule) {
    if (failed(validateModule(txnModule))) {
      signalPassFailure();
    }
  });
}

LogicalResult ActionCallValidationPass::validateModule(txn::ModuleOp module) {
  LLVM_DEBUG(llvm::dbgs() << "Validating action calls in module: " 
             << module.getSymName() << "\n");
  
  bool hasErrors = false;
  
  // Check each rule
  module.walk([&](RuleOp rule) {
    if (failed(validateAction(rule, module))) {
      hasErrors = true;
    }
  });
  
  // Check each action method
  module.walk([&](ActionMethodOp actionMethod) {
    if (failed(validateAction(actionMethod, module))) {
      hasErrors = true;
    }
  });
  
  return hasErrors ? failure() : success();
}

LogicalResult ActionCallValidationPass::validateAction(Operation *action, 
                                                      txn::ModuleOp module) {
  auto actionName = cast<SymbolOpInterface>(action).getNameAttr().getValue();
  LLVM_DEBUG(llvm::dbgs() << "  Validating action: " << actionName << "\n");
  
  bool hasErrors = false;
  
  // Walk all calls within this action
  action->walk([&](CallOp call) {
    if (isLocalActionCall(call, module)) {
      auto targetName = getCallTargetName(call);
      call.emitError() << "action '" << actionName 
                      << "' cannot call action '" << targetName.value_or("<unknown>")
                      << "' in the same module";
      mlir::emitRemark(call.getLoc()) << "actions can only call value methods in the same module "
                                       << "or methods of child module instances";
      hasErrors = true;
    }
  });
  
  return hasErrors ? failure() : success();
}

bool ActionCallValidationPass::isLocalActionCall(CallOp call, 
                                                 txn::ModuleOp module) {
  // Get the callee - it's either a direct call or an instance method call
  auto callee = call.getCalleeAttr();
  
  // Check if it's a nested reference (instance method call)
  if (auto nestedRef = dyn_cast<SymbolRefAttr>(callee)) {
    if (!nestedRef.getNestedReferences().empty()) {
      // This is an instance method call (e.g., @instance::@method)
      // These are allowed
      return false;
    }
  }
  
  // It's a direct call - check if the target is an action in this module
  auto targetName = cast<FlatSymbolRefAttr>(callee).getValue();
  
  // Find the operation with this name in the module
  Operation *targetOp = nullptr;
  module.walk([&](Operation *op) {
    if (auto symbolOp = dyn_cast<SymbolOpInterface>(op)) {
      if (symbolOp.getNameAttr() == targetName) {
        targetOp = op;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  
  if (!targetOp) {
    // Target not found in this module - might be an error but not our concern
    return false;
  }
  
  // Check if the target is an action (rule or action method)
  return isa<RuleOp>(targetOp) || isa<ActionMethodOp>(targetOp);
}

std::optional<StringRef> ActionCallValidationPass::getCallTargetName(CallOp call) {
  auto callee = call.getCalleeAttr();
  
  if (auto flatRef = dyn_cast<FlatSymbolRefAttr>(callee)) {
    return flatRef.getValue();
  }
  
  if (auto nestedRef = dyn_cast<SymbolRefAttr>(callee)) {
    if (!nestedRef.getNestedReferences().empty()) {
      // For instance calls, return the method name
      return cast<FlatSymbolRefAttr>(nestedRef.getNestedReferences().back()).getValue();
    }
    return nestedRef.getRootReference().getValue();
  }
  
  return std::nullopt;
}

} // namespace

std::unique_ptr<mlir::Pass> createActionCallValidationPass() {
  return std::make_unique<ActionCallValidationPass>();
}

} // namespace sharp
} // namespace mlir