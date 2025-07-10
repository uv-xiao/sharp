//===- MethodAttributeValidation.cpp - Validate method attributes --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the method attribute validation pass for Sharp Txn
// modules. The pass validates attributes used in FIRRTL signal generation
// to ensure correctness during translation.
//
//===----------------------------------------------------------------------===//

#include "sharp/Dialect/Txn/TxnOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sharp-method-attribute-validation"

namespace mlir {
namespace sharp {

using ::sharp::txn::CallOp;
using ::sharp::txn::InstanceOp;
using ::sharp::txn::ValueMethodOp;
using ::sharp::txn::ActionMethodOp;
using ::sharp::txn::RuleOp;
using ::sharp::txn::ScheduleOp;
using ::sharp::txn::IfOp;
namespace txn = ::sharp::txn;

#define GEN_PASS_DEF_METHODATTRIBUTEVALIDATION
#include "sharp/Analysis/Passes.h.inc"

namespace {

class MethodAttributeValidationPass
    : public impl::MethodAttributeValidationBase<MethodAttributeValidationPass> {
public:
  void runOnOperation() override;

private:
  /// Validate all methods in a module
  LogicalResult validateModule(txn::ModuleOp module);
  
  /// Check for signal name conflicts
  LogicalResult checkNameConflicts(txn::ModuleOp module);
  
  /// Validate always_ready attribute
  LogicalResult validateAlwaysReady(ActionMethodOp method,
                                    txn::ModuleOp module);
  
  /// Validate always_enable attribute  
  LogicalResult validateAlwaysEnable(ActionMethodOp method);
  
  /// Get the method name
  std::string getMethodName(Operation *methodOp);
  
  /// Get the effective signal name for a method
  std::string getMethodSignalName(Operation *methodOp, StringRef postfix);
  
  /// Check if a method is always ready (no conflicts)
  bool isMethodAlwaysReady(ActionMethodOp method, txn::ModuleOp module);
  
  /// Check if all callers of a method are always enabled
  bool areAllCallersAlwaysEnabled(ActionMethodOp method,
                                   ModuleOp topModule);
};

void MethodAttributeValidationPass::runOnOperation() {
  auto module = getOperation();
  bool hasErrors = false;
  
  // Validate each txn module
  module.walk([&](txn::ModuleOp txnModule) {
    if (failed(validateModule(txnModule))) {
      hasErrors = true;
    }
  });
  
  if (hasErrors) {
    signalPassFailure();
  } else {
    llvm::outs() << "Method attribute validation passed\n";
  }
}

LogicalResult MethodAttributeValidationPass::validateModule(txn::ModuleOp module) {
  // Check for name conflicts
  if (failed(checkNameConflicts(module))) {
    return failure();
  }
  
  // Validate action method attributes
  auto result = success();
  module.walk([&](ActionMethodOp method) {
    // Validate always_ready
    if (method->hasAttr("always_ready")) {
      if (failed(validateAlwaysReady(method, module))) {
        result = failure();
      }
    }
    
    // Validate always_enable
    if (method->hasAttr("always_enable")) {
      if (failed(validateAlwaysEnable(method))) {
        result = failure();
      }
    }
  });
  
  return result;
}

LogicalResult MethodAttributeValidationPass::checkNameConflicts(
    txn::ModuleOp module) {
  StringSet<> usedNames;
  auto result = success();
  
  // Add module name
  if (!usedNames.insert(module.getSymName()).second) {
    module.emitError("Module name conflicts with existing name: ")
        << module.getSymName();
    result = failure();
  }

  
  // Check for conflicts with instance names
  module.walk([&](InstanceOp inst) {
    if (!usedNames.insert(inst.getSymName()).second) {
      inst.emitError("Instance name conflicts with existing name: ")
          << inst.getSymName();
      result = failure();
    }
  });
  
  // Check value methods
  module.walk([&](ValueMethodOp method) {
    // Get effective signal names
    std::string dataName = getMethodSignalName(method, "_OUT");
    
    if (!usedNames.insert(dataName).second) {
      method.emitError("Method signal name conflicts with existing name: ")
          << dataName;
      result = failure();
    }
  });
  
  // Check action methods
  module.walk([&](ActionMethodOp method) {
    std::string methodName = getMethodName(method);
    // Get effective signal names
    std::string dataName = getMethodSignalName(method, "_OUT");
    std::string enableName = getMethodSignalName(method, "_EN");
    std::string readyName = getMethodSignalName(method, "_RDY");

    if (!usedNames.insert(methodName).second) {
      method.emitError("Method name conflicts with existing name: ")
          << methodName;
      result = failure();
    }
    
    // Check data/result signal
    if (method.getNumArguments() > 0 || method.getNumResults() > 0) {
      if (!usedNames.insert(dataName).second) {
        method.emitError("Method signal name conflicts with existing name: ")
            << dataName;
        result = failure();
      }
    }
    
    // Check enable signal (unless always_enable)
    if (!method->hasAttr("always_enable")) {
      if (!usedNames.insert(enableName).second) {
        method.emitError("Method enable signal conflicts with existing name: ")
            << enableName;
        result = failure();
      }
    }
    
    // Check ready signal (unless always_ready)
    if (!method->hasAttr("always_ready")) {
      if (!usedNames.insert(readyName).second) {
        method.emitError("Method ready signal conflicts with existing name: ")
            << readyName;
        result = failure();
      }
    }
  });
  
  
  return result;
}

LogicalResult MethodAttributeValidationPass::validateAlwaysReady(
    ActionMethodOp method, txn::ModuleOp module) {
  if (!isMethodAlwaysReady(method, module)) {
    return method.emitError(
        "Method marked always_ready but has potential conflicts");
  }
  return success();
}

LogicalResult MethodAttributeValidationPass::validateAlwaysEnable(
    ActionMethodOp method) {
  ModuleOp topModule = method->getParentOfType<ModuleOp>();
  if (!areAllCallersAlwaysEnabled(method, topModule)) {
    return method.emitError(
        "Method marked always_enable but has conditional callers");
  }
  return success();
}

std::string MethodAttributeValidationPass::getMethodName(
   Operation *methodOp) {
  StringRef methodName = methodOp->getAttrOfType<StringAttr>("sym_name").getValue();
  if (auto prefixAttr = methodOp->getAttrOfType<StringAttr>("prefix")) {
    methodName = prefixAttr.getValue();
  }
  return methodName.str();
}


std::string MethodAttributeValidationPass::getMethodSignalName(
    Operation *methodOp, StringRef postfix) {
  StringRef methodName = methodOp->getAttrOfType<StringAttr>("sym_name").getValue();
  
  // Check for prefix attribute
  if (auto prefixAttr = methodOp->getAttrOfType<StringAttr>("prefix")) {
    methodName = prefixAttr.getValue();
  }
  
  // Check for custom postfix attributes
  if (postfix == "_OUT" && methodOp->hasAttr("result")) {
    postfix = methodOp->getAttrOfType<StringAttr>("result").getValue();
  } else if (postfix == "_EN" && methodOp->hasAttr("enable")) {
    postfix = methodOp->getAttrOfType<StringAttr>("enable").getValue();
  } else if (postfix == "_RDY" && methodOp->hasAttr("ready")) {
    postfix = methodOp->getAttrOfType<StringAttr>("ready").getValue();
  }
  
  return (methodName + postfix).str();
}

bool MethodAttributeValidationPass::isMethodAlwaysReady(
    ActionMethodOp method, txn::ModuleOp module) {
  // A method is always ready if it has no conflicts with any other action
  // This requires checking the conflict matrix
  
  // Find the schedule operation
  ScheduleOp schedule = nullptr;
  module.walk([&](ScheduleOp op) {
    schedule = op;
    return WalkResult::interrupt();
  });
  
  if (!schedule) {
    // No schedule means no conflicts
    return true;
  }
  
  // Check if this method appears in any conflict relationships
  StringRef methodName = method.getSymName();
  auto conflictMatrix = schedule.getConflictMatrix();
  
  if (!conflictMatrix) {
    // No conflict matrix means no conflicts
    return true;
  }
  
  // Check all conflict entries
  for (auto &entry : conflictMatrix.value()) {
    auto key = entry.getName().getValue();
    auto value = cast<IntegerAttr>(entry.getValue()).getInt();
    
    // Parse the key (format: "action1,action2")
    size_t commaPos = key.find(',');
    if (commaPos == StringRef::npos) continue;
    
    StringRef action1 = key.substr(0, commaPos);
    StringRef action2 = key.substr(commaPos + 1);
    
    // Check if this method is involved in any conflict (C) or ordering (SA/SB)
    if ((action1 == methodName || action2 == methodName) && value != 3) {
      // Not conflict-free
      return false;
    }
  }
  
  return true;
}

bool MethodAttributeValidationPass::areAllCallersAlwaysEnabled(
    ActionMethodOp method, ModuleOp topModule) {
  StringRef targetMethod = method.getSymName();
  txn::ModuleOp parentModule = method->getParentOfType<txn::ModuleOp>();
  StringRef parentModuleName = parentModule.getSymName();
  
  // Check all calls to this method
  bool foundConditionalCall = false;
  
  topModule.walk([&](CallOp call) {
    // Parse the callee to get the method name
    auto callee = call.getCallee();
    StringRef callMethod = callee.getLeafReference();
    if (callMethod != targetMethod) {
      return;
    }
    
    // Get instance name from callee
    StringRef callInstance;
    if (callee.getNestedReferences().size() == 1) {
      callInstance = callee.getRootReference();
    }
    
    // Check if the instance is of the right module type
    if (auto instOp = call->getParentOfType<txn::ModuleOp>()
                          .lookupSymbol<InstanceOp>(callInstance)) {
      if (instOp.getModuleName() != parentModuleName) {
        return;
      }
    }
    
    // Check if this call is inside a conditional (txn.if)
    Operation *parent = call->getParentOp();
    while (parent) {
      if (isa<IfOp>(parent)) {
        foundConditionalCall = true;
        return;
      }
      // Stop at action boundaries
      if (isa<RuleOp>(parent) || isa<ActionMethodOp>(parent) ||
          isa<ValueMethodOp>(parent)) {
        break;
      }
      parent = parent->getParentOp();
    }
  });
  
  return !foundConditionalCall;
}

} // namespace

std::unique_ptr<mlir::Pass> createMethodAttributeValidationPass() {
  return std::make_unique<MethodAttributeValidationPass>();
}

} // namespace sharp
} // namespace mlir