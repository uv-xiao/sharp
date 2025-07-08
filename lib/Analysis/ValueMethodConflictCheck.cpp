//===- ValueMethodConflictCheck.cpp - Check value method conflicts -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the value method conflict checking pass for Sharp Txn
// modules. The pass ensures that value methods are conflict-free with all
// actions, as required by Sharp's execution model.
//
//===----------------------------------------------------------------------===//

#include "sharp/Dialect/Txn/TxnOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sharp-value-method-conflict-check"

namespace mlir {
namespace sharp {

using ::sharp::txn::ModuleOp;
using ::sharp::txn::ValueMethodOp;
using ::sharp::txn::ActionMethodOp;
using ::sharp::txn::RuleOp;
using ::sharp::txn::ScheduleOp;
using ::sharp::txn::ConflictRelation;
namespace txn = ::sharp::txn;

#define GEN_PASS_DEF_VALUEMETHODCONFLICTCHECK
#include "sharp/Analysis/Passes.h.inc"

namespace {

class ValueMethodConflictCheckPass
    : public impl::ValueMethodConflictCheckBase<ValueMethodConflictCheckPass> {
public:
  void runOnOperation() override;

private:
  /// Check all value methods in a module
  LogicalResult checkModule(txn::ModuleOp module);
  
  /// Check a single value method for conflicts
  LogicalResult checkValueMethod(ValueMethodOp valueMethod, 
                                txn::ModuleOp module,
                                const llvm::StringMap<ConflictRelation> &conflictMatrix);
  
  /// Parse conflict matrix from schedule operation
  llvm::StringMap<ConflictRelation> parseConflictMatrix(ScheduleOp schedule);
  
  /// Get the string representation of a conflict relation
  StringRef conflictToString(ConflictRelation rel);
};

void ValueMethodConflictCheckPass::runOnOperation() {
  auto moduleOp = getOperation();
  
  // Process each txn module
  moduleOp.walk([&](txn::ModuleOp txnModule) {
    if (failed(checkModule(txnModule))) {
      signalPassFailure();
    }
  });
}

LogicalResult ValueMethodConflictCheckPass::checkModule(txn::ModuleOp module) {
  LLVM_DEBUG(llvm::dbgs() << "Checking value method conflicts in module: " 
             << module.getSymName() << "\n");
  
  // Get the schedule and conflict matrix
  SmallVector<txn::ScheduleOp> schedules;
  module.walk([&](txn::ScheduleOp schedule) {
    schedules.push_back(schedule);
  });
  
  if (schedules.empty()) {
    // No schedule means no conflicts to check
    return success();
  }
  
  // Use the first schedule's conflict matrix
  auto conflictMatrix = parseConflictMatrix(schedules[0]);
  
  // Check each value method
  bool hasErrors = false;
  module.walk([&](ValueMethodOp valueMethod) {
    if (failed(checkValueMethod(valueMethod, module, conflictMatrix))) {
      hasErrors = true;
    }
  });
  
  return hasErrors ? failure() : success();
}

LogicalResult ValueMethodConflictCheckPass::checkValueMethod(
    ValueMethodOp valueMethod,
    txn::ModuleOp module,
    const llvm::StringMap<ConflictRelation> &conflictMatrix) {
  
  auto methodName = valueMethod.getSymName();
  LLVM_DEBUG(llvm::dbgs() << "  Checking value method: " << methodName << "\n");
  
  // Get all actions (rules and action methods) in the module
  SmallVector<StringRef> actions;
  
  module.walk([&](Operation *op) {
    if (isa<RuleOp>(op) || isa<ActionMethodOp>(op)) {
      if (auto symbolOp = dyn_cast<SymbolOpInterface>(op)) {
        actions.push_back(symbolOp.getNameAttr().getValue());
      }
    }
  });
  
  // Check conflicts with each action
  bool hasConflict = false;
  for (auto action : actions) {
    // Check both directions of the conflict relationship
    auto checkConflict = [&](StringRef first, StringRef second) {
      auto key = (first + "," + second).str();
      auto it = conflictMatrix.find(key);
      if (it != conflictMatrix.end() && it->second != txn::ConflictRelation::ConflictFree) {
        valueMethod.emitError() 
            << "value method '" << methodName 
            << "' has non-CF conflict with action '" << action
            << "' (" << conflictToString(it->second) << ")";
        // Note: value methods must be conflict-free (CF) with all actions and other value methods
        hasConflict = true;
        return true;
      }
      return false;
    };
    
    // Check both orderings
    if (checkConflict(methodName, action) || 
        checkConflict(action, methodName)) {
      break; // Report only the first conflict found
    }
  }
  
  // Also check if this value method appears in any non-CF relation
  for (const auto &entry : conflictMatrix) {
    if (entry.second != txn::ConflictRelation::ConflictFree) {
      // Split the key to get the two method names
      auto key = entry.first();
      auto comma = key.find(',');
      if (comma != StringRef::npos) {
        auto first = key.substr(0, comma);
        auto second = key.substr(comma + 1);
        
        if (first == methodName || second == methodName) {
          // Only report if we haven't already reported this conflict
          if (!hasConflict) {
            valueMethod.emitError()
                << "value method '" << methodName 
                << "' appears in conflict matrix with non-CF relation ("
                << conflictToString(entry.second) << ")";
            // Note: value methods must only have conflict-free (CF) relationships
            hasConflict = true;
          }
        }
      }
    }
  }
  
  return hasConflict ? failure() : success();
}

llvm::StringMap<ConflictRelation> 
ValueMethodConflictCheckPass::parseConflictMatrix(ScheduleOp schedule) {
  llvm::StringMap<ConflictRelation> result;
  
  auto cmAttr = schedule.getConflictMatrixAttr();
  if (!cmAttr) {
    return result;
  }
  
  auto dict = cast<DictionaryAttr>(cmAttr);
  for (auto entry : dict) {
    auto key = entry.getName().str();
    if (auto conflictAttr = dyn_cast<IntegerAttr>(entry.getValue())) {
      result[key] = static_cast<txn::ConflictRelation>(conflictAttr.getInt());
    }
  }
  
  return result;
}

StringRef ValueMethodConflictCheckPass::conflictToString(ConflictRelation rel) {
  switch (rel) {
    case txn::ConflictRelation::SequenceBefore: return "SB (Sequence Before)";
    case txn::ConflictRelation::SequenceAfter: return "SA (Sequence After)";
    case txn::ConflictRelation::Conflict: return "C (Conflict)";
    case txn::ConflictRelation::ConflictFree: return "CF (Conflict-Free)";
  }
}

} // namespace

std::unique_ptr<mlir::Pass> createValueMethodConflictCheckPass() {
  return std::make_unique<ValueMethodConflictCheckPass>();
}

} // namespace sharp
} // namespace mlir