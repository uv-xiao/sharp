//===- GeneralCheck.cpp - Sharp General Semantic Validation Pass --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the General Check pass, which combines multiple semantic
// validation passes for Sharp Txn modules. This pass validates core execution
// model constraints that apply to all Sharp code, regardless of target.
//
//===----------------------------------------------------------------------===//

#include "sharp/Analysis/Passes.h"
#include "sharp/Analysis/AnalysisError.h"
#include "sharp/Dialect/Txn/TxnDialect.h"
#include "sharp/Dialect/Txn/TxnOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sharp-general-check"

using namespace mlir;

namespace mlir {
namespace sharp {

#define GEN_PASS_DEF_GENERALCHECK
#include "sharp/Analysis/Passes.h.inc"

namespace {

/// General semantic validation pass that combines multiple validation checks
/// for Sharp Txn modules. This pass validates core execution model constraints
/// that apply to all Sharp code.
struct GeneralCheck : public impl::GeneralCheckBase<GeneralCheck> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    
    // Report pass execution
    LLVM_DEBUG(llvm::dbgs() << "[GeneralCheck] Starting general semantic validation pass\n");
    
    // Check dependencies: both ConflictMatrixInference and ReachabilityAnalysis must have completed
    if (!moduleOp->hasAttr("sharp.conflict_matrix_inferred")) {
      AnalysisError(moduleOp, "GeneralCheck")
        .setCategory(ErrorCategory::MissingDependency)
        .setDetails("sharp-infer-conflict-matrix must be run before sharp-general-check")
        .setReason("General validation requires complete conflict matrices to verify scheduling constraints")
        .setSolution("Please run sharp-infer-conflict-matrix first to ensure all conflict relationships are properly inferred")
        .emit();
      signalPassFailure();
      return;
    }
    
    if (!moduleOp->hasAttr("sharp.reachability_analyzed")) {
      AnalysisError(moduleOp, "GeneralCheck")
        .setCategory(ErrorCategory::MissingDependency)
        .setDetails("sharp-reachability-analysis must be run before sharp-general-check")
        .setReason("General validation requires reachability analysis to verify method call conditions")
        .setSolution("Please run sharp-reachability-analysis first to ensure all method calls have proper reachability conditions")
        .emit();
      signalPassFailure();
      return;
    }
    
    // Run all general semantic validation checks
    if (failed(validateScheduleConstraints(moduleOp)) ||
        failed(validateScheduleCompleteness(moduleOp)) ||
        failed(validateValueMethodConstraints(moduleOp)) ||
        failed(validateActionCallConstraints(moduleOp))) {
      signalPassFailure();
      return;
    }
    
    // Mark module as having passed general semantic validation
    moduleOp->setAttr("sharp.general_checked", 
                      UnitAttr::get(moduleOp.getContext()));
    
    LLVM_DEBUG(llvm::dbgs() << "[GeneralCheck] General semantic validation completed successfully\n");
  }

private:
  /// Validate that schedules only contain actions (not value methods)
  LogicalResult validateScheduleConstraints(ModuleOp moduleOp) {
    for (auto txnModule : moduleOp.getOps<::sharp::txn::ModuleOp>()) {
      for (auto schedule : txnModule.getOps<::sharp::txn::ScheduleOp>()) {
        for (auto actionRef : schedule.getActions()) {
          auto actionName = cast<FlatSymbolRefAttr>(actionRef).getValue();
          
          // Look up the referenced operation in the module
          auto referencedOp = txnModule.lookupSymbol(actionName);
          if (!referencedOp) {
            return AnalysisError(schedule, "GeneralCheck")
                   .setCategory(ErrorCategory::ValidationFailure)
                   .setDetails("schedule in module '" + txnModule.getName().str() + "' references unknown operation '" + actionName.str() + "'")
                   .setReason("The referenced symbol does not exist in the module")
                   .setSolution("Please ensure all scheduled items are properly defined operations")
                   .emit(), failure();
          }
          
          // Check that it's an action (rule or action method), not a value method
          if (!isa<::sharp::txn::RuleOp, ::sharp::txn::ActionMethodOp>(referencedOp)) {
            if (isa<::sharp::txn::ValueMethodOp>(referencedOp)) {
              return AnalysisError(schedule, "GeneralCheck")
                     .setCategory(ErrorCategory::ValidationFailure)
                     .setDetails("schedule in module '" + txnModule.getName().str() + "' contains value method '" + actionName.str() + "'")
                     .setReason("Value methods are computed automatically once per cycle and cannot be scheduled")
                     .setSolution("Only actions (rules and action methods) can appear in schedules")
                     .emit(), failure();
            } else {
              return AnalysisError(schedule, "GeneralCheck")
                     .setCategory(ErrorCategory::ValidationFailure)
                     .setDetails("schedule in module '" + txnModule.getName().str() + "' contains non-action operation '" + actionName.str() + "' of type " + referencedOp->getName().getStringRef().str())
                     .setReason("Only actions (rules and action methods) can be scheduled")
                     .setSolution("Please remove non-action operations from the schedule")
                     .emit(), failure();
            }
          }
        }
      }
    }
    return success();
  }
  
  /// Validate that schedules contain all actions (completeness check)
  LogicalResult validateScheduleCompleteness(ModuleOp moduleOp) {
    for (auto txnModule : moduleOp.getOps<::sharp::txn::ModuleOp>()) {
      // Collect all actions (rules and action methods) in this module
      llvm::SmallSet<StringRef, 16> allActions;
      for (auto rule : txnModule.getOps<::sharp::txn::RuleOp>()) {
        allActions.insert(rule.getSymName());
      }
      for (auto actionMethod : txnModule.getOps<::sharp::txn::ActionMethodOp>()) {
        allActions.insert(actionMethod.getSymName());
      }
      
      // If no actions exist, skip this module
      if (allActions.empty()) {
        continue;
      }
      
      // Get scheduled actions from the schedule
      auto scheduleOps = txnModule.getOps<::sharp::txn::ScheduleOp>();
      if (scheduleOps.empty()) {
        // No schedule exists but actions are present
        return AnalysisError(txnModule, "GeneralCheck")
               .setCategory(ErrorCategory::ValidationFailure)
               .setDetails("module '" + txnModule.getName().str() + "' has actions but no schedule operation")
               .setReason("Schedules must include ALL actions (rules and action methods) in the module")
               .setSolution("Please add a txn.schedule operation that includes all actions, or use sharp-action-scheduling to automatically generate a complete schedule")
               .emit(), failure();
      }
      
      auto schedule = *scheduleOps.begin();
      llvm::SmallSet<StringRef, 16> scheduledActions;
      
      // Collect all actions listed in the schedule
      for (auto actionRef : schedule.getActions()) {
        auto actionName = cast<FlatSymbolRefAttr>(actionRef).getValue();
        scheduledActions.insert(actionName);
      }
      
      // Check for missing actions
      llvm::SmallVector<StringRef> missingActions;
      for (StringRef actionName : allActions) {
        if (!scheduledActions.contains(actionName)) {
          missingActions.push_back(actionName);
        }
      }
      
      // Check for extra actions (already covered by validateScheduleConstraints but good to be explicit)
      llvm::SmallVector<StringRef> extraActions;
      for (StringRef scheduledAction : scheduledActions) {
        if (!allActions.contains(scheduledAction)) {
          extraActions.push_back(scheduledAction);
        }
      }
      
      if (!missingActions.empty()) {
        std::string missingList;
        for (size_t i = 0; i < missingActions.size(); ++i) {
          if (i > 0) missingList += ", ";
          missingList += missingActions[i].str();
        }
        
        return AnalysisError(schedule, "GeneralCheck")
               .setCategory(ErrorCategory::ValidationFailure)
               .setDetails("schedule in module '" + txnModule.getName().str() + "' is missing " + std::to_string(missingActions.size()) + " action(s): [" + missingList + "]")
               .setReason("Schedules must include ALL actions (rules and action methods) in the module. Incomplete schedules lead to unscheduled actions that cannot execute")
               .setSolution("Please add the missing actions to the schedule, or use sharp-action-scheduling to automatically generate a complete schedule")
               .emit(), failure();
      }
      
      if (!extraActions.empty()) {
        std::string extraList;
        for (size_t i = 0; i < extraActions.size(); ++i) {
          if (i > 0) extraList += ", ";
          extraList += extraActions[i].str();
        }
        
        return schedule.emitError("[GeneralCheck] Schedule completeness validation failed - invalid references")
               << ": schedule in module '" << txnModule.getName() << "' references " 
               << extraActions.size() << " non-existent action(s): [" << extraList << "] at " << schedule.getLoc() << ". "
               << "Reason: Schedules can only reference actions that actually exist in the module. "
               << "Please remove the invalid references or define the missing actions.";
      }
    }
    return success();
  }
  
  /// Validate that value methods are conflict-free with all actions
  LogicalResult validateValueMethodConstraints(ModuleOp moduleOp) {
    for (auto txnModule : moduleOp.getOps<::sharp::txn::ModuleOp>()) {
      // Get the conflict matrix from the schedule
      auto scheduleOps = txnModule.getOps<::sharp::txn::ScheduleOp>();
      if (scheduleOps.empty()) {
        continue; // No schedule, no conflicts to check
      }
      
      auto schedule = *scheduleOps.begin();
      auto conflictMatrix = schedule.getConflictMatrix();
      if (!conflictMatrix) {
        continue; // No conflict matrix, assume all CF
      }
      
      // Check each value method
      for (auto valueMethod : txnModule.getOps<::sharp::txn::ValueMethodOp>()) {
        StringRef methodName = valueMethod.getSymName();
        
        // Check conflicts with all scheduled actions
        for (auto actionRef : schedule.getActions()) {
          auto actionName = cast<FlatSymbolRefAttr>(actionRef).getValue();
          
          // Look for conflict entries
          std::string forwardKey = (methodName + "," + actionName).str();
          std::string reverseKey = (actionName + "," + methodName).str();
          
          auto checkConflict = [&](StringRef key) -> LogicalResult {
            if (conflictMatrix && conflictMatrix->contains(key)) {
              auto conflictAttr = conflictMatrix->get(key);
              auto conflictValue = cast<IntegerAttr>(conflictAttr).getInt();
              // 0=SB, 1=SA, 2=C, 3=CF - only CF (3) is allowed for value methods
              if (conflictValue != 3) { // Not CF
                const char* conflictNames[] = {"SB", "SA", "C", "CF", "UNK"};
                return valueMethod.emitError("[GeneralCheck] Value method constraint validation failed - conflict violation")
                       << ": value method '" << methodName << "' in module '" << txnModule.getName()
                       << "' has " << (conflictValue < 5 ? conflictNames[conflictValue] : "unknown")
                       << " (" << conflictValue << ") conflict with action '" << actionName << "' at "
                       << valueMethod.getLoc() << ". "
                       << "Reason: Value methods must be conflict-free (CF) with all scheduled actions "
                       << "because they are computed automatically once per cycle. "
                       << "Please ensure the value method does not interfere with any scheduled actions.";
              }
            }
            return success();
          };
          
          if (failed(checkConflict(forwardKey)) || failed(checkConflict(reverseKey))) {
            return failure();
          }
        }
      }
    }
    return success();
  }
  
  /// Validate that actions don't call other actions in the same module
  LogicalResult validateActionCallConstraints(ModuleOp moduleOp) {
    for (auto txnModule : moduleOp.getOps<::sharp::txn::ModuleOp>()) {
      // Collect all actions in this module
      llvm::SmallSet<StringRef, 16> actions;
      for (auto rule : txnModule.getOps<::sharp::txn::RuleOp>()) {
        actions.insert(rule.getSymName());
      }
      for (auto actionMethod : txnModule.getOps<::sharp::txn::ActionMethodOp>()) {
        actions.insert(actionMethod.getSymName());
      }
      
      // Check each action for invalid calls
      auto checkActionBody = [&](Operation* action, Region& body) -> LogicalResult {
        auto result = body.walk([&](::sharp::txn::CallOp callOp) -> WalkResult {
          auto callee = callOp.getCallee();
          
          // Check for direct calls to actions in the same module
          if (auto directRef = dyn_cast<FlatSymbolRefAttr>(callee)) {
            StringRef calledName = directRef.getValue();
            if (actions.contains(calledName)) {
              callOp.emitError("[GeneralCheck] Action call validation failed - invalid call target")
                     << ": action '" << action->getAttr("sym_name") << "' in module '" 
                     << txnModule.getName() << "' calls another action '" << calledName 
                     << "' at " << callOp.getLoc() << ". "
                     << "Reason: Actions cannot call other actions within the same module "
                     << "because this would create recursive dependencies in the execution model. "
                     << "Actions can only call value methods in the same module or methods of child module instances.";
              return WalkResult::interrupt();
            }
          }
          
          return WalkResult::advance();
        });
        
        return result.wasInterrupted() ? failure() : success();
      };
      
      // Check all rules
      for (auto rule : txnModule.getOps<::sharp::txn::RuleOp>()) {
        if (failed(checkActionBody(rule, rule.getBody()))) {
          return failure();
        }
      }
      
      // Check all action methods
      for (auto actionMethod : txnModule.getOps<::sharp::txn::ActionMethodOp>()) {
        if (failed(checkActionBody(actionMethod, actionMethod.getBody()))) {
          return failure();
        }
      }
    }
    return success();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createGeneralCheckPass() {
  return std::make_unique<GeneralCheck>();
}

} // namespace sharp
} // namespace mlir