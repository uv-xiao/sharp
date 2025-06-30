//===- ReachabilityAnalysis.cpp - Compute method call reachability -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the reachability analysis pass for Sharp Txn modules.
// The pass computes reachability conditions for method calls within rules and
// action methods by tracking control flow through txn.if operations.
//
//===----------------------------------------------------------------------===//

#include "sharp/Dialect/Txn/TxnOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sharp-reachability-analysis"

namespace mlir {
namespace sharp {

using ::sharp::txn::CallOp;
using ::sharp::txn::RuleOp;
using ::sharp::txn::ActionMethodOp;
using ::sharp::txn::IfOp;
namespace txn = ::sharp::txn;

#define GEN_PASS_DEF_REACHABILITYANALYSIS
#include "sharp/Analysis/Passes.h.inc"

namespace {

/// Analysis state for tracking reachability conditions
struct ReachabilityState {
  /// Map from CallOp to its reachability condition
  DenseMap<Operation*, StringAttr> reachabilityConditions;
  
  /// Map from SSA values to their condition names
  DenseMap<Value, std::string> conditionNames;
  
  /// Counter for generating unique condition names
  unsigned conditionCounter = 0;
  
  /// Get a unique name for a condition
  std::string getConditionName(Value condition) {
    auto it = conditionNames.find(condition);
    if (it != conditionNames.end()) {
      return it->second;
    }
    
    std::string name = "cond_" + std::to_string(conditionCounter++);
    conditionNames[condition] = name;
    return name;
  }
};

class ReachabilityAnalysisPass
    : public impl::ReachabilityAnalysisBase<ReachabilityAnalysisPass> {
public:
  void runOnOperation() override;

private:
  /// Analyze reachability for a single action (rule or method)
  void analyzeAction(Operation *actionOp, ReachabilityState &state);
  
  /// Process a region with a given path condition
  void processRegion(Region &region, StringRef pathCondition,
                     ReachabilityState &state, OpBuilder &builder);
  
  /// Process an operation and its nested regions
  void processOperation(Operation *op, StringRef pathCondition,
                        ReachabilityState &state, OpBuilder &builder);
  
  /// Build the condition expression for reaching a point
  StringAttr buildConditionExpr(StringRef baseCondition, Value ifCondition,
                                bool negated, ReachabilityState &state,
                                OpBuilder &builder);
};

void ReachabilityAnalysisPass::runOnOperation() {
  auto module = getOperation();
  ReachabilityState state;
  
  
  // Process each txn module
  module.walk([&](txn::ModuleOp txnModule) {
    // Analyze each rule
    txnModule.walk([&](RuleOp rule) {
      analyzeAction(rule, state);
    });
    
    // Analyze each action method
    txnModule.walk([&](ActionMethodOp method) {
      analyzeAction(method, state);
    });
  });
  
  // Apply reachability conditions as attributes
  for (auto &[callOp, condition] : state.reachabilityConditions) {
    callOp->setAttr("reachability_condition", condition);
  }
  
  // If no conditions were found, might be an issue
  if (state.reachabilityConditions.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Warning: No reachability conditions found\n");
  }
  
  LLVM_DEBUG({
    llvm::dbgs() << "Reachability Analysis Results:\n";
    for (auto &[callOp, condition] : state.reachabilityConditions) {
      if (auto call = dyn_cast<CallOp>(callOp)) {
        llvm::dbgs() << "  Call to " << call.getCallee() << " reachable when: "
                     << condition.getValue() << "\n";
      }
    }
  });
}

void ReachabilityAnalysisPass::analyzeAction(Operation *actionOp,
                                              ReachabilityState &state) {
  MLIRContext *ctx = actionOp->getContext();
  OpBuilder builder(ctx);
  
  // Start with "true" condition (always reachable at action entry)
  StringRef initialCondition = "true";
  
  // Get the body region
  Region *bodyRegion = nullptr;
  if (auto rule = dyn_cast<RuleOp>(actionOp)) {
    bodyRegion = &rule.getBody();
    LLVM_DEBUG(llvm::dbgs() << "Analyzing rule: " << rule.getSymName() << "\n");
  } else if (auto method = dyn_cast<ActionMethodOp>(actionOp)) {
    bodyRegion = &method.getBody();
    LLVM_DEBUG(llvm::dbgs() << "Analyzing method: " << method.getSymName() << "\n");
  } else {
    return;
  }
  
  // Process the body region
  processRegion(*bodyRegion, initialCondition, state, builder);
}

void ReachabilityAnalysisPass::processRegion(Region &region,
                                              StringRef pathCondition,
                                              ReachabilityState &state,
                                              OpBuilder &builder) {
  for (auto &block : region) {
    for (auto &op : block) {
      processOperation(&op, pathCondition, state, builder);
    }
  }
}

void ReachabilityAnalysisPass::processOperation(Operation *op,
                                                 StringRef pathCondition,
                                                 ReachabilityState &state,
                                                 OpBuilder &builder) {
  // Handle txn.if operations
  if (auto ifOp = dyn_cast<IfOp>(op)) {
    // Build condition for then branch
    StringAttr thenCondition = buildConditionExpr(
        pathCondition, ifOp.getCondition(), false, state, builder);
    processRegion(ifOp.getThenRegion(), thenCondition.getValue(), state, builder);
    
    // Build condition for else branch (if exists)
    if (!ifOp.getElseRegion().empty()) {
      StringAttr elseCondition = buildConditionExpr(
          pathCondition, ifOp.getCondition(), true, state, builder);
      processRegion(ifOp.getElseRegion(), elseCondition.getValue(), state, builder);
    }
    
    // Don't process nested operations again
    return;
  }
  
  // Handle txn.call operations
  if (auto callOp = dyn_cast<CallOp>(op)) {
    // Record the reachability condition for this call
    state.reachabilityConditions[op] = StringAttr::get(op->getContext(), pathCondition);
    LLVM_DEBUG(llvm::dbgs() << "  Found call to " << callOp.getCallee() 
                            << " with condition: " << pathCondition << "\n");
    return;
  }
  
  // For other operations, process nested regions if any
  for (auto &region : op->getRegions()) {
    processRegion(region, pathCondition, state, builder);
  }
}

StringAttr ReachabilityAnalysisPass::buildConditionExpr(StringRef baseCondition,
                                                         Value ifCondition,
                                                         bool negated,
                                                         ReachabilityState &state,
                                                         OpBuilder &builder) {
  MLIRContext *ctx = builder.getContext();
  
  // Get the symbolic name for the if condition value
  std::string condName = state.getConditionName(ifCondition);
  
  // Build the expression string
  std::string expr;
  if (baseCondition == "true") {
    // Simplify: true && cond => cond, true && !cond => !cond
    expr = negated ? ("!" + condName) : condName;
  } else {
    // General case: base && cond or base && !cond
    expr = std::string(baseCondition) + " && ";
    expr += negated ? ("!" + condName) : condName;
  }
  
  return StringAttr::get(ctx, expr);
}

} // namespace

std::unique_ptr<mlir::Pass> createReachabilityAnalysisPass() {
  return std::make_unique<ReachabilityAnalysisPass>();
}

} // namespace sharp
} // namespace mlir