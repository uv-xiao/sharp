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
// The pass adds reachability conditions as operands to txn.call operations,
// enabling the Txn-to-FIRRTL conversion to generate proper conflict_inside logic.
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
  /// Map from CallOp to its reachability condition value
  DenseMap<Operation*, Value> reachabilityConditions;
  
  /// Map to cache created condition values (key is condition value + negation flag as int)
  DenseMap<std::pair<Value, int>, Value> conditionCache;
  
  /// Counter for generating unique condition names
  unsigned conditionCounter = 0;
  
  /// Get a unique name for a condition
  std::string getConditionName() {
    return "reach_cond_" + std::to_string(conditionCounter++);
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
  void processRegion(Region &region, Value pathCondition,
                     ReachabilityState &state, OpBuilder &builder);
  
  /// Process an operation and its nested regions
  void processOperation(Operation *op, Value pathCondition,
                        ReachabilityState &state, OpBuilder &builder);
  
  /// Build the condition expression for reaching a point
  Value buildConditionExpr(Value baseCondition, Value ifCondition,
                          bool negated, ReachabilityState &state,
                          OpBuilder &builder);
  
  /// Update a CallOp to include its reachability condition
  void updateCallOp(CallOp callOp, Value condition, OpBuilder &builder);
  
  /// Update an AbortOp to include its reachability condition
  void updateAbortOp(txn::AbortOp abortOp, Value condition, OpBuilder &builder);
  
  /// Find a safe insertion point after the given value
  Block::iterator findInsertionPoint(Value value);
};

void ReachabilityAnalysisPass::runOnOperation() {
  auto module = getOperation();
  
  // Report pass execution
  LLVM_DEBUG(llvm::dbgs() << "[ReachabilityAnalysis] Starting reachability analysis pass\n");
  
  // Check dependency: PrimitiveGen must have completed
  if (!module->hasAttr("sharp.primitive_gen_complete")) {
    module.emitError("[ReachabilityAnalysis] Pass failed - missing dependency")
        << ": sharp-primitive-gen must be run before sharp-reachability-analysis. "
        << "Primitive definitions are needed to properly analyze method calls and their reachability. "
        << "Please run sharp-primitive-gen first to ensure all referenced primitives are available.";
    signalPassFailure();
    return;
  }
  
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
  
  // Update CallOps and AbortOps with their reachability conditions
  OpBuilder builder(module);
  for (auto &[op, condition] : state.reachabilityConditions) {
    if (auto call = dyn_cast<CallOp>(op)) {
      updateCallOp(call, condition, builder);
    } else if (auto abort = dyn_cast<txn::AbortOp>(op)) {
      updateAbortOp(abort, condition, builder);
    }
  }
  
  // Mark module as having completed reachability analysis
  module->setAttr("sharp.reachability_analyzed", 
                  UnitAttr::get(module.getContext()));
  
  LLVM_DEBUG(llvm::dbgs() << "[ReachabilityAnalysis] Reachability analysis completed successfully\n");
  
  LLVM_DEBUG({
    llvm::dbgs() << "Reachability Analysis Results:\n";
    for (auto &[callOp, condition] : state.reachabilityConditions) {
      if (auto call = dyn_cast<CallOp>(callOp)) {
        llvm::dbgs() << "  Call to " << call.getCallee() 
                     << " has condition: " << condition << "\n";
      }
    }
  });
}

void ReachabilityAnalysisPass::analyzeAction(Operation *actionOp,
                                              ReachabilityState &state) {
  MLIRContext *ctx = actionOp->getContext();
  OpBuilder builder(ctx);
  
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
  
  // For the initial condition, we use a placeholder that will be replaced
  // with "true" only when actually needed
  Value initialCondition = nullptr;
  
  // Process the body region
  processRegion(*bodyRegion, initialCondition, state, builder);
}

void ReachabilityAnalysisPass::processRegion(Region &region,
                                              Value pathCondition,
                                              ReachabilityState &state,
                                              OpBuilder &builder) {
  for (auto &block : region) {
    for (auto &op : block) {
      processOperation(&op, pathCondition, state, builder);
    }
  }
}

void ReachabilityAnalysisPass::processOperation(Operation *op,
                                                 Value pathCondition,
                                                 ReachabilityState &state,
                                                 OpBuilder &builder) {
  // Handle txn.if operations
  if (auto ifOp = dyn_cast<IfOp>(op)) {
    // Build condition for then branch
    Value thenCondition = buildConditionExpr(
        pathCondition, ifOp.getCondition(), false, state, builder);
    processRegion(ifOp.getThenRegion(), thenCondition, state, builder);
    
    // Build condition for else branch (if exists)
    if (!ifOp.getElseRegion().empty()) {
      Value elseCondition = buildConditionExpr(
          pathCondition, ifOp.getCondition(), true, state, builder);
      processRegion(ifOp.getElseRegion(), elseCondition, state, builder);
    }
    
    // Don't process nested operations again
    return;
  }
  
  // Handle txn.call operations
  if (auto callOp = dyn_cast<CallOp>(op)) {
    // Only record non-trivial reachability conditions
    if (pathCondition) {
      // Record the reachability condition for this call
      state.reachabilityConditions[op] = pathCondition;
      LLVM_DEBUG(llvm::dbgs() << "  Found call to " << callOp.getCallee() 
                              << " with condition: " << pathCondition << "\n");
    }
    return;
  }
  
  // Handle txn.abort operations similarly to calls
  if (auto abortOp = dyn_cast<txn::AbortOp>(op)) {
    // Record the reachability condition for this abort
    if (pathCondition) {
      state.reachabilityConditions[op] = pathCondition;
      LLVM_DEBUG(llvm::dbgs() << "  Found abort with condition: " << pathCondition << "\n");
    } else {
      // Even without a path condition, record that we found an abort
      // Use a true constant to indicate it's always reachable at the top level
      auto trueVal = builder.create<arith::ConstantIntOp>(op->getLoc(), 1, 1);
      state.reachabilityConditions[op] = trueVal;
      LLVM_DEBUG(llvm::dbgs() << "  Found unconditional abort\n");
    }
    return;
  }
  
  // For other operations, process nested regions if any
  for (auto &region : op->getRegions()) {
    processRegion(region, pathCondition, state, builder);
  }
}

Block::iterator ReachabilityAnalysisPass::findInsertionPoint(Value value) {
  // If the value is a block argument, insert at the beginning of the block
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    return blockArg.getOwner()->begin();
  }
  
  // Otherwise, insert after the defining operation
  Operation *defOp = value.getDefiningOp();
  assert(defOp && "Value must have a defining operation");
  
  // Find the operation in its block
  Block *block = defOp->getBlock();
  auto it = block->begin();
  while (&*it != defOp) {
    ++it;
  }
  
  // Insert after the defining operation
  return ++it;
}

Value ReachabilityAnalysisPass::buildConditionExpr(Value baseCondition,
                                                    Value ifCondition,
                                                    bool negated,
                                                    ReachabilityState &state,
                                                    OpBuilder &builder) {
  // Check cache first
  auto key = std::make_pair(ifCondition, negated ? 1 : 0);
  auto it = state.conditionCache.find(key);
  if (it != state.conditionCache.end()) {
    return it->second;
  }
  
  // Find a safe insertion point
  Block::iterator insertPt = findInsertionPoint(ifCondition);
  
  // If we have a base condition, we need to ensure we insert after it too
  if (baseCondition) {
    Block::iterator basePt = findInsertionPoint(baseCondition);
    Block *block = ifCondition.getDefiningOp() ? 
                   ifCondition.getDefiningOp()->getBlock() : 
                   cast<BlockArgument>(ifCondition).getOwner();
    
    // Use the later insertion point
    if (basePt->getBlock() == block) {
      // Compare iterators in the same block
      for (auto it = block->begin(); it != block->end(); ++it) {
        if (&*it == &*basePt) {
          // basePt comes first, use insertPt
          break;
        }
        if (&*it == &*insertPt) {
          // insertPt comes first, use basePt
          insertPt = basePt;
          break;
        }
      }
    }
  }
  
  // Set insertion point
  Block *insertBlock = ifCondition.getDefiningOp() ? 
                       ifCondition.getDefiningOp()->getBlock() : 
                       cast<BlockArgument>(ifCondition).getOwner();
  builder.setInsertionPoint(insertBlock, insertPt);
  
  // Create the condition expression
  Value condExpr = ifCondition;
  
  // Negate if needed
  if (negated) {
    auto trueVal = builder.create<arith::ConstantIntOp>(ifCondition.getLoc(), 1, 1);
    condExpr = builder.create<arith::XOrIOp>(ifCondition.getLoc(), 
                                              condExpr, trueVal);
  }
  
  // If no base condition, this is the root condition
  if (!baseCondition) {
    state.conditionCache[key] = condExpr;
    return condExpr;
  }
  
  // General case: base && cond
  auto result = builder.create<arith::AndIOp>(ifCondition.getLoc(), 
                                               baseCondition, condExpr);
  state.conditionCache[key] = result;
  return result;
}

void ReachabilityAnalysisPass::updateCallOp(CallOp callOp, Value condition, 
                                             OpBuilder &builder) {
  // Don't update if the call already has a condition
  if (callOp.getCondition()) {
    return;
  }
  
  // Create a new CallOp with the condition
  builder.setInsertionPoint(callOp);
  
  // Create new call with condition
  auto newCall = builder.create<CallOp>(
      callOp.getLoc(),
      callOp.getResultTypes(),
      callOp.getCalleeAttr(),
      condition,
      callOp.getArgs()
  );
  
  // Copy attributes except operandSegmentSizes (which will be set by the builder)
  SmallVector<NamedAttribute> newAttrs;
  for (auto attr : callOp->getAttrs()) {
    if (attr.getName() != "operandSegmentSizes") {
      newAttrs.push_back(attr);
    }
  }
  newCall->setAttrs(newAttrs);
  
  // Replace uses and erase old call
  callOp.replaceAllUsesWith(newCall);
  callOp.erase();
}

void ReachabilityAnalysisPass::updateAbortOp(txn::AbortOp abortOp, Value condition, 
                                              OpBuilder &builder) {
  // Don't update if the abort already has a condition
  if (abortOp.getCondition()) {
    return;
  }
  
  // Create a new AbortOp with the condition
  builder.setInsertionPoint(abortOp);
  
  // Create new abort with condition
  auto newAbort = builder.create<txn::AbortOp>(abortOp.getLoc(), condition);
  
  // Copy attributes
  newAbort->setAttrs(abortOp->getAttrs());
  
  // Since AbortOp has no results, we can directly erase the old one
  abortOp.erase();
}

} // namespace

std::unique_ptr<mlir::Pass> createReachabilityAnalysisPass() {
  return std::make_unique<ReachabilityAnalysisPass>();
}

} // namespace sharp
} // namespace mlir