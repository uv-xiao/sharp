//===- ConflictMatrixInference.cpp - Conflict Matrix Inference Pass ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the conflict matrix inference pass for Sharp Txn dialect.
//
//===----------------------------------------------------------------------===//

#include "sharp/Analysis/Passes.h"
#include "sharp/Dialect/Txn/TxnDialect.h"
#include "sharp/Dialect/Txn/TxnOps.h"
#include "sharp/Dialect/Txn/TxnAttrs.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sharp-conflict-matrix-inference"

namespace mlir {
namespace sharp {

#define GEN_PASS_DEF_CONFLICTMATRIXINFERENCE
#include "sharp/Analysis/Passes.h.inc"

namespace {

/// Represents a conflict relationship between two actions
enum class ConflictRelation {
  SB = 0,  // Sequence Before
  SA = 1,  // Sequence After
  C = 2,   // Conflict
  CF = 3   // Conflict-Free
};

/// Conflict matrix type - using string keys "action1,action2"
using ConflictMatrix = llvm::StringMap<ConflictRelation>;

class ConflictMatrixInferencePass
    : public impl::ConflictMatrixInferenceBase<ConflictMatrixInferencePass> {
public:
  void runOnOperation() override;

private:
  /// Infer conflicts for a single module
  void inferModuleConflicts(::sharp::txn::ModuleOp module);
  
  /// Process a schedule operation and update its conflict matrix
  void processSchedule(::sharp::txn::ScheduleOp schedule);
  
  /// Extract existing conflict matrix from schedule operation
  ConflictMatrix extractConflictMatrix(::sharp::txn::ScheduleOp schedule);
  
  /// Apply inference rules to complete the conflict matrix
  void applyInferenceRules(ConflictMatrix &cm, 
                          ArrayRef<std::string> actions,
                          ::sharp::txn::ModuleOp module);
  
  /// Update schedule operation with inferred conflict matrix
  void updateScheduleConflictMatrix(::sharp::txn::ScheduleOp schedule, 
                                   const ConflictMatrix &cm);
  
  /// Helper to check if two actions call the same method
  bool callsSameMethod(const std::string &action1, const std::string &action2, 
                      ::sharp::txn::ModuleOp module);
  
  /// Get the inverse relation (SA <-> SB)
  ConflictRelation inverseRelation(ConflictRelation rel);
};

void ConflictMatrixInferencePass::runOnOperation() {
  ModuleOp module = getOperation();
  
  // Process each Sharp Txn module
  module.walk([&](::sharp::txn::ModuleOp txnModule) {
    inferModuleConflicts(txnModule);
  });
}

void ConflictMatrixInferencePass::inferModuleConflicts(::sharp::txn::ModuleOp module) {
  LLVM_DEBUG(llvm::dbgs() << "Inferring conflicts for module: " 
                          << module.getName() << "\n");
  
  // Find all schedule operations in the module
  module.walk([&](::sharp::txn::ScheduleOp schedule) {
    processSchedule(schedule);
  });
}

void ConflictMatrixInferencePass::processSchedule(::sharp::txn::ScheduleOp schedule) {
  // Extract existing conflict matrix
  ConflictMatrix cm = extractConflictMatrix(schedule);
  
  // Get list of actions from schedule
  auto actions = schedule.getActions();
  SmallVector<std::string> actionNames;
  for (auto action : actions) {
    actionNames.push_back(cast<FlatSymbolRefAttr>(action).getValue().str());
  }
  
  // Apply inference rules
  auto module = schedule->getParentOfType<::sharp::txn::ModuleOp>();
  applyInferenceRules(cm, actionNames, module);
  
  // Update schedule with inferred conflict matrix
  updateScheduleConflictMatrix(schedule, cm);
}

ConflictMatrix ConflictMatrixInferencePass::extractConflictMatrix(
    ::sharp::txn::ScheduleOp schedule) {
  ConflictMatrix cm;
  
  if (auto cmAttr = schedule.getConflictMatrix()) {
    auto cmDict = cast<DictionaryAttr>(*cmAttr);
    for (auto entry : cmDict) {
      auto key = entry.getName().str();
      auto value = cast<IntegerAttr>(entry.getValue()).getInt();
      
      // Parse key format: "action1,action2"
      size_t commaPos = key.find(',');
      if (commaPos != std::string::npos) {
        cm[key] = static_cast<ConflictRelation>(value);
      }
    }
  }
  
  return cm;
}

void ConflictMatrixInferencePass::applyInferenceRules(
    ConflictMatrix &cm, ArrayRef<std::string> actions, ::sharp::txn::ModuleOp module) {
  
  // Rule 1: Any action conflicts with itself
  for (auto action : actions) {
    cm[action + "," + action] = ConflictRelation::C;
  }
  
  // Rule 2: If action A is SA to B and B is SB to A, they conflict
  // This means if we have A->B = SA (1) and B->A = SB (0), both should be C (2)
  for (size_t i = 0; i < actions.size(); ++i) {
    for (size_t j = 0; j < actions.size(); ++j) {
      if (i == j) continue;
      
      auto key_ij = actions[i] + "," + actions[j];
      auto key_ji = actions[j] + "," + actions[i];
      
      auto it_ij = cm.find(key_ij);
      auto it_ji = cm.find(key_ji);
      
      if (it_ij != cm.end() && it_ji != cm.end()) {
        // Check if i SA j (1) and j SB i (0), which means cyclic dependency
        if (it_ij->second == ConflictRelation::SA && 
            it_ji->second == ConflictRelation::SB) {
          cm[key_ij] = ConflictRelation::C;
          cm[key_ji] = ConflictRelation::C;
        }
      }
    }
  }
  
  // Rule 3: If two actions call the same action method, they conflict
  for (size_t i = 0; i < actions.size(); ++i) {
    for (size_t j = i + 1; j < actions.size(); ++j) {
      if (callsSameMethod(actions[i], actions[j], module)) {
        cm[actions[i] + "," + actions[j]] = ConflictRelation::C;
        cm[actions[j] + "," + actions[i]] = ConflictRelation::C;
      }
    }
  }
  
  // Rule 4: Propagate conflicts through method calls
  // TODO: Implement method call analysis and propagation
  
  // Rule 5: Default to conflict-free if not determined
  for (size_t i = 0; i < actions.size(); ++i) {
    for (size_t j = 0; j < actions.size(); ++j) {
      if (i != j) {
        std::string key = actions[i] + "," + actions[j];
        if (cm.find(key) == cm.end()) {
          cm[key] = ConflictRelation::CF;
        }
      }
    }
  }
}

void ConflictMatrixInferencePass::updateScheduleConflictMatrix(
    ::sharp::txn::ScheduleOp schedule, const ConflictMatrix &cm) {
  
  // Build new conflict matrix dictionary
  SmallVector<NamedAttribute> cmAttrs;
  
  for (const auto &entry : cm) {
    auto value = IntegerAttr::get(
        IntegerType::get(schedule.getContext(), 32),
        static_cast<int32_t>(entry.second));
    cmAttrs.push_back(NamedAttribute(
        StringAttr::get(schedule.getContext(), entry.first()), value));
  }
  
  auto newCmAttr = DictionaryAttr::get(schedule.getContext(), cmAttrs);
  schedule.setConflictMatrixAttr(newCmAttr);
}

bool ConflictMatrixInferencePass::callsSameMethod(
    const std::string &action1, const std::string &action2, ::sharp::txn::ModuleOp module) {
  // TODO: Implement analysis to check if two actions call the same method
  // This requires analyzing the body of rules/methods to find method calls
  return false;
}

ConflictRelation ConflictMatrixInferencePass::inverseRelation(
    ConflictRelation rel) {
  switch (rel) {
    case ConflictRelation::SA: return ConflictRelation::SB;
    case ConflictRelation::SB: return ConflictRelation::SA;
    default: return rel;
  }
}

} // namespace

std::unique_ptr<mlir::Pass> createConflictMatrixInferencePass() {
  return std::make_unique<ConflictMatrixInferencePass>();
}

} // namespace sharp
} // namespace mlir