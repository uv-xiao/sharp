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
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include <set>
#include <optional>
#include <vector>

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
  CF = 3,  // Conflict-Free (user-provided, cannot be overridden)
  UNK = 4  // Unknown (can be overridden by inference)
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
  
  /// Infer conflicts for a single primitive
  void inferPrimitiveConflicts(::sharp::txn::PrimitiveOp primitive);
  
  /// Process a schedule operation and update its conflict matrix
  void processSchedule(::sharp::txn::ScheduleOp schedule);
  
  /// Extract existing conflict matrix from schedule operation
  ConflictMatrix extractConflictMatrix(::sharp::txn::ScheduleOp schedule);
  
  /// Apply inference rules to complete the conflict matrix
  void applyInferenceRules(ConflictMatrix &cm, 
                          ArrayRef<std::string> actions,
                          Operation *parentOp);
  
  /// Update schedule operation with inferred conflict matrix
  void updateScheduleConflictMatrix(::sharp::txn::ScheduleOp schedule, 
                                   const ConflictMatrix &cm);
  
  /// Helper to check if two actions call the same method (deprecated - use method-based inference)
  bool callsSameMethod(const std::string &action1, const std::string &action2, 
                      ::sharp::txn::ModuleOp module);
  
  /// Get all method calls from an action
  std::vector<std::string> getMethodCalls(const std::string &actionName, 
                                         Operation *parentOp);
  
  /// Look up conflict relationship between two methods in primitive schedules
  std::optional<ConflictRelation> getMethodConflict(const std::string &method1, 
                                                    const std::string &method2,
                                                    Operation *parentOp);
  
  /// Get the inverse relation (SA <-> SB)
  ConflictRelation inverseRelation(ConflictRelation rel);
  
  /// Sort modules topologically based on dependencies
  SmallVector<::sharp::txn::ModuleOp> topologicalSortModules(
      ArrayRef<::sharp::txn::ModuleOp> modules);
};

void ConflictMatrixInferencePass::runOnOperation() {
  ModuleOp module = getOperation();
  
  // First process all primitives
  module.walk([&](::sharp::txn::PrimitiveOp primitive) {
    inferPrimitiveConflicts(primitive);
  });
  
  // Then collect all Txn modules and sort them topologically
  SmallVector<::sharp::txn::ModuleOp> txnModules;
  module.walk([&](::sharp::txn::ModuleOp txnModule) {
    txnModules.push_back(txnModule);
  });
  
  // Process modules in topological order (primitives already done, then modules that use them)
  auto sortedModules = topologicalSortModules(txnModules);
  
  for (auto txnModule : sortedModules) {
    inferModuleConflicts(txnModule);
  }
}

void ConflictMatrixInferencePass::inferPrimitiveConflicts(::sharp::txn::PrimitiveOp primitive) {
  LLVM_DEBUG(llvm::dbgs() << "Inferring conflicts for primitive: " 
                          << primitive.getName() << "\n");
  
  // Find all schedule operations in the primitive
  primitive.walk([&](::sharp::txn::ScheduleOp schedule) {
    processSchedule(schedule);
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
  auto parentOp = schedule->getParentOp();
  applyInferenceRules(cm, actionNames, parentOp);
  
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
    ConflictMatrix &cm, ArrayRef<std::string> actions, Operation *parentOp) {
  
  LLVM_DEBUG(llvm::dbgs() << "Applying inference rules for " << actions.size() << " actions\n");
  
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
  
  // Rule 3: Method-based inference - propagate conflicts from primitive methods to actions
  for (size_t i = 0; i < actions.size(); ++i) {
    for (size_t j = 0; j < actions.size(); ++j) {
      if (i == j) continue;
      
      auto key_ij = actions[i] + "," + actions[j];
      
      // Skip if already determined and cannot be overridden
      // We can only override UNK (4), CF (3) is user-provided and preserved
      auto existing = cm.find(key_ij);
      if (existing != cm.end() && existing->second != ConflictRelation::UNK) continue;
      
      LLVM_DEBUG(llvm::dbgs() << "Checking method conflicts for " << key_ij << "\n");
      
      // Get all method calls from both actions
      auto calls_i = getMethodCalls(actions[i], parentOp);
      auto calls_j = getMethodCalls(actions[j], parentOp);
      
      LLVM_DEBUG(llvm::dbgs() << "  Action " << actions[i] << " calls: ");
      for (const auto &call : calls_i) LLVM_DEBUG(llvm::dbgs() << call << " ");
      LLVM_DEBUG(llvm::dbgs() << "\n");
      
      LLVM_DEBUG(llvm::dbgs() << "  Action " << actions[j] << " calls: ");
      for (const auto &call : calls_j) LLVM_DEBUG(llvm::dbgs() << call << " ");
      LLVM_DEBUG(llvm::dbgs() << "\n");
      
      // Find the strongest conflict relationship between any pair of method calls
      std::optional<ConflictRelation> strongestConflict;
      
      for (const auto &call_i : calls_i) {
        for (const auto &call_j : calls_j) {
          LLVM_DEBUG(llvm::dbgs() << "  Checking method pair: " << call_i << " vs " << call_j << "\n");
          auto methodConflict = getMethodConflict(call_i, call_j, parentOp);
          if (methodConflict) {
            LLVM_DEBUG(llvm::dbgs() << "  Found method conflict: " << call_i << " vs " << call_j << " = " << static_cast<int>(*methodConflict) << "\n");
            // Update strongest conflict (C > SA/SB > CF)
            if (!strongestConflict || 
                (*methodConflict == ConflictRelation::C) ||
                (*strongestConflict == ConflictRelation::CF && *methodConflict != ConflictRelation::CF)) {
              strongestConflict = *methodConflict;
            }
          } else {
            LLVM_DEBUG(llvm::dbgs() << "  No conflict found for: " << call_i << " vs " << call_j << "\n");
          }
        }
      }
      
      // Apply the strongest conflict found
      if (strongestConflict) {
        cm[key_ij] = *strongestConflict;
      }
    }
  }
  
  // Rule 4: Propagate conflicts through method calls
  // TODO: Implement method call analysis and propagation
  
  // Rule 5: Default to unknown if not determined
  for (size_t i = 0; i < actions.size(); ++i) {
    for (size_t j = 0; j < actions.size(); ++j) {
      if (i != j) {
        std::string key = actions[i] + "," + actions[j];
        if (cm.find(key) == cm.end()) {
          cm[key] = ConflictRelation::UNK;
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
  // Get the action operations
  auto getActionOp = [&](const std::string &actionName) -> Operation* {
    Operation *actionOp = nullptr;
    module.walk([&](Operation *op) {
      if (auto ruleOp = dyn_cast<::sharp::txn::RuleOp>(op)) {
        if (ruleOp.getName() == actionName) {
          actionOp = op;
          return WalkResult::interrupt();
        }
      } else if (auto actionMethodOp = dyn_cast<::sharp::txn::ActionMethodOp>(op)) {
        if (actionMethodOp.getName() == actionName) {
          actionOp = op;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    return actionOp;
  };
  
  Operation *op1 = getActionOp(action1);
  Operation *op2 = getActionOp(action2);
  
  if (!op1 || !op2) return false;
  
  // Collect all method calls from both actions
  auto getMethodCalls = [](Operation *op) -> std::set<std::string> {
    std::set<std::string> calls;
    op->walk([&](::sharp::txn::CallOp callOp) {
      // Convert symbol reference to string key
      auto callee = callOp.getCallee();
      std::string callKey;
      if (callee.getNestedReferences().empty()) {
        // Simple method call like @method
        callKey = callee.getRootReference().getValue().str();
      } else {
        // Instance method call like @instance::@method
        callKey = callee.getRootReference().getValue().str() + "::" + 
                  callee.getLeafReference().getValue().str();
      }
      calls.insert(callKey);
    });
    return calls;
  };
  
  std::set<std::string> calls1 = getMethodCalls(op1);
  std::set<std::string> calls2 = getMethodCalls(op2);
  
  // Check if there's any overlap in method calls
  for (const auto &call1 : calls1) {
    if (calls2.count(call1)) {
      LLVM_DEBUG(llvm::dbgs() << "Actions " << action1 << " and " << action2 
                              << " both call " << call1 << "\n");
      return true;
    }
  }
  
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

SmallVector<::sharp::txn::ModuleOp> ConflictMatrixInferencePass::topologicalSortModules(
    ArrayRef<::sharp::txn::ModuleOp> modules) {
  // Build dependency graph
  llvm::DenseMap<::sharp::txn::ModuleOp, SmallVector<::sharp::txn::ModuleOp>> dependencies;
  llvm::DenseMap<::sharp::txn::ModuleOp, int> inDegree;
  
  // Initialize structures
  for (auto module : modules) {
    dependencies[module] = {};
    inDegree[module] = 0;
  }
  
  // Build dependency edges: if module A instantiates module B, then A depends on B
  for (auto module : modules) {
    module.walk([&](::sharp::txn::InstanceOp instanceOp) {
      // Find the module being instantiated
      auto moduleName = instanceOp.getModuleName();
      for (auto otherModule : modules) {
        if (otherModule.getName() == moduleName) {
          dependencies[module].push_back(otherModule);
          inDegree[otherModule]++;
          break;
        }
      }
    });
  }
  
  // Topological sort using Kahn's algorithm
  SmallVector<::sharp::txn::ModuleOp> result;
  SmallVector<::sharp::txn::ModuleOp> queue;
  
  // Start with modules that have no dependencies
  for (auto module : modules) {
    if (inDegree[module] == 0) {
      queue.push_back(module);
    }
  }
  
  while (!queue.empty()) {
    auto current = queue.pop_back_val();
    result.push_back(current);
    
    // Process all modules that depend on current
    for (auto dependent : dependencies[current]) {
      inDegree[dependent]--;
      if (inDegree[dependent] == 0) {
        queue.push_back(dependent);
      }
    }
  }
  
  // If we couldn't sort all modules, there's a cycle - just return original order
  if (result.size() != modules.size()) {
    LLVM_DEBUG(llvm::dbgs() << "Cycle detected in module dependencies, using original order\n");
    return SmallVector<::sharp::txn::ModuleOp>(modules.begin(), modules.end());
  }
  
  return result;
}

std::vector<std::string> ConflictMatrixInferencePass::getMethodCalls(
    const std::string &actionName, Operation *parentOp) {
  std::vector<std::string> calls;
  
  // Find the action operation
  Operation *actionOp = nullptr;
  parentOp->walk([&](Operation *op) {
    if (auto ruleOp = dyn_cast<::sharp::txn::RuleOp>(op)) {
      if (ruleOp.getName() == actionName) {
        actionOp = op;
        return WalkResult::interrupt();
      }
    } else if (auto actionMethodOp = dyn_cast<::sharp::txn::ActionMethodOp>(op)) {
      if (actionMethodOp.getName() == actionName) {
        actionOp = op;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  
  if (!actionOp) return calls;
  
  // Collect all method calls
  actionOp->walk([&](::sharp::txn::CallOp callOp) {
    auto callee = callOp.getCallee();
    std::string callKey;
    if (callee.getNestedReferences().empty()) {
      // Simple method call like @method
      callKey = callee.getRootReference().getValue().str();
    } else {
      // Instance method call like @instance::@method
      callKey = callee.getRootReference().getValue().str() + "::" + 
                callee.getLeafReference().getValue().str();
    }
    calls.push_back(callKey);
  });
  
  return calls;
}

std::optional<ConflictRelation> ConflictMatrixInferencePass::getMethodConflict(
    const std::string &method1, const std::string &method2, Operation *parentOp) {
  
  // Parse method call format: "instance::method" or "method"
  auto parseMethodCall = [](const std::string &call) -> std::pair<std::string, std::string> {
    size_t colonPos = call.find("::");
    if (colonPos != std::string::npos) {
      return {call.substr(0, colonPos), call.substr(colonPos + 2)};
    }
    return {"", call}; // Direct method call
  };
  
  auto [instance1, methodName1] = parseMethodCall(method1);
  auto [instance2, methodName2] = parseMethodCall(method2);
  
  // Only compare methods from the same instance
  if (instance1 != instance2) {
    return std::nullopt; // Different instances, no conflict
  }
  
  if (instance1.empty()) {
    return std::nullopt; // Direct method calls not handled here
  }
  
  // Find the instance and its type
  ::sharp::txn::InstanceOp instanceOp = nullptr;
  parentOp->walk([&](::sharp::txn::InstanceOp instOp) {
    if (instOp.getName() == instance1) {
      instanceOp = instOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  if (!instanceOp) {
    LLVM_DEBUG(llvm::dbgs() << "    Instance not found: " << instance1 << "\n");
    return std::nullopt;
  }
  
  LLVM_DEBUG(llvm::dbgs() << "    Found instance: " << instance1 << "\n");
  
  // Find the primitive/module definition
  // Reconstruct the full primitive name for parametric primitives
  auto moduleNameAttr = instanceOp.getModuleNameAttr();
  auto baseName = moduleNameAttr.getValue().str();
  auto typeArgs = instanceOp.getTypeArguments();
  
  std::string moduleName = baseName;
  if (typeArgs && !typeArgs->empty()) {
    // Reconstruct the full primitive name like "Register<i1>"
    moduleName = baseName + "<";
    for (size_t i = 0; i < typeArgs->size(); ++i) {
      if (i > 0) moduleName += ",";
      if (auto typeAttr = dyn_cast<TypeAttr>((*typeArgs)[i])) {
        auto type = typeAttr.getValue();
        if (auto intType = dyn_cast<IntegerType>(type)) {
          moduleName += "i" + std::to_string(intType.getWidth());
        } else {
          moduleName += "unknown";
        }
      }
    }
    moduleName += ">";
  }
  
  LLVM_DEBUG(llvm::dbgs() << "    Looking for module: " << moduleName << "\n");
  ::sharp::txn::PrimitiveOp primitiveOp = nullptr;
  ::sharp::txn::ModuleOp subModuleOp = nullptr;
  
  // Look in current parent (module or primitive) first
  parentOp->walk([&](Operation *op) {
    if (auto primOp = dyn_cast<::sharp::txn::PrimitiveOp>(op)) {
      if (primOp.getName() == moduleName) {
        primitiveOp = primOp;
        return WalkResult::interrupt();
      }
    } else if (auto modOp = dyn_cast<::sharp::txn::ModuleOp>(op)) {
      if (modOp.getName() == moduleName) {
        subModuleOp = modOp;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  
  // Look in global module scope if not found
  if (!primitiveOp && !subModuleOp) {
    auto topModule = parentOp->getParentOfType<ModuleOp>();
    if (topModule) {
      topModule.walk([&](Operation *op) {
        if (auto primOp = dyn_cast<::sharp::txn::PrimitiveOp>(op)) {
          if (primOp.getName() == moduleName) {
            primitiveOp = primOp;
            return WalkResult::interrupt();
          }
        } else if (auto modOp = dyn_cast<::sharp::txn::ModuleOp>(op)) {
          if (modOp.getName() == moduleName) {
            subModuleOp = modOp;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
    }
  }
  
  Operation *targetOp = primitiveOp ? primitiveOp.getOperation() : 
                       (subModuleOp ? subModuleOp.getOperation() : nullptr);
  if (!targetOp) {
    LLVM_DEBUG(llvm::dbgs() << "    Target primitive/module not found: " << moduleName << "\n");
    return std::nullopt;
  }
  
  LLVM_DEBUG(llvm::dbgs() << "    Found target: " << moduleName << "\n");
  
  // Find the schedule and look up conflict matrix
  ::sharp::txn::ScheduleOp scheduleOp = nullptr;
  targetOp->walk([&](::sharp::txn::ScheduleOp schedOp) {
    scheduleOp = schedOp;
    return WalkResult::interrupt();
  });
  
  if (!scheduleOp || !scheduleOp.getConflictMatrix()) {
    return std::nullopt;
  }
  
  // Look up the conflict relationship
  auto cmAttr = *scheduleOp.getConflictMatrix();
  auto cmDict = cast<DictionaryAttr>(cmAttr);
  
  std::string key = methodName1 + "," + methodName2;
  LLVM_DEBUG(llvm::dbgs() << "    Looking up conflict: " << key << "\n");
  LLVM_DEBUG(llvm::dbgs() << "    Available keys in primitive conflict matrix: ");
  for (auto entry : cmDict) {
    LLVM_DEBUG(llvm::dbgs() << entry.getName().str() << " ");
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");
  
  for (auto entry : cmDict) {
    if (entry.getName().str() == key) {
      auto value = cast<IntegerAttr>(entry.getValue()).getInt();
      LLVM_DEBUG(llvm::dbgs() << "    Found conflict: " << key << " = " << value << "\n");
      return static_cast<ConflictRelation>(value);
    }
  }
  
  return std::nullopt;
}

} // namespace

std::unique_ptr<mlir::Pass> createConflictMatrixInferencePass() {
  return std::make_unique<ConflictMatrixInferencePass>();
}

} // namespace sharp
} // namespace mlir