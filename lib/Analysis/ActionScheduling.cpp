//===- ActionScheduling.cpp - Action Scheduling Pass Implementation ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sharp/Analysis/ActionScheduling.h"
#include "sharp/Analysis/Passes.h"
#include "sharp/Dialect/Txn/TxnOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <queue>

#define DEBUG_TYPE "sharp-action-scheduling"

using namespace mlir;
using namespace mlir::sharp;
using namespace mlir::sharp::txn;

namespace mlir {
namespace sharp {

#define GEN_PASS_DEF_ACTIONSCHEDULING
#include "sharp/Analysis/Passes.h.inc"

//===----------------------------------------------------------------------===//
// ConflictMatrix Implementation
//===----------------------------------------------------------------------===//

ConflictMatrix::ActionPair ConflictMatrix::makeKey(StringRef a1, StringRef a2) const {
  // Ensure consistent ordering for symmetric access
  if (a1 < a2)
    return {a1.str(), a2.str()};
  return {a2.str(), a1.str()};
}

ConflictRelation ConflictMatrix::get(StringRef a1, StringRef a2) const {
  auto key = makeKey(a1, a2);
  auto it = matrix.find(key);
  if (it != matrix.end()) {
    // Handle symmetry for SA/SB
    if (a1 > a2) {
      if (it->second == ConflictRelation::SB)
        return ConflictRelation::SA;
      if (it->second == ConflictRelation::SA)
        return ConflictRelation::SB;
    }
    return it->second;
  }
  return ConflictRelation::CF; // Default to conflict-free
}

void ConflictMatrix::set(StringRef a1, StringRef a2, ConflictRelation rel) {
  auto key = makeKey(a1, a2);
  // Store with canonical ordering
  if (a1 > a2) {
    // Flip SB/SA for reversed order
    if (rel == ConflictRelation::SB)
      rel = ConflictRelation::SA;
    else if (rel == ConflictRelation::SA)
      rel = ConflictRelation::SB;
  }
  matrix[key] = rel;
}

bool ConflictMatrix::hasRelation(StringRef a1, StringRef a2) const {
  auto key = makeKey(a1, a2);
  return matrix.find(key) != matrix.end();
}

SmallVector<std::tuple<StringRef, StringRef, ConflictRelation>> 
ConflictMatrix::getAllPairs() const {
  SmallVector<std::tuple<StringRef, StringRef, ConflictRelation>> result;
  for (const auto &[pair, rel] : matrix) {
    result.push_back({pair.first, pair.second, rel});
  }
  return result;
}

//===----------------------------------------------------------------------===//
// ActionSchedulingPass Implementation
//===----------------------------------------------------------------------===//

namespace {
class ActionSchedulingPass : public impl::ActionSchedulingBase<ActionSchedulingPass> {
public:
  void runOnOperation() override;
  
private:
  // Process a single txn module
  void processModule(::sharp::txn::ModuleOp module);
  
  // Check if module has a complete schedule
  bool hasCompleteSchedule(::sharp::txn::ModuleOp module);
  
  // Build dependency graph from module and conflict matrix
  void buildSchedulingGraph(::sharp::txn::ModuleOp module, const ConflictMatrix &cm,
                           SchedulingGraph &graph);
  
  // Compute optimal schedule for small modules
  SmallVector<StringRef> computeOptimalSchedule(const SchedulingGraph &graph,
                                                const ConflictMatrix &cm);
  
  // Compute heuristic schedule for large modules
  SmallVector<StringRef> computeHeuristicSchedule(const SchedulingGraph &graph,
                                                  const ConflictMatrix &cm);
  
  // Evaluate the cost of a given schedule
  SchedulingCost evaluateSchedule(ArrayRef<StringRef> schedule,
                                 const ConflictMatrix &cm);
  
  // Check if dependency graph has cycles
  bool hasCycle(const DenseMap<StringRef, SmallVector<StringRef>> &adj);
  
  // Generate all topological orderings (for small graphs)
  void generateTopologicalOrderings(const SchedulingGraph &graph,
                                   SmallVector<SmallVector<StringRef>> &result);
  
  // Select best action from ready set (for heuristic)
  StringRef selectBestAction(ArrayRef<StringRef> ready,
                            const DenseSet<StringRef> &scheduled,
                            ArrayRef<StringRef> currentSchedule,
                            const ConflictMatrix &cm,
                            const DenseMap<StringRef, int> &partialOrder);
  
  // Replace or create schedule operation
  void replaceScheduleOp(::sharp::txn::ModuleOp module, ArrayRef<StringRef> schedule);
};
} // namespace

void ActionSchedulingPass::runOnOperation() {
  auto module = getOperation();
  
  // Process each txn module
  module.walk([&](::sharp::txn::ModuleOp txnModule) {
    LLVM_DEBUG(llvm::dbgs() << "Running action scheduling on module: "
                            << txnModule.getSymName() << "\n");
    
    // Skip if schedule is already complete
    if (hasCompleteSchedule(txnModule)) {
      LLVM_DEBUG(llvm::dbgs() << "Module already has complete schedule\n");
      return;
    }
    
    processModule(txnModule);
  });
}

void ActionSchedulingPass::processModule(::sharp::txn::ModuleOp module) {
  
  // Build conflict matrix from schedule operation
  ConflictMatrix cm;
  
  // Extract conflict matrix from existing schedule
  for (auto scheduleOp : module.getOps<::sharp::txn::ScheduleOp>()) {
    if (auto cmAttrOpt = scheduleOp.getConflictMatrix()) {
      auto cmAttr = cmAttrOpt.value();
      for (auto namedAttr : cmAttr) {
        auto keyStr = cast<StringAttr>(namedAttr.getName()).getValue();
        auto intValue = cast<IntegerAttr>(namedAttr.getValue()).getInt();
        
        // Parse key format "action1,action2"
        size_t commaPos = keyStr.find(',');
        if (commaPos != StringRef::npos) {
          StringRef a1 = keyStr.substr(0, commaPos);
          StringRef a2 = keyStr.substr(commaPos + 1);
          cm.set(a1, a2, static_cast<ConflictRelation>(intValue));
        }
      }
    }
  }
  
  // Build scheduling graph
  SchedulingGraph graph;
  buildSchedulingGraph(module, cm, graph);
  
  if (graph.actions.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No actions to schedule\n");
    return;
  }
  
  // Compute schedule
  SmallVector<StringRef> schedule;
  if (graph.actions.size() <= 10) {
    // Use exact algorithm for small modules
    LLVM_DEBUG(llvm::dbgs() << "Using exact algorithm for " 
                            << graph.actions.size() << " actions\n");
    schedule = computeOptimalSchedule(graph, cm);
  } else {
    // Use heuristic for larger modules
    LLVM_DEBUG(llvm::dbgs() << "Using heuristic algorithm for " 
                            << graph.actions.size() << " actions\n");
    schedule = computeHeuristicSchedule(graph, cm);
  }
  
  if (schedule.empty()) {
    module.emitError("Failed to compute valid schedule - possible cyclic dependencies");
    signalPassFailure();
    return;
  }
  
  // Replace or create schedule operation
  replaceScheduleOp(module, schedule);
  
  LLVM_DEBUG({
    llvm::dbgs() << "Final schedule: [";
    for (size_t i = 0; i < schedule.size(); ++i) {
      if (i > 0) llvm::dbgs() << ", ";
      llvm::dbgs() << schedule[i];
    }
    llvm::dbgs() << "]\n";
    
    auto cost = evaluateSchedule(schedule, cm);
    llvm::dbgs() << "Schedule cost - SB violations: " << cost.sbViolations
                 << ", SA violations: " << cost.saViolations
                 << ", conflicts: " << cost.conflicts << "\n";
  });
}

bool ActionSchedulingPass::hasCompleteSchedule(::sharp::txn::ModuleOp module) {
  // Get all actions
  DenseSet<StringRef> allActions;
  for (auto rule : module.getOps<::sharp::txn::RuleOp>())
    allActions.insert(rule.getSymName());
  for (auto method : module.getOps<::sharp::txn::ActionMethodOp>())
    allActions.insert(method.getSymName());
  
  if (allActions.empty())
    return true; // No actions to schedule
  
  // Check if existing schedule contains all actions
  for (auto scheduleOp : module.getOps<::sharp::txn::ScheduleOp>()) {
    auto scheduled = scheduleOp.getActions();
    if (scheduled.size() == allActions.size()) {
      // Verify all actions are present
      bool complete = true;
      for (auto action : scheduled) {
        if (!allActions.count(cast<FlatSymbolRefAttr>(action).getValue())) {
          complete = false;
          break;
        }
      }
      if (complete)
        return true;
    }
  }
  
  return false;
}

void ActionSchedulingPass::buildSchedulingGraph(::sharp::txn::ModuleOp module,
                                               const ConflictMatrix &cm,
                                               SchedulingGraph &graph) {
  // 1. Add all actions as nodes
  for (auto rule : module.getOps<::sharp::txn::RuleOp>())
    graph.actions.insert(rule.getSymName());
  for (auto method : module.getOps<::sharp::txn::ActionMethodOp>())
    graph.actions.insert(method.getSymName());
  
  // 2. Extract partial schedule constraints
  for (auto scheduleOp : module.getOps<::sharp::txn::ScheduleOp>()) {
    auto partialActions = scheduleOp.getActions();
    for (size_t i = 0; i < partialActions.size(); ++i) {
      auto action = cast<FlatSymbolRefAttr>(partialActions[i]).getValue();
      graph.partialOrder[action] = i;
      
      // Add edges to maintain partial order
      for (size_t j = i + 1; j < partialActions.size(); ++j) {
        auto laterAction = cast<FlatSymbolRefAttr>(partialActions[j]).getValue();
        graph.mustPrecede[action].push_back(laterAction);
      }
    }
  }
  
  // 3. Add edges from conflict matrix (SA relationships)
  for (auto a1 : graph.actions) {
    for (auto a2 : graph.actions) {
      if (a1 != a2 && cm.hasRelation(a1, a2)) {
        auto rel = cm.get(a1, a2);
        if (rel == ConflictRelation::SA) {
          // a1 Sequential After a2 means a2 must precede a1
          graph.mustPrecede[a2].push_back(a1);
        }
      }
    }
  }
}

bool ActionSchedulingPass::hasCycle(const DenseMap<StringRef, SmallVector<StringRef>> &adj) {
  DenseMap<StringRef, int> state; // 0: unvisited, 1: visiting, 2: visited
  DenseSet<StringRef> allNodes;
  
  // Collect all nodes (both sources and targets)
  for (const auto &[from, toList] : adj) {
    allNodes.insert(from);
    for (auto to : toList) {
      allNodes.insert(to);
    }
  }
  
  std::function<bool(StringRef)> dfs = [&](StringRef node) {
    state[node] = 1; // Mark as visiting
    
    auto it = adj.find(node);
    if (it != adj.end()) {
      for (auto neighbor : it->second) {
        if (state[neighbor] == 1) // Back edge found
          return true;
        if (state[neighbor] == 0 && dfs(neighbor))
          return true;
      }
    }
    
    state[node] = 2; // Mark as visited
    return false;
  };
  
  // Check all nodes
  for (auto node : allNodes) {
    if (state[node] == 0 && dfs(node))
      return true;
  }
  
  return false;
}

void ActionSchedulingPass::generateTopologicalOrderings(
    const SchedulingGraph &graph,
    SmallVector<SmallVector<StringRef>> &result) {
  
  // For small graphs, generate all valid topological orderings
  SmallVector<StringRef> current;
  DenseSet<StringRef> visited;
  DenseMap<StringRef, int> inDegree;
  
  
  // Compute in-degrees
  for (auto action : graph.actions)
    inDegree[action] = 0;
  
  for (const auto &[from, toList] : graph.mustPrecede) {
    for (auto to : toList)
      inDegree[to]++;
  }
  
  
  std::function<void()> generateAll = [&]() {
    if (current.size() == graph.actions.size()) {
      result.push_back(current);
      return;
    }
    
    // Find all vertices with in-degree 0
    for (auto action : graph.actions) {
      if (!visited.count(action) && inDegree[action] == 0) {
        // Choose this vertex
        visited.insert(action);
        current.push_back(action);
        
        // Reduce in-degree of neighbors
        auto it = graph.mustPrecede.find(action);
        if (it != graph.mustPrecede.end()) {
          for (auto neighbor : it->second)
            inDegree[neighbor]--;
        }
        
        // Recurse
        generateAll();
        
        // Backtrack
        if (it != graph.mustPrecede.end()) {
          for (auto neighbor : it->second)
            inDegree[neighbor]++;
        }
        
        current.pop_back();
        visited.erase(action);
      }
    }
  };
  
  generateAll();
}

SmallVector<StringRef> ActionSchedulingPass::computeOptimalSchedule(
    const SchedulingGraph &graph, const ConflictMatrix &cm) {
  
  // Check for cycles first
  if (hasCycle(graph.mustPrecede)) {
    LLVM_DEBUG(llvm::dbgs() << "Cycle detected in dependency graph\n");
    return {};
  }
  
  LLVM_DEBUG({
    llvm::dbgs() << "Building optimal schedule for " << graph.actions.size() << " actions\n";
    llvm::dbgs() << "Dependency edges:\n";
    for (const auto &[from, toList] : graph.mustPrecede) {
      for (auto to : toList) {
        llvm::dbgs() << "  " << from << " -> " << to << "\n";
      }
    }
  });
  
  // Generate all valid topological orderings
  SmallVector<SmallVector<StringRef>> validSchedules;
  generateTopologicalOrderings(graph, validSchedules);
  
  if (validSchedules.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No valid topological orderings found\n");
    return {};
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Generated " << validSchedules.size() << " valid orderings\n");
  
  // Evaluate each schedule and pick the best
  SmallVector<StringRef> bestSchedule = validSchedules[0]; // Initialize with first schedule
  SchedulingCost bestCost = evaluateSchedule(bestSchedule, cm);
  
  for (size_t i = 1; i < validSchedules.size(); ++i) {
    const auto &schedule = validSchedules[i];
    SchedulingCost cost = evaluateSchedule(schedule, cm);
    
    // Prefer schedules with fewer SB+SA violations, then fewer conflicts
    if (cost.sbViolations + cost.saViolations < bestCost.sbViolations + bestCost.saViolations ||
        (cost.sbViolations + cost.saViolations == bestCost.sbViolations + bestCost.saViolations && 
         cost.conflicts < bestCost.conflicts)) {
      bestCost = cost;
      bestSchedule = schedule;
    }
  }
  
  return bestSchedule;
}

SmallVector<StringRef> ActionSchedulingPass::computeHeuristicSchedule(
    const SchedulingGraph &graph, const ConflictMatrix &cm) {
  
  SmallVector<StringRef> schedule;
  DenseSet<StringRef> scheduled;
  DenseMap<StringRef, int> inDegree;
  
  // Compute initial in-degrees
  for (auto action : graph.actions)
    inDegree[action] = 0;
  
  for (const auto &[from, toList] : graph.mustPrecede) {
    for (auto to : toList)
      inDegree[to]++;
  }
  
  // Kahn's algorithm with conflict-aware selection
  while (scheduled.size() < graph.actions.size()) {
    // Find all actions with in-degree 0
    SmallVector<StringRef> ready;
    for (auto action : graph.actions) {
      if (!scheduled.count(action) && inDegree[action] == 0) {
        ready.push_back(action);
      }
    }
    
    if (ready.empty()) {
      // Cycle detected
      LLVM_DEBUG(llvm::dbgs() << "Cycle detected during heuristic scheduling\n");
      return {};
    }
    
    // Select best action from ready set
    StringRef best = selectBestAction(ready, scheduled, schedule, cm, graph.partialOrder);
    schedule.push_back(best);
    scheduled.insert(best);
    
    // Update in-degrees
    auto it = graph.mustPrecede.find(best);
    if (it != graph.mustPrecede.end()) {
      for (auto successor : it->second)
        inDegree[successor]--;
    }
  }
  
  return schedule;
}

StringRef ActionSchedulingPass::selectBestAction(
    ArrayRef<StringRef> ready,
    const DenseSet<StringRef> &scheduled,
    ArrayRef<StringRef> currentSchedule,
    const ConflictMatrix &cm,
    const DenseMap<StringRef, int> &partialOrder) {
  
  StringRef bestAction = ready[0];
  int bestScore = INT_MAX;
  
  for (auto candidate : ready) {
    int score = 0;
    
    // Penalize placing this action if it has SB relationships
    // with already scheduled actions
    for (auto prev : currentSchedule) {
      auto rel = cm.get(candidate, prev);
      if (rel == ConflictRelation::SB) {
        score += 100;  // Heavy penalty for SB violation
      } else if (rel == ConflictRelation::SA) {
        // This should not happen if dependencies are correct
        score += 1000; // Very heavy penalty
      } else if (rel == ConflictRelation::C) {
        score += 1;    // Light penalty for conflicts
      }
    }
    
    // Prefer actions from partial schedule in their original order
    auto it = partialOrder.find(candidate);
    if (it != partialOrder.end()) {
      // Give bonus based on original position
      score -= 50 * (partialOrder.size() - it->second);
    }
    
    if (score < bestScore) {
      bestScore = score;
      bestAction = candidate;
    }
  }
  
  return bestAction;
}

SchedulingCost ActionSchedulingPass::evaluateSchedule(
    ArrayRef<StringRef> schedule, const ConflictMatrix &cm) {
  
  SchedulingCost cost;
  DenseMap<StringRef, size_t> position;
  
  // Build position map
  for (size_t i = 0; i < schedule.size(); ++i)
    position[schedule[i]] = i;
  
  // Check all pairs
  for (size_t i = 0; i < schedule.size(); ++i) {
    for (size_t j = i + 1; j < schedule.size(); ++j) {
      auto rel = cm.get(schedule[i], schedule[j]);
      
      if (rel == ConflictRelation::SA) {
        // schedule[i] should be after schedule[j], but it's before
        cost.saViolations++;
      } else if (rel == ConflictRelation::C) {
        // Conflict exists
        cost.conflicts++;
      }
    }
  }
  
  // Check for SB violations
  for (auto [a1, a2, rel] : cm.getAllPairs()) {
    if (rel == ConflictRelation::SB) {
      auto it1 = position.find(a1);
      auto it2 = position.find(a2);
      if (it1 != position.end() && it2 != position.end()) {
        if (it1->second > it2->second) {
          cost.sbViolations++;
        }
      }
    }
  }
  
  return cost;
}

void ActionSchedulingPass::replaceScheduleOp(::sharp::txn::ModuleOp module,
                                            ArrayRef<StringRef> schedule) {
  OpBuilder builder(module.getContext());
  
  // Build action attributes
  SmallVector<Attribute> actionAttrs;
  for (auto action : schedule) {
    actionAttrs.push_back(FlatSymbolRefAttr::get(builder.getContext(), action));
  }
  
  // Find existing schedule op
  ::sharp::txn::ScheduleOp existingSchedule;
  for (auto schedOp : module.getOps<::sharp::txn::ScheduleOp>()) {
    existingSchedule = schedOp;
    break;
  }
  
  if (existingSchedule) {
    // Update existing schedule
    builder.setInsertionPoint(existingSchedule);
    auto conflictMatrix = existingSchedule.getConflictMatrix().value_or(DictionaryAttr());
    builder.create<::sharp::txn::ScheduleOp>(
        existingSchedule.getLoc(),
        builder.getArrayAttr(actionAttrs),
        conflictMatrix);
    existingSchedule.erase();
  } else {
    // Create new schedule at end of module
    builder.setInsertionPointToEnd(&module.getBody().front());
    builder.create<::sharp::txn::ScheduleOp>(
        module.getLoc(),
        builder.getArrayAttr(actionAttrs),
        DictionaryAttr::get(builder.getContext()));
  }
}

std::unique_ptr<mlir::Pass> createActionSchedulingPass() {
  return std::make_unique<ActionSchedulingPass>();
}

} // namespace sharp
} // namespace mlir