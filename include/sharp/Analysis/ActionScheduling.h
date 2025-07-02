//===- ActionScheduling.h - Action Scheduling Pass -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Action Scheduling analysis pass which completes
// partial schedules to ensure all actions are properly ordered while
// minimizing conflicts.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_ANALYSIS_ACTIONSCHEDULING_H
#define SHARP_ANALYSIS_ACTIONSCHEDULING_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <unordered_map>
#include <string>

namespace mlir {
class ModuleOp;

namespace sharp {

// Forward declarations
namespace txn {
class ModuleOp;
class ScheduleOp;
} // namespace txn

/// Represents the conflict relationship between two actions
enum class ConflictRelation : int {
  SB = 0, // Sequential Before
  SA = 1, // Sequential After
  C = 2,  // Conflict
  CF = 3  // Conflict-Free
};

/// Conflict matrix storing relationships between actions
class ConflictMatrix {
public:
  ConflictRelation get(StringRef a1, StringRef a2) const;
  void set(StringRef a1, StringRef a2, ConflictRelation rel);
  bool hasRelation(StringRef a1, StringRef a2) const;
  
  // Get all action pairs with their relationships
  SmallVector<std::tuple<StringRef, StringRef, ConflictRelation>> getAllPairs() const;
  
private:
  // Use ordered pair as key for symmetric access
  using ActionPair = std::pair<std::string, std::string>;
  
  // Hash function for ActionPair
  struct PairHash {
    std::size_t operator()(const ActionPair &p) const {
      std::size_t h1 = std::hash<std::string>{}(p.first);
      std::size_t h2 = std::hash<std::string>{}(p.second);
      return h1 ^ (h2 << 1);
    }
  };
  
  std::unordered_map<ActionPair, ConflictRelation, PairHash> matrix;
  
  ActionPair makeKey(StringRef a1, StringRef a2) const;
};

/// Graph structure for scheduling dependencies
struct SchedulingGraph {
  // All actions in the module (using SetVector for consistent iteration order)
  SetVector<StringRef> actions;
  
  // Edges represent ordering constraints (a -> b means a must come before b)
  DenseMap<StringRef, SmallVector<StringRef>> mustPrecede;
  
  // Track partial schedule constraints
  DenseMap<StringRef, int> partialOrder;
};

/// Cost metrics for evaluating schedules
struct SchedulingCost {
  int sbViolations = 0;  // Count of SB relationships violated
  int saViolations = 0;  // Count of SA relationships violated  
  int conflicts = 0;     // Count of C relationships
};


} // namespace sharp
} // namespace mlir

#endif // SHARP_ANALYSIS_ACTIONSCHEDULING_H