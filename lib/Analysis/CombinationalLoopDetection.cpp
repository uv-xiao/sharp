//===- CombinationalLoopDetection.cpp - Detect combinational loops -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the combinational loop detection pass for Sharp Txn
// modules. The pass builds a dependency graph of combinational paths and uses
// depth-first search to detect cycles.
//
//===----------------------------------------------------------------------===//

#include "sharp/Dialect/Txn/TxnOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sharp-combinational-loop-detection"

namespace mlir {
namespace sharp {

using ::sharp::txn::CallOp;
using ::sharp::txn::InstanceOp;
using ::sharp::txn::ValueMethodOp;
namespace txn = ::sharp::txn;

#define GEN_PASS_DEF_COMBINATIONALLOOPDETECTION
#include "sharp/Analysis/Passes.h.inc"

namespace {

/// Node in the dependency graph
struct DependencyNode {
  std::string name;
  SmallVector<DependencyNode*> dependencies;
  
  DependencyNode(StringRef name) : name(name.str()) {}
};

/// Graph for tracking combinational dependencies
class DependencyGraph {
  llvm::StringMap<std::unique_ptr<DependencyNode>> nodes;
  
public:
  /// Get or create a node
  DependencyNode* getOrCreateNode(StringRef name) {
    auto &node = nodes[name];
    if (!node) {
      node = std::make_unique<DependencyNode>(name);
    }
    return node.get();
  }
  
  /// Add a dependency edge from 'from' to 'to'
  void addDependency(StringRef from, StringRef to) {
    auto *fromNode = getOrCreateNode(from);
    auto *toNode = getOrCreateNode(to);
    fromNode->dependencies.push_back(toNode);
  }
  
  /// Check for cycles using DFS
  bool hasCycle(SmallVectorImpl<StringRef> &cyclePath) {
    DenseMap<DependencyNode*, int> state; // 0=unvisited, 1=visiting, 2=visited
    SmallVector<DependencyNode*> path;
    
    for (auto &[name, node] : nodes) {
      if (state[node.get()] == 0) {
        if (dfsHasCycle(node.get(), state, path, cyclePath)) {
          return true;
        }
      }
    }
    return false;
  }
  
private:
  bool dfsHasCycle(DependencyNode *node,
                   DenseMap<DependencyNode*, int> &state,
                   SmallVectorImpl<DependencyNode*> &path,
                   SmallVectorImpl<StringRef> &cyclePath) {
    state[node] = 1; // Mark as visiting
    path.push_back(node);
    
    for (auto *dep : node->dependencies) {
      if (state[dep] == 1) {
        // Found a cycle - build the cycle path
        bool inCycle = false;
        for (auto *pathNode : path) {
          if (pathNode == dep) inCycle = true;
          if (inCycle) {
            cyclePath.push_back(pathNode->name);
          }
        }
        cyclePath.push_back(dep->name); // Close the cycle
        return true;
      }
      
      if (state[dep] == 0) {
        if (dfsHasCycle(dep, state, path, cyclePath)) {
          return true;
        }
      }
    }
    
    path.pop_back();
    state[node] = 2; // Mark as visited
    return false;
  }
};

class CombinationalLoopDetectionPass
    : public impl::CombinationalLoopDetectionBase<CombinationalLoopDetectionPass> {
public:
  void runOnOperation() override;

private:
  /// Build dependency graph for a module
  void buildDependencyGraph(txn::ModuleOp module, DependencyGraph &graph);
  
  /// Add dependencies for a value method
  void analyzeValueMethod(ValueMethodOp method, txn::ModuleOp module,
                          DependencyGraph &graph);
  
  /// Add dependencies for method calls in a region
  void analyzeCallsInRegion(Region &region, StringRef callerName,
                            txn::ModuleOp module, DependencyGraph &graph);
  
  /// Get the full name of a method (module::method)
  std::string getMethodFullName(StringRef instance, StringRef method,
                                txn::ModuleOp currentModule);
  
  /// Check if a primitive creates combinational paths
  bool isCombinationalPrimitive(StringRef primitiveName);
};

void CombinationalLoopDetectionPass::runOnOperation() {
  auto module = getOperation();
  bool hasErrors = false;
  
  // Process each txn module
  module.walk([&](txn::ModuleOp txnModule) {
    DependencyGraph graph;
    buildDependencyGraph(txnModule, graph);
    
    // Check for cycles
    SmallVector<StringRef> cyclePath;
    if (graph.hasCycle(cyclePath)) {
      // Format the cycle path for error message
      std::string cycleStr;
      for (size_t i = 0; i < cyclePath.size(); ++i) {
        if (i > 0) cycleStr += " -> ";
        cycleStr += cyclePath[i].str();
      }
      
      txnModule.emitError("Combinational loop detected: ") << cycleStr;
      hasErrors = true;
    }
  });
  
  if (hasErrors) {
    signalPassFailure();
  }
}

void CombinationalLoopDetectionPass::buildDependencyGraph(txn::ModuleOp module,
                                                          DependencyGraph &graph) {
  // Analyze value methods - they create combinational paths
  module.walk([&](ValueMethodOp method) {
    analyzeValueMethod(method, module, graph);
  });
  
  // Analyze instances for combinational primitives
  module.walk([&](InstanceOp inst) {
    StringRef instanceName = inst.getSymName();
    StringRef instanceType = inst.getModuleName();
    
    // Check if this is a combinational primitive (e.g., Wire)
    if (isCombinationalPrimitive(instanceType)) {
      // For combinational primitives, reads depend on writes
      std::string readName = (instanceName + "::read").str();
      std::string writeName = (instanceName + "::write").str();
      graph.addDependency(readName, writeName);
    }
  });
}

void CombinationalLoopDetectionPass::analyzeValueMethod(ValueMethodOp method,
                                                         txn::ModuleOp module,
                                                         DependencyGraph &graph) {
  StringRef moduleName = module.getSymName();
  std::string methodFullName = (moduleName + "::" + method.getSymName()).str();
  
  // Analyze all calls within the method body
  analyzeCallsInRegion(method.getBody(), methodFullName, module, graph);
}

void CombinationalLoopDetectionPass::analyzeCallsInRegion(Region &region,
                                                           StringRef callerName,
                                                           txn::ModuleOp module,
                                                           DependencyGraph &graph) {
  region.walk([&](CallOp call) {
    // Parse the callee symbol reference
    auto callee = call.getCallee();
    StringRef instance, method;
    
    // Handle nested symbol references (instance::method)
    if (callee.getNestedReferences().size() == 1) {
      instance = callee.getRootReference();
      method = callee.getLeafReference();
    } else {
      // Direct method reference in current module
      instance = "";
      method = callee.getRootReference();
    }
    
    // Get the type of the instance to determine if it's a value method
    InstanceOp instOp = module.lookupSymbol<InstanceOp>(instance);
    if (!instOp) {
      // Could be a primitive or error - skip for now
      return;
    }
    
    // Get the called method's full name
    std::string calleeFullName = getMethodFullName(instance, method, module);
    
    // Add dependency: caller depends on callee
    graph.addDependency(callerName, calleeFullName);
    
    LLVM_DEBUG(llvm::dbgs() << "  Dependency: " << callerName << " -> "
                            << calleeFullName << "\n");
  });
}

std::string CombinationalLoopDetectionPass::getMethodFullName(
    StringRef instance, StringRef method, txn::ModuleOp currentModule) {
  // Look up the instance to get its module type
  if (auto instOp = currentModule.lookupSymbol<InstanceOp>(instance)) {
    StringRef moduleType = instOp.getModuleName();
    return (moduleType + "::" + method).str();
  }
  
  // For primitives or local references, use instance name
  return (instance + "::" + method).str();
}

bool CombinationalLoopDetectionPass::isCombinationalPrimitive(
    StringRef primitiveName) {
  // Wire is a combinational primitive - reads immediately reflect writes
  // Register is sequential - reads are delayed by a clock cycle
  return primitiveName == "Wire";
}

} // namespace

std::unique_ptr<mlir::Pass> createCombinationalLoopDetectionPass() {
  return std::make_unique<CombinationalLoopDetectionPass>();
}

} // namespace sharp
} // namespace mlir