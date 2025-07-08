//===- CollectPrimitiveActions.cpp - Collect primitive action calls -------===//
//
// Part of the Sharp project.
//
//===----------------------------------------------------------------------===//
//
// This pass collects all primitive action calls made by each action in a module.
// It adds a "primitive_calls" attribute to each action containing the list of
// primitive instance paths that the action calls methods on.
//
//===----------------------------------------------------------------------===//

#include "sharp/Analysis/Passes.h"
#include "sharp/Dialect/Txn/TxnOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "collect-primitive-actions"

namespace mlir {
namespace sharp {

#define GEN_PASS_DEF_SHARPANALYSISCOLLECTPRIMITIVEACTIONSPASS
#include "sharp/Analysis/Passes.h.inc"

namespace {

struct CollectPrimitiveActionsPass
    : public impl::SharpAnalysisCollectPrimitiveActionsPassBase<
          CollectPrimitiveActionsPass> {
  
  void runOnOperation() override;
  
private:
  /// Collect all primitive calls made by an action
  void collectPrimitiveCalls(Operation *action, ::sharp::txn::ModuleOp module);
  
  /// Trace through instance calls to find primitive actions
  void tracePrimitiveCall(::sharp::txn::CallOp call, ::sharp::txn::ModuleOp currentModule,
                         SmallVectorImpl<std::string> &primitives,
                         StringRef pathPrefix = "");
};

void CollectPrimitiveActionsPass::runOnOperation() {
  auto moduleOp = getOperation();
  
  // Process each txn module
  moduleOp.walk([&](::sharp::txn::ModuleOp txnModule) {
    LLVM_DEBUG(llvm::dbgs() << "Collecting primitive actions in module: " 
               << txnModule.getSymName() << "\n");
    
    // Process each rule
    txnModule.walk([&](::sharp::txn::RuleOp rule) {
      collectPrimitiveCalls(rule, txnModule);
    });
    
    // Process each action method
    txnModule.walk([&](::sharp::txn::ActionMethodOp actionMethod) {
      collectPrimitiveCalls(actionMethod, txnModule);
    });
  });
}

void CollectPrimitiveActionsPass::collectPrimitiveCalls(Operation *action,
                                                        ::sharp::txn::ModuleOp module) {
  SmallVector<std::string> primitiveCalls;
  llvm::SmallSet<std::string, 8> seen;
  
  LLVM_DEBUG(llvm::dbgs() << "Collecting primitive calls for action\n");
  
  // Walk through the action body to find all calls
  action->walk([&](::sharp::txn::CallOp call) {
    LLVM_DEBUG(llvm::dbgs() << "  Found call: " << call << "\n");
    tracePrimitiveCall(call, module, primitiveCalls);
  });
  
  // Remove duplicates
  SmallVector<std::string> uniqueCalls;
  for (const auto &call : primitiveCalls) {
    if (seen.insert(call).second) {
      uniqueCalls.push_back(call);
    }
  }
  
  // Add attribute if we found any primitive calls
  if (!uniqueCalls.empty()) {
    SmallVector<Attribute> attrs;
    for (const auto &call : uniqueCalls) {
      attrs.push_back(StringAttr::get(&getContext(), call));
    }
    action->setAttr("primitive_calls", ArrayAttr::get(&getContext(), attrs));
    
    LLVM_DEBUG({
      llvm::dbgs() << "Action ";
      if (auto rule = dyn_cast<::sharp::txn::RuleOp>(action))
        llvm::dbgs() << rule.getSymName();
      else if (auto method = dyn_cast<::sharp::txn::ActionMethodOp>(action))
        llvm::dbgs() << method.getSymName();
      llvm::dbgs() << " calls primitives: ";
      for (const auto &call : uniqueCalls) {
        llvm::dbgs() << call << " ";
      }
      llvm::dbgs() << "\n";
    });
  }
}

void CollectPrimitiveActionsPass::tracePrimitiveCall(
    ::sharp::txn::CallOp call, ::sharp::txn::ModuleOp currentModule,
    SmallVectorImpl<std::string> &primitives, StringRef pathPrefix) {
  
  auto callee = call.getCalleeAttr();
  
  LLVM_DEBUG(llvm::dbgs() << "  Tracing call: " << callee << " with prefix: " << pathPrefix << "\n");
  
  // Handle nested references (instance::method calls)
  if (callee.getNestedReferences().size() == 1) {
    StringRef instanceName = callee.getRootReference();
    StringRef methodName = cast<FlatSymbolRefAttr>(callee.getNestedReferences()[0]).getValue();
    
    LLVM_DEBUG(llvm::dbgs() << "    Instance: " << instanceName << " Method: " << methodName << "\n");
    
    // Find the instance
    ::sharp::txn::InstanceOp instance = nullptr;
    currentModule.walk([&](::sharp::txn::InstanceOp inst) {
      if (inst.getSymName() == instanceName) {
        instance = inst;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    
    if (!instance) return;
    
    // Get the referenced module
    StringRef moduleName = instance.getModuleName();
    
    // Check if this is a primitive
    auto parentModule = currentModule->getParentOfType<ModuleOp>();
    if (!parentModule) return;
    
    bool isPrimitive = false;
    ::sharp::txn::ModuleOp referencedModule = nullptr;
    
    parentModule.walk([&](::sharp::txn::PrimitiveOp prim) {
      if (prim.getSymName() == moduleName) {
        isPrimitive = true;
        LLVM_DEBUG(llvm::dbgs() << "    Found primitive: " << moduleName << "\n");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    
    // Check if this is a known primitive name even if not defined
    if (!isPrimitive && (moduleName == "Register" || moduleName == "Wire" || 
                        moduleName == "FIFO" || moduleName == "Memory" ||
                        moduleName == "SpecFIFO" || moduleName == "SpecMemory")) {
      isPrimitive = true;
      LLVM_DEBUG(llvm::dbgs() << "    Treating as primitive (known name): " << moduleName << "\n");
    }
    
    if (!isPrimitive) {
      // Find the referenced module
      parentModule.walk([&](::sharp::txn::ModuleOp mod) {
        if (mod.getSymName() == moduleName) {
          referencedModule = mod;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
    }
    
    std::string fullPath = pathPrefix.empty() ? 
        instanceName.str() : (pathPrefix.str() + "::" + instanceName.str());
    
    if (isPrimitive) {
      // This is a primitive call - record it
      std::string primitiveCall = fullPath + "::" + methodName.str();
      primitives.push_back(primitiveCall);
      LLVM_DEBUG(llvm::dbgs() << "  Found primitive call: " << primitiveCall << "\n");
    } else if (referencedModule) {
      // Recurse into the referenced module to find primitive calls
      // Look for all methods in the module that match the called method
      referencedModule.walk([&](Operation *op) {
        if (auto valueMethod = dyn_cast<::sharp::txn::ValueMethodOp>(op)) {
          if (valueMethod.getSymName() == methodName) {
            // Trace calls within this method
            valueMethod.walk([&](::sharp::txn::CallOp innerCall) {
              tracePrimitiveCall(innerCall, referencedModule, primitives, fullPath);
            });
          }
        } else if (auto actionMethod = dyn_cast<::sharp::txn::ActionMethodOp>(op)) {
          if (actionMethod.getSymName() == methodName) {
            // Trace calls within this method
            actionMethod.walk([&](::sharp::txn::CallOp innerCall) {
              tracePrimitiveCall(innerCall, referencedModule, primitives, fullPath);
            });
          }
        }
      });
    }
  }
}

} // namespace

std::unique_ptr<mlir::Pass> createCollectPrimitiveActionsPass() {
  return std::make_unique<CollectPrimitiveActionsPass>();
}

} // namespace sharp
} // namespace mlir