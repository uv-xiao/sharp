//===- PreSynthesisCheck.cpp - Pre-synthesis checking pass ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the pre-synthesis checking pass for Sharp Txn modules.
// It identifies non-synthesizable constructs that would prevent successful
// translation to FIRRTL/Verilog.
//
//===----------------------------------------------------------------------===//

#include "sharp/Analysis/Passes.h"
#include "sharp/Dialect/Txn/TxnOps.h"
#include "sharp/Dialect/Txn/TxnDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sharp-pre-synthesis-check"

namespace mlir {
namespace sharp {

#define GEN_PASS_DEF_PRESYNTHESISCHECK
#include "sharp/Analysis/Passes.h.inc"

namespace {

/// Pre-synthesis checking pass implementation
class PreSynthesisCheckPass : public impl::PreSynthesisCheckBase<PreSynthesisCheckPass> {
public:
  void runOnOperation() override;

private:
  /// Check if a primitive is synthesizable
  bool checkPrimitive(::sharp::txn::PrimitiveOp primitive);
  
  /// Check if a module has multi-cycle operations
  bool checkForMultiCycle(::sharp::txn::ModuleOp module);
  
  /// Check if a rule has multi-cycle timing
  bool checkRuleForMultiCycle(::sharp::txn::RuleOp rule);
  
  /// Check if a method has multi-cycle timing
  bool checkMethodForMultiCycle(Operation *method);
  
  /// Mark a module as non-synthesizable
  void markNonSynthesizable(::sharp::txn::ModuleOp module, StringRef reason);
  
  /// Mark a primitive as non-synthesizable
  void markNonSynthesizable(::sharp::txn::PrimitiveOp primitive, StringRef reason);
  
  /// Propagate non-synthesizable status through the hierarchy
  void propagateNonSynthesizable(ModuleOp topModule);
  
  /// Track non-synthesizable modules
  llvm::StringSet<> nonSynthesizableModules;
  
  /// Track reasons for non-synthesizability
  llvm::StringMap<std::string> nonSynthesizableReasons;
};

void PreSynthesisCheckPass::runOnOperation() {
  auto module = getOperation();
  
  LLVM_DEBUG(llvm::dbgs() << "Running pre-synthesis check on module\n");
  
  // First pass: Check all primitives and modules for non-synthesizable constructs
  module.walk([&](Operation *op) {
    if (auto primitive = dyn_cast<::sharp::txn::PrimitiveOp>(op)) {
      if (!checkPrimitive(primitive)) {
        markNonSynthesizable(primitive, "spec primitive type");
      }
    } else if (auto txnModule = dyn_cast<::sharp::txn::ModuleOp>(op)) {
      if (checkForMultiCycle(txnModule)) {
        markNonSynthesizable(txnModule, "contains multi-cycle operations");
      }
    }
  });
  
  // Second pass: Propagate non-synthesizable status through the hierarchy
  propagateNonSynthesizable(module);
  
  // Report all non-synthesizable modules
  for (const auto &entry : nonSynthesizableReasons) {
    module.emitError() << "Module '" << entry.first() 
                       << "' is non-synthesizable: " << entry.second;
  }
}

bool PreSynthesisCheckPass::checkPrimitive(::sharp::txn::PrimitiveOp primitive) {
  // Check if the primitive type is "hw" (synthesizable)
  auto typeAttr = primitive.getType();
  if (typeAttr != "hw") {
    LLVM_DEBUG(llvm::dbgs() << "Primitive " << primitive.getSymName() 
                            << " has type '" << typeAttr << "' (not 'hw')\n");
    return false;
  }
  
  // Check if the primitive has a FIRRTL implementation reference
  auto firrtlImpl = primitive->getAttrOfType<StringAttr>("firrtl.impl");
  if (!firrtlImpl) {
    LLVM_DEBUG(llvm::dbgs() << "Primitive " << primitive.getSymName() 
                            << " lacks firrtl.impl attribute\n");
    primitive.emitError() << "synthesizable primitive lacks firrtl.impl attribute";
    return false;
  }
  
  return true;
}

bool PreSynthesisCheckPass::checkForMultiCycle(::sharp::txn::ModuleOp module) {
  bool hasMultiCycle = false;
  
  // Check all rules
  module.walk([&](::sharp::txn::RuleOp rule) {
    if (checkRuleForMultiCycle(rule)) {
      hasMultiCycle = true;
      rule.emitError() << "multi-cycle rules are not yet supported for synthesis";
    }
  });
  
  // Check all methods
  module.walk([&](Operation *op) {
    if (isa<::sharp::txn::ValueMethodOp>(op) || 
        isa<::sharp::txn::ActionMethodOp>(op) ||
        isa<::sharp::txn::FirValueMethodOp>(op) ||
        isa<::sharp::txn::FirActionMethodOp>(op)) {
      if (checkMethodForMultiCycle(op)) {
        hasMultiCycle = true;
        op->emitError() << "multi-cycle methods are not yet supported for synthesis";
      }
    }
  });
  
  return hasMultiCycle;
}

bool PreSynthesisCheckPass::checkRuleForMultiCycle(::sharp::txn::RuleOp rule) {
  auto timingAttr = rule.getTiming();
  if (!timingAttr) {
    // No timing attribute means combinational (default)
    return false;
  }
  
  StringRef timing = timingAttr.value();
  if (timing != "combinational") {
    LLVM_DEBUG(llvm::dbgs() << "Rule " << rule.getSymName() 
                            << " has multi-cycle timing: " << timing << "\n");
    return true;
  }
  
  return false;
}

bool PreSynthesisCheckPass::checkMethodForMultiCycle(Operation *method) {
  auto timingAttr = method->getAttrOfType<StringAttr>("timing");
  if (!timingAttr) {
    // No timing attribute means combinational (default)
    return false;
  }
  
  StringRef timing = timingAttr.getValue();
  if (timing != "combinational") {
    StringAttr nameAttr;
    if (auto valueMeth = dyn_cast<::sharp::txn::ValueMethodOp>(method)) {
      nameAttr = valueMeth.getSymNameAttr();
    } else if (auto actionMeth = dyn_cast<::sharp::txn::ActionMethodOp>(method)) {
      nameAttr = actionMeth.getSymNameAttr();
    } else if (auto firValueMeth = dyn_cast<::sharp::txn::FirValueMethodOp>(method)) {
      nameAttr = firValueMeth.getSymNameAttr();
    } else if (auto firActionMeth = dyn_cast<::sharp::txn::FirActionMethodOp>(method)) {
      nameAttr = firActionMeth.getSymNameAttr();
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Method " << (nameAttr ? nameAttr.getValue() : "unknown")
                            << " has multi-cycle timing: " << timing << "\n");
    return true;
  }
  
  return false;
}

void PreSynthesisCheckPass::markNonSynthesizable(::sharp::txn::ModuleOp module, 
                                                 StringRef reason) {
  auto moduleName = module.getSymName();
  nonSynthesizableModules.insert(moduleName);
  nonSynthesizableReasons[moduleName] = reason.str();
  
  // Add nonsynthesizable attribute to the module
  module->setAttr("nonsynthesizable", UnitAttr::get(module.getContext()));
}

void PreSynthesisCheckPass::markNonSynthesizable(::sharp::txn::PrimitiveOp primitive, 
                                                 StringRef reason) {
  auto primitiveName = primitive.getSymName();
  nonSynthesizableModules.insert(primitiveName);
  nonSynthesizableReasons[primitiveName] = reason.str();
  
  // Add nonsynthesizable attribute to the primitive
  primitive->setAttr("nonsynthesizable", UnitAttr::get(primitive.getContext()));
}

void PreSynthesisCheckPass::propagateNonSynthesizable(ModuleOp topModule) {
  // Build a map of module instantiations
  llvm::StringMap<llvm::StringSet<>> moduleInstantiations;
  
  topModule.walk([&](::sharp::txn::InstanceOp inst) {
    auto parentModule = inst->getParentOfType<::sharp::txn::ModuleOp>();
    if (!parentModule)
      return;
    
    auto parentName = parentModule.getSymName();
    auto instancedModule = inst.getModuleName();
    moduleInstantiations[parentName].insert(instancedModule);
  });
  
  // Fixed-point iteration to propagate non-synthesizable status
  bool changed = true;
  while (changed) {
    changed = false;
    
    for (const auto &entry : moduleInstantiations) {
      StringRef parentModule = entry.first();
      
      // Skip if already marked as non-synthesizable
      if (nonSynthesizableModules.contains(parentModule))
        continue;
      
      // Check if any instantiated module is non-synthesizable
      for (const auto &instancedModule : entry.second) {
        if (nonSynthesizableModules.contains(instancedModule.first())) {
          nonSynthesizableModules.insert(parentModule);
          nonSynthesizableReasons[parentModule] = 
              "instantiates non-synthesizable module '" + 
              instancedModule.first().str() + "'";
          
          // Find and mark the actual module operation
          topModule.walk([&](::sharp::txn::ModuleOp mod) {
            if (mod.getSymName() == parentModule) {
              mod->setAttr("nonsynthesizable", UnitAttr::get(mod.getContext()));
            }
          });
          
          changed = true;
          break;
        }
      }
    }
  }
}

} // end anonymous namespace

std::unique_ptr<mlir::Pass> createPreSynthesisCheckPass() {
  return std::make_unique<PreSynthesisCheckPass>();
}

} // namespace sharp
} // namespace mlir