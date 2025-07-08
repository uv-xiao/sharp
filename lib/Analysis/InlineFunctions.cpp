//===- InlineFunctions.cpp - Inline txn.func calls -----------------------===//
//
// Part of the Sharp Project.
//
//===----------------------------------------------------------------------===//

#include "sharp/Analysis/Passes.h"
#include "sharp/Dialect/Txn/TxnOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sharp-inline-functions"

namespace mlir {
namespace sharp {

#define GEN_PASS_CLASSES
#include "sharp/Analysis/Passes.h.inc"

namespace {

/// Pass to inline txn.func calls within txn modules
struct InlineFunctionsPass : public InlineFunctionsBase<InlineFunctionsPass> {
  void runOnOperation() override;
  
private:
  /// Inline a single function call
  LogicalResult inlineCall(::sharp::txn::FuncCallOp callOp);
  
  /// Check if a function can be inlined
  bool canInline(::sharp::txn::FuncOp funcOp);
  
  /// Collect all functions in a module
  void collectFunctions(::sharp::txn::ModuleOp moduleOp, 
                        DenseMap<StringRef, ::sharp::txn::FuncOp> &functions);
};

} // namespace

void InlineFunctionsPass::runOnOperation() {
  auto module = getOperation();
  
  // Process each txn.module
  module.walk([&](::sharp::txn::ModuleOp txnModule) {
    // Collect all functions in this module
    DenseMap<StringRef, ::sharp::txn::FuncOp> functions;
    collectFunctions(txnModule, functions);
    
    // Find all function calls and inline them
    SmallVector<::sharp::txn::FuncCallOp> callsToInline;
    txnModule.walk([&](::sharp::txn::FuncCallOp callOp) {
      callsToInline.push_back(callOp);
    });
    
    // Inline each call
    for (auto callOp : callsToInline) {
      if (failed(inlineCall(callOp))) {
        return signalPassFailure();
      }
    }
    
    // Remove functions that are no longer used
    SmallVector<::sharp::txn::FuncOp> funcsToRemove;
    for (auto &funcPair : functions) {
      auto funcOp = funcPair.second;
      // Check if the function is still used
      bool isUsed = false;
      txnModule.walk([&](::sharp::txn::FuncCallOp callOp) {
        if (callOp.getCallee() == funcOp.getSymName()) {
          isUsed = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      
      if (!isUsed) {
        funcsToRemove.push_back(funcOp);
      }
    }
    
    // Erase unused functions
    for (auto funcOp : funcsToRemove) {
      funcOp->erase();
    }
  });
}

void InlineFunctionsPass::collectFunctions(::sharp::txn::ModuleOp moduleOp,
                                          DenseMap<StringRef, ::sharp::txn::FuncOp> &functions) {
  moduleOp.walk([&](::sharp::txn::FuncOp funcOp) {
    functions[funcOp.getSymName()] = funcOp;
  });
}

bool InlineFunctionsPass::canInline(::sharp::txn::FuncOp funcOp) {
  // For now, we inline all functions
  // In the future, we might want to add heuristics
  return true;
}

LogicalResult InlineFunctionsPass::inlineCall(::sharp::txn::FuncCallOp callOp) {
  // Get the parent module
  auto moduleOp = callOp->getParentOfType<::sharp::txn::ModuleOp>();
  if (!moduleOp) {
    return callOp.emitError("func_call must be inside a txn.module");
  }
  
  // Look up the function
  auto funcName = callOp.getCallee();
  auto funcOp = moduleOp.lookupSymbol<::sharp::txn::FuncOp>(funcName);
  if (!funcOp) {
    return callOp.emitError("cannot find function '")
           << funcName << "' in module";
  }
  
  // Check if we can inline this function
  if (!canInline(funcOp)) {
    return success(); // Skip inlining
  }
  
  // Check argument count matches
  if (callOp.getOperands().size() != funcOp.getArgumentTypes().size()) {
    return callOp.emitError("argument count mismatch: call has ")
           << callOp.getOperands().size() << " arguments but function expects "
           << funcOp.getArgumentTypes().size();
  }
  
  // Check that the function has exactly one block
  if (!funcOp.getBody().hasOneBlock()) {
    return funcOp.emitError("function must have exactly one block for inlining");
  }
  
  auto &funcBlock = funcOp.getBody().front();
  
  // Create a mapping from function arguments to call operands
  IRMapping mapping;
  for (auto [arg, operand] : llvm::zip(funcBlock.getArguments(), callOp.getOperands())) {
    mapping.map(arg, operand);
  }
  
  // Clone the function body
  OpBuilder builder(callOp);
  SmallVector<Value> results;
  
  for (auto &op : funcBlock.without_terminator()) {
    auto *clonedOp = builder.clone(op, mapping);
    
    // Map the results
    for (auto [oldResult, newResult] : 
         llvm::zip(op.getResults(), clonedOp->getResults())) {
      mapping.map(oldResult, newResult);
    }
  }
  
  // Handle the terminator (txn.return)
  auto returnOp = dyn_cast<::sharp::txn::ReturnOp>(funcBlock.getTerminator());
  if (!returnOp) {
    return funcOp.emitError("function must be terminated with txn.return");
  }
  
  // Map the return values
  for (auto operand : returnOp.getOperands()) {
    auto mappedValue = mapping.lookupOrDefault(operand);
    results.push_back(mappedValue);
  }
  
  // Replace the call with the results
  callOp.replaceAllUsesWith(results);
  callOp->erase();
  
  LLVM_DEBUG(llvm::dbgs() << "Inlined function call to '" << funcName << "'\n");
  
  return success();
}

std::unique_ptr<mlir::Pass> createInlineFunctionsPass() {
  return std::make_unique<InlineFunctionsPass>();
}

} // namespace sharp
} // namespace mlir