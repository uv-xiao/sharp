//===- TxnToFuncPass.cpp - Txn to Func lowering pass ---------------------===//
//
// This file implements the pass to lower txn dialect to func dialect.
//
//===----------------------------------------------------------------------===//

#include "sharp/Conversion/TxnToFunc/TxnToFunc.h"
#include "sharp/Dialect/Txn/TxnOps.h"
#include "sharp/Dialect/Txn/TxnAttrs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace sharp {

#define GEN_PASS_DEF_CONVERTTXNTOFUNCPASS
#include "sharp/Conversion/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

/// Type converter for Txn to Func conversion.
class TxnToFuncTypeConverter : public mlir::TypeConverter {
public:
  TxnToFuncTypeConverter() {
    // All types are kept as-is for now
    addConversion([](mlir::Type type) { return type; });
  }
};

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Convert txn.module to func.func operations
struct ModuleToFuncPattern : public mlir::OpConversionPattern<::sharp::txn::ModuleOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(::sharp::txn::ModuleOp moduleOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = moduleOp.getLoc();
    
    // Create a main function for the module
    auto funcType = rewriter.getFunctionType({}, {});
    auto funcOp = rewriter.create<mlir::func::FuncOp>(
        loc, (moduleOp.getName() + "_main").str(), funcType);
    
    // Create entry block
    auto *entryBlock = funcOp.addEntryBlock();
    rewriter.setInsertionPointToStart(entryBlock);
    
    // Convert module body
    // For now, we'll create a simple structure that calls methods
    // TODO: Implement proper state management and method dispatch
    
    // Add return
    rewriter.create<mlir::func::ReturnOp>(loc);
    
    // Convert methods and rules within the module
    for (auto &op : moduleOp.getBodyBlock()->getOperations()) {
      if (auto valueMethod = mlir::dyn_cast<::sharp::txn::ValueMethodOp>(op)) {
        convertValueMethod(valueMethod, rewriter);
      } else if (auto actionMethod = mlir::dyn_cast<::sharp::txn::ActionMethodOp>(op)) {
        convertActionMethod(actionMethod, rewriter);
      } else if (auto rule = mlir::dyn_cast<::sharp::txn::RuleOp>(op)) {
        convertRule(rule, rewriter);
      }
    }
    
    rewriter.eraseOp(moduleOp);
    return mlir::success();
  }
  
private:
  void convertValueMethod(::sharp::txn::ValueMethodOp method,
                         mlir::ConversionPatternRewriter &rewriter) const {
    auto loc = method.getLoc();
    auto parentModule = method->getParentOfType<::sharp::txn::ModuleOp>();
    
    // Create a function for the value method
    auto funcName = parentModule.getName().str() + "_" + method.getName().str();
    auto funcType = rewriter.getFunctionType(method.getArgumentTypes(),
                                           method.getResultTypes());
    
    rewriter.setInsertionPoint(method->getParentOp());
    auto funcOp = rewriter.create<mlir::func::FuncOp>(loc, funcName, funcType);
    
    // Clone the method body into the function
    mlir::IRMapping mapping;
    method.getBody().cloneInto(&funcOp.getBody(), mapping);
    
    // Update block arguments
    auto &entryBlock = funcOp.getBody().front();
    for (auto [oldArg, newArg] : llvm::zip(method.getBody().getArguments(),
                                           entryBlock.getArguments())) {
      oldArg.replaceAllUsesWith(newArg);
    }
  }
  
  void convertActionMethod(::sharp::txn::ActionMethodOp method,
                          mlir::ConversionPatternRewriter &rewriter) const {
    auto loc = method.getLoc();
    auto parentModule = method->getParentOfType<::sharp::txn::ModuleOp>();
    
    // Create a function for the action method
    auto funcName = parentModule.getName().str() + "_" + method.getName().str();
    auto funcType = rewriter.getFunctionType(method.getArgumentTypes(), {});
    
    rewriter.setInsertionPoint(method->getParentOp());
    auto funcOp = rewriter.create<mlir::func::FuncOp>(loc, funcName, funcType);
    
    // Clone the method body into the function
    mlir::IRMapping mapping;
    method.getBody().cloneInto(&funcOp.getBody(), mapping);
    
    // Update block arguments
    auto &entryBlock = funcOp.getBody().front();
    for (auto [oldArg, newArg] : llvm::zip(method.getBody().getArguments(),
                                           entryBlock.getArguments())) {
      oldArg.replaceAllUsesWith(newArg);
    }
  }
  
  void convertRule(::sharp::txn::RuleOp rule,
                  mlir::ConversionPatternRewriter &rewriter) const {
    auto loc = rule.getLoc();
    auto parentModule = rule->getParentOfType<::sharp::txn::ModuleOp>();
    
    // Create a function for the rule
    auto funcName = parentModule.getName().str() + "_rule_" + 
                   rule.getSymName().str();
    auto funcType = rewriter.getFunctionType({}, {});
    
    rewriter.setInsertionPoint(rule->getParentOp());
    auto funcOp = rewriter.create<mlir::func::FuncOp>(loc, funcName, funcType);
    
    // Clone the rule body into the function
    mlir::IRMapping mapping;
    rule.getBody().cloneInto(&funcOp.getBody(), mapping);
  }
};

/// Convert txn.return to func.return
struct ReturnToFuncReturnPattern : public mlir::OpConversionPattern<::sharp::txn::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(::sharp::txn::ReturnOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, adaptor.getOperands());
    return mlir::success();
  }
};

/// Convert txn.yield to func.return (for void returns)
struct YieldToFuncReturnPattern : public mlir::OpConversionPattern<::sharp::txn::YieldOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(::sharp::txn::YieldOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op);
    return mlir::success();
  }
};

/// Convert txn.call to func.call
struct CallToFuncCallPattern : public mlir::OpConversionPattern<::sharp::txn::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(::sharp::txn::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Get the parent module name
    auto parentModule = op->getParentOfType<::sharp::txn::ModuleOp>();
    if (!parentModule) {
      return mlir::failure();
    }
    
    // Construct the function name
    auto funcName = parentModule.getName().str() + "_" + op.getCallee().getRootReference().str();
    
    // Create func.call
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, funcName, op.getResultTypes(), adaptor.getOperands());
    
    return mlir::success();
  }
};

/// Convert txn.if to scf.if (requires SCF dialect)
struct IfToSCFIfPattern : public mlir::OpConversionPattern<::sharp::txn::IfOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(::sharp::txn::IfOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // For now, we'll need to handle this with func dialect constructs
    // This is a simplified version that doesn't handle all cases
    
    // TODO: Implement proper if conversion using cf dialect operations
    return mlir::failure();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct ConvertTxnToFuncPass
    : public impl::ConvertTxnToFuncPassBase<ConvertTxnToFuncPass> {
  
  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());
    
    // Mark func dialect as legal
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    
    // Mark txn dialect as illegal
    target.addIllegalDialect<::sharp::txn::TxnDialect>();
    
    // Allow txn operations within func bodies temporarily
    target.addDynamicallyLegalOp<::sharp::txn::ModuleOp>([](::sharp::txn::ModuleOp op) {
      return false; // Always convert
    });
    
    TxnToFuncTypeConverter typeConverter;
    mlir::RewritePatternSet patterns(&getContext());
    
    populateTxnToFuncConversionPatterns(typeConverter, patterns);
    
    if (failed(applyPartialConversion(getOperation(), target,
                                     std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

void populateTxnToFuncConversionPatterns(mlir::TypeConverter &typeConverter,
                                        mlir::RewritePatternSet &patterns) {
  patterns.add<ModuleToFuncPattern, ReturnToFuncReturnPattern,
               YieldToFuncReturnPattern, CallToFuncCallPattern>(
      typeConverter, patterns.getContext());
}

// Pass creation is handled by generated code

} // namespace sharp
} // namespace mlir