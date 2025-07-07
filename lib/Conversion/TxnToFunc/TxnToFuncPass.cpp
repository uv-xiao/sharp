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
#include "mlir/Dialect/SCF/IR/SCF.h"
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

/// Convert txn.value_method to func.func
struct ValueMethodToFuncPattern : public mlir::OpConversionPattern<::sharp::txn::ValueMethodOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(::sharp::txn::ValueMethodOp method, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = method.getLoc();
    
    // Find parent module - it might be a direct parent or we might need to look up
    auto parentModule = method->getParentOfType<::sharp::txn::ModuleOp>();
    std::string funcName;
    if (parentModule) {
      funcName = parentModule.getName().str() + "_" + method.getName().str();
    } else {
      // If method was moved out, use just the method name
      funcName = method.getName().str();
    }
    
    auto funcType = rewriter.getFunctionType(method.getArgumentTypes(),
                                           method.getResultTypes());
    
    // Create the function
    auto funcOp = rewriter.create<mlir::func::FuncOp>(loc, funcName, funcType);
    
    // Clone the body with proper mapping
    rewriter.inlineRegionBefore(method.getBody(), funcOp.getBody(), funcOp.end());
    
    rewriter.eraseOp(method);
    return mlir::success();
  }
};

/// Convert txn.action_method to func.func
struct ActionMethodToFuncPattern : public mlir::OpConversionPattern<::sharp::txn::ActionMethodOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(::sharp::txn::ActionMethodOp method, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = method.getLoc();
    
    // Find parent module - it might be a direct parent or we might need to look up
    auto parentModule = method->getParentOfType<::sharp::txn::ModuleOp>();
    std::string funcName;
    if (parentModule) {
      funcName = parentModule.getName().str() + "_" + method.getName().str();
    } else {
      // If method was moved out, use just the method name
      funcName = method.getName().str();
    }
    
    auto funcType = rewriter.getFunctionType(method.getArgumentTypes(), 
                                           method.getResultTypes());
    
    // Create the function
    auto funcOp = rewriter.create<mlir::func::FuncOp>(loc, funcName, funcType);
    
    // Clone the body
    rewriter.inlineRegionBefore(method.getBody(), funcOp.getBody(), funcOp.end());
    
    rewriter.eraseOp(method);
    return mlir::success();
  }
};

/// Convert txn.rule to func.func
struct RuleToFuncPattern : public mlir::OpConversionPattern<::sharp::txn::RuleOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(::sharp::txn::RuleOp rule, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = rule.getLoc();
    
    // Find parent module - it might be a direct parent or we might need to look up
    auto parentModule = rule->getParentOfType<::sharp::txn::ModuleOp>();
    std::string funcName;
    if (parentModule) {
      funcName = parentModule.getName().str() + "_rule_" + rule.getSymName().str();
    } else {
      // If rule was moved out, use just the rule name
      funcName = "rule_" + rule.getSymName().str();
    }
    
    auto funcType = rewriter.getFunctionType({}, {});
    
    // Create the function
    auto funcOp = rewriter.create<mlir::func::FuncOp>(loc, funcName, funcType);
    
    // Clone the body
    rewriter.inlineRegionBefore(rule.getBody(), funcOp.getBody(), funcOp.end());
    
    rewriter.eraseOp(rule);
    return mlir::success();
  }
};

/// Convert txn.module to func.func operations
struct ModuleToFuncPattern : public mlir::OpConversionPattern<::sharp::txn::ModuleOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(::sharp::txn::ModuleOp moduleOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = moduleOp.getLoc();
    
    // Create a main function for the module
    rewriter.setInsertionPoint(moduleOp);
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
    
    // Move methods out of the module before erasing it
    rewriter.setInsertionPoint(moduleOp);
    for (auto &op : llvm::make_early_inc_range(moduleOp.getBodyBlock()->getOperations())) {
      if (isa<::sharp::txn::ValueMethodOp, ::sharp::txn::ActionMethodOp, 
              ::sharp::txn::RuleOp>(op)) {
        op.moveBefore(moduleOp);
      }
    }
    
    // Erase the module
    rewriter.eraseOp(moduleOp);
    return mlir::success();
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

/// Convert txn.if to scf.if
struct IfToSCFIfPattern : public mlir::OpConversionPattern<::sharp::txn::IfOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(::sharp::txn::IfOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Get the condition
    auto condition = adaptor.getCondition();
    
    // Create the scf.if operation
    auto scfIf = rewriter.create<mlir::scf::IfOp>(
        op.getLoc(), op.getResultTypes(), condition,
        /*withElseRegion=*/!op.getElseRegion().empty());
    
    // Convert the then region
    if (!op.getThenRegion().empty()) {
      rewriter.setInsertionPointToStart(&scfIf.getThenRegion().front());
      
      mlir::IRMapping mapping;
      for (auto &innerOp : op.getThenRegion().front()) {
        if (auto yieldOp = dyn_cast<::sharp::txn::YieldOp>(&innerOp)) {
          // Convert txn.yield to scf.yield
          rewriter.create<mlir::scf::YieldOp>(yieldOp.getLoc(), 
                                              yieldOp.getOperands());
        } else {
          // Clone other operations
          rewriter.clone(innerOp, mapping);
        }
      }
    }
    
    // Convert the else region if it exists
    if (!op.getElseRegion().empty() && !scfIf.getElseRegion().empty()) {
      rewriter.setInsertionPointToStart(&scfIf.getElseRegion().front());
      
      mlir::IRMapping mapping;
      for (auto &innerOp : op.getElseRegion().front()) {
        if (auto yieldOp = dyn_cast<::sharp::txn::YieldOp>(&innerOp)) {
          // Convert txn.yield to scf.yield
          rewriter.create<mlir::scf::YieldOp>(yieldOp.getLoc(), 
                                              yieldOp.getOperands());
        } else {
          // Clone other operations
          rewriter.clone(innerOp, mapping);
        }
      }
    }
    
    // Replace the original operation with the results
    rewriter.replaceOp(op, scfIf.getResults());
    return mlir::success();
  }
};

/// Convert txn.abort to func.return (empty return for early exit)
struct AbortToReturnPattern : public mlir::OpConversionPattern<::sharp::txn::AbortOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(::sharp::txn::AbortOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Check if we're inside an scf.if - if so, we shouldn't convert yet
    if (op->getParentOfType<mlir::scf::IfOp>()) {
      // The abort should be handled by the control flow transformation
      // For now, just remove it as it represents early termination
      rewriter.eraseOp(op);
      return mlir::success();
    }
    
    // Find the parent function
    auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
    if (!funcOp) {
      return mlir::failure();
    }
    
    // Create a return with appropriate values
    // For now, we'll return default values for each return type
    mlir::SmallVector<mlir::Value> returnValues;
    for (auto resultType : funcOp.getResultTypes()) {
      // Create zero/default value for each return type
      if (auto intType = dyn_cast<mlir::IntegerType>(resultType)) {
        auto zero = rewriter.create<mlir::arith::ConstantIntOp>(
            op.getLoc(), 0, intType);
        returnValues.push_back(zero);
      } else {
        // For other types, we might need more sophisticated handling
        return mlir::failure();
      }
    }
    
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, returnValues);
    return mlir::success();
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
    target.addLegalDialect<mlir::scf::SCFDialect>();
    
    // Mark specific txn operations as illegal
    target.addIllegalOp<::sharp::txn::ModuleOp>();
    target.addIllegalOp<::sharp::txn::ValueMethodOp>();
    target.addIllegalOp<::sharp::txn::ActionMethodOp>();
    target.addIllegalOp<::sharp::txn::RuleOp>();
    target.addIllegalOp<::sharp::txn::ReturnOp>();
    target.addIllegalOp<::sharp::txn::YieldOp>();
    target.addIllegalOp<::sharp::txn::CallOp>();
    target.addIllegalOp<::sharp::txn::IfOp>();
    target.addIllegalOp<::sharp::txn::AbortOp>();
    
    // Allow txn.schedule and other structural ops temporarily
    target.addLegalOp<::sharp::txn::ScheduleOp>();
    
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
  patterns.add<ModuleToFuncPattern, ValueMethodToFuncPattern,
               ActionMethodToFuncPattern, RuleToFuncPattern,
               ReturnToFuncReturnPattern, YieldToFuncReturnPattern,
               CallToFuncCallPattern, IfToSCFIfPattern,
               AbortToReturnPattern>(
      typeConverter, patterns.getContext());
}

// Pass creation is handled by generated code

} // namespace sharp
} // namespace mlir