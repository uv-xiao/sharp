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
    
    // Find parent module name - either from parent or attribute
    std::string funcName;
    if (auto parentModule = method->getParentOfType<::sharp::txn::ModuleOp>()) {
      funcName = parentModule.getName().str() + "_" + method.getName().str();
    } else if (auto parentModuleAttr = method->getAttrOfType<mlir::StringAttr>("txn.parent_module")) {
      funcName = parentModuleAttr.getValue().str() + "_" + method.getName().str();
    } else {
      // Fallback to just method name
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
    
    // Find parent module name - either from parent or attribute
    std::string funcName;
    if (auto parentModule = method->getParentOfType<::sharp::txn::ModuleOp>()) {
      funcName = parentModule.getName().str() + "_" + method.getName().str();
    } else if (auto parentModuleAttr = method->getAttrOfType<mlir::StringAttr>("txn.parent_module")) {
      funcName = parentModuleAttr.getValue().str() + "_" + method.getName().str();
    } else {
      // Fallback to just method name
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
    
    // Find parent module name - either from parent or attribute
    std::string funcName;
    if (auto parentModule = rule->getParentOfType<::sharp::txn::ModuleOp>()) {
      funcName = parentModule.getName().str() + "_rule_" + rule.getSymName().str();
    } else if (auto parentModuleAttr = rule->getAttrOfType<mlir::StringAttr>("txn.parent_module")) {
      funcName = parentModuleAttr.getValue().str() + "_rule_" + rule.getSymName().str();
    } else {
      // Fallback to just rule name
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
    
    // Store module name on methods before moving them out
    auto moduleName = moduleOp.getName();
    for (auto &op : llvm::make_early_inc_range(moduleOp.getBodyBlock()->getOperations())) {
      if (isa<::sharp::txn::ValueMethodOp, ::sharp::txn::ActionMethodOp, 
              ::sharp::txn::RuleOp>(op)) {
        // Store the parent module name as an attribute
        op.setAttr("txn.parent_module", rewriter.getStringAttr(moduleName));
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
    // Check if we're inside an scf.if - if so, we need special handling
    if (op->getParentOfType<mlir::scf::IfOp>()) {
      // In an scf.if, a return should become a func.return
      // This will cause early exit from the containing function
      rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, adaptor.getOperands());
    } else {
      // Normal case - convert to func.return
      rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, adaptor.getOperands());
    }
    return mlir::success();
  }
};

/// Convert txn.yield to scf.yield (when inside scf.if) or func.return
struct YieldToSCFYieldPattern : public mlir::OpConversionPattern<::sharp::txn::YieldOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(::sharp::txn::YieldOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Check if we're inside an scf.if
    if (op->getParentOfType<mlir::scf::IfOp>()) {
      // Convert to scf.yield
      rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, adaptor.getOperands());
    } else {
      // Otherwise, this is probably an error - yields should only be in if/while constructs
      return mlir::failure();
    }
    return mlir::success();
  }
};

/// Convert txn.call to func.call
struct CallToFuncCallPattern : public mlir::OpConversionPattern<::sharp::txn::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(::sharp::txn::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Get the parent module name - either from module or from parent function's attribute
    std::string moduleName;
    if (auto parentModule = op->getParentOfType<::sharp::txn::ModuleOp>()) {
      moduleName = parentModule.getName().str();
    } else if (auto parentFunc = op->getParentOfType<mlir::func::FuncOp>()) {
      // Extract module name from function name (format: ModuleName_methodName)
      auto funcName = parentFunc.getName();
      auto underscorePos = funcName.find('_');
      if (underscorePos != StringRef::npos) {
        moduleName = funcName.substr(0, underscorePos).str();
      } else {
        return mlir::failure();
      }
    } else {
      return mlir::failure();
    }
    
    // Construct the function name
    auto funcName = moduleName + "_" + op.getCallee().getRootReference().str();
    
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
      auto &srcBlock = op.getThenRegion().front();
      auto &dstBlock = scfIf.getThenRegion().front();
      
      // Set insertion point at the beginning of destination block
      rewriter.setInsertionPointToStart(&dstBlock);
      
      // Move all operations from source to destination
      for (auto &op : llvm::make_early_inc_range(srcBlock)) {
        op.moveBefore(&dstBlock, dstBlock.end());
      }
    }
    
    // Convert the else region if it exists
    if (!op.getElseRegion().empty() && !scfIf.getElseRegion().empty()) {
      auto &srcBlock = op.getElseRegion().front();
      auto &dstBlock = scfIf.getElseRegion().front();
      
      // Set insertion point at the beginning of destination block
      rewriter.setInsertionPointToStart(&dstBlock);
      
      // Move all operations from source to destination
      for (auto &op : llvm::make_early_inc_range(srcBlock)) {
        op.moveBefore(&dstBlock, dstBlock.end());
      }
    }
    
    // Replace the original operation with the results
    rewriter.replaceOp(op, scfIf.getResults());
    return mlir::success();
  }
};

/// Convert txn.abort to func.return (early exit)
struct AbortToReturnPattern : public mlir::OpConversionPattern<::sharp::txn::AbortOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(::sharp::txn::AbortOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
    if (!funcOp) {
      return mlir::failure();
    }
    
    // Create a return with appropriate values for the function signature
    mlir::SmallVector<mlir::Value> returnValues;
    for (auto resultType : funcOp.getResultTypes()) {
      if (auto intType = dyn_cast<mlir::IntegerType>(resultType)) {
        auto zero = rewriter.create<mlir::arith::ConstantIntOp>(
            op.getLoc(), 0, intType);
        returnValues.push_back(zero);
      } else {
        return mlir::failure();
      }
    }
    
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, returnValues);
    return mlir::success();
  }
};

/// Erase txn.schedule operations
struct ScheduleErasePattern : public mlir::OpConversionPattern<::sharp::txn::ScheduleOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(::sharp::txn::ScheduleOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Just erase the schedule operation
    rewriter.eraseOp(op);
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
    target.addIllegalOp<::sharp::txn::CallOp>();
    target.addIllegalOp<::sharp::txn::IfOp>();
    target.addIllegalOp<::sharp::txn::AbortOp>();
    
    // Special handling for txn.yield - it's legal in rule functions
    target.addDynamicallyLegalOp<::sharp::txn::YieldOp>([](::sharp::txn::YieldOp op) {
      // Check if we're in a function that was converted from a rule
      auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
      if (funcOp && funcOp.getName().contains("_rule_")) {
        return true;  // Legal in rule functions
      }
      return false;  // Illegal elsewhere
    });
    
    // Mark txn.schedule as illegal too
    target.addIllegalOp<::sharp::txn::ScheduleOp>();
    
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
               ReturnToFuncReturnPattern, YieldToSCFYieldPattern,
               CallToFuncCallPattern, IfToSCFIfPattern,
               AbortToReturnPattern, ScheduleErasePattern>(
      typeConverter, patterns.getContext());
}

// Pass creation is handled by generated code

} // namespace sharp
} // namespace mlir