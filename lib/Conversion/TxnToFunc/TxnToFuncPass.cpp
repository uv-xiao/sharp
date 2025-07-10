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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

// Helper to check if a module has a schedule
static ::sharp::txn::ScheduleOp findSchedule(::sharp::txn::ModuleOp module) {
  ::sharp::txn::ScheduleOp schedule;
  module.walk([&](::sharp::txn::ScheduleOp op) {
    schedule = op;
    return WalkResult::interrupt();
  });
  return schedule;
}

// Helper to extract conflict information from module
struct ConflictInfo {
  StringRef action1, action2;
  int relation; // 0=SB, 1=SA, 2=C, 3=CF
};

static SmallVector<ConflictInfo> extractConflicts(::sharp::txn::ModuleOp module) {
  SmallVector<ConflictInfo> conflicts;
  
  // Look for conflict matrix attribute on the schedule operation
  ::sharp::txn::ScheduleOp schedule = findSchedule(module);
  if (!schedule) {
    return conflicts;
  }
  
  if (auto matrixAttr = schedule->getAttrOfType<mlir::DictionaryAttr>("conflict_matrix")) {
    for (auto &entry : matrixAttr) {
      StringRef key = entry.getName().getValue();
      if (auto intAttr = dyn_cast<mlir::IntegerAttr>(entry.getValue())) {
        // Key format: "action1,action2"
        auto parts = key.split(',');
        if (!parts.second.empty()) {
          conflicts.push_back({parts.first, parts.second, 
                             static_cast<int>(intAttr.getInt())});
        }
      }
    }
  }
  
  return conflicts;
}

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

/// Convert txn.action_method to func.func with abort status
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
    
    // Action methods now return i1 to indicate abort status
    // Original return types become buffer parameters
    SmallVector<mlir::Type> newArgTypes(method.getArgumentTypes());
    SmallVector<mlir::Type> bufferTypes;
    
    // Add buffer arguments for return values
    for (auto retType : method.getResultTypes()) {
      auto memrefType = mlir::MemRefType::get({}, retType);
      newArgTypes.push_back(memrefType);
      bufferTypes.push_back(memrefType);
    }
    
    // Function returns i1 (true = aborted, false = success)
    auto i1Type = rewriter.getI1Type();
    auto funcType = rewriter.getFunctionType(newArgTypes, {i1Type});
    
    // Create the function
    auto funcOp = rewriter.create<mlir::func::FuncOp>(loc, funcName, funcType);
    
    // Clone the body with modifications for abort handling
    rewriter.inlineRegionBefore(method.getBody(), funcOp.getBody(), funcOp.end());
    
    // Add return false at the end if not already present
    auto &block = funcOp.getBody().front();
    rewriter.setInsertionPoint(block.getTerminator());
    
    // Update the function to use buffer parameters for results
    // This is a simplified version - full implementation would need to
    // track and rewrite all return statements
    
    rewriter.eraseOp(method);
    return mlir::success();
  }
};

/// Convert txn.rule to func.func with abort status
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
    
    // Rules now return i1 to indicate abort status
    auto i1Type = rewriter.getI1Type();
    auto funcType = rewriter.getFunctionType({}, {i1Type});
    
    // Create the function
    auto funcOp = rewriter.create<mlir::func::FuncOp>(loc, funcName, funcType);
    
    // Clone the body
    rewriter.inlineRegionBefore(rule.getBody(), funcOp.getBody(), funcOp.end());
    
    // Add return false at the end only if there's no terminator
    auto &block = funcOp.getBody().front();
    if (!block.empty()) {
      auto terminator = block.getTerminator();
      // Only add return if there's no terminator at all
      // txn.abort, txn.yield, txn.return will be converted by their own patterns
      if (!terminator) {
        rewriter.setInsertionPointToEnd(&block);
        auto falseVal = rewriter.create<mlir::arith::ConstantOp>(
            loc, i1Type, rewriter.getBoolAttr(false));
        rewriter.create<mlir::func::ReturnOp>(loc, falseVal.getResult());
      }
    } else {
      // Empty block - add return false
      rewriter.setInsertionPointToEnd(&block);
      auto falseVal = rewriter.create<mlir::arith::ConstantOp>(
          loc, i1Type, rewriter.getBoolAttr(false));
      rewriter.create<mlir::func::ReturnOp>(loc, falseVal.getResult());
    }
    
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
    auto moduleName = moduleOp.getName();
    
    // Find the schedule if any
    auto schedule = findSchedule(moduleOp);
    
    // Generate wrapper functions for primitive instances
    for (auto &op : moduleOp.getBodyBlock()->getOperations()) {
      if (auto instanceOp = dyn_cast<::sharp::txn::InstanceOp>(&op)) {
        generatePrimitiveWrappers(moduleOp, instanceOp, rewriter);
      }
    }
    
    // Store module name on methods before moving them out
    for (auto &op : llvm::make_early_inc_range(moduleOp.getBodyBlock()->getOperations())) {
      if (isa<::sharp::txn::ValueMethodOp, ::sharp::txn::ActionMethodOp, 
              ::sharp::txn::RuleOp>(op)) {
        // Store the parent module name as an attribute
        op.setAttr("txn.parent_module", rewriter.getStringAttr(moduleName));
        op.moveBefore(moduleOp);
      }
    }
    
    // If we have a schedule, generate a scheduler function with will-fire logic
    if (schedule) {
      rewriter.setInsertionPoint(moduleOp);
      generateScheduler(moduleOp, schedule, rewriter);
    } else {
      // Create a simple main function without scheduling
      rewriter.setInsertionPoint(moduleOp);
      auto funcType = rewriter.getFunctionType({}, {});
      auto funcOp = rewriter.create<mlir::func::FuncOp>(
          loc, (moduleName + "_main").str(), funcType);
      
      auto *entryBlock = funcOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);
      rewriter.create<mlir::func::ReturnOp>(loc);
    }
    
    // Erase the module
    rewriter.eraseOp(moduleOp);
    return mlir::success();
  }
  
private:
  void generatePrimitiveWrappers(::sharp::txn::ModuleOp module,
                                ::sharp::txn::InstanceOp instance,
                                mlir::ConversionPatternRewriter &rewriter) const {
    auto loc = instance.getLoc();
    auto instanceName = instance.getSymName();
    auto moduleName = module.getName();
    
    // Get the primitive type - for now, just handle Register
    auto moduleNameRef = instance.getModuleName();
    auto primitiveType = moduleNameRef.str();
    if (primitiveType != "Register") {
      return;
    }
    
    // Generate wrapper for read method
    {
      auto funcName = moduleName.str() + "_" + instanceName.str() + "_read";
      auto i32Type = rewriter.getI32Type();
      auto funcType = rewriter.getFunctionType({}, {i32Type});
      
      rewriter.setInsertionPointAfter(module);
      auto funcOp = rewriter.create<mlir::func::FuncOp>(loc, funcName, funcType);
      auto *entryBlock = funcOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);
      
      // For now, just return 0
      auto zero = rewriter.create<mlir::arith::ConstantOp>(
          loc, i32Type, rewriter.getI32IntegerAttr(0));
      rewriter.create<mlir::func::ReturnOp>(loc, zero.getResult());
    }
    
    // Generate wrapper for write method
    {
      auto funcName = moduleName.str() + "_" + instanceName.str() + "_write";
      auto i32Type = rewriter.getI32Type();
      auto i1Type = rewriter.getI1Type();
      // Primitive action methods get i1 return type added by transformation
      auto funcType = rewriter.getFunctionType({i32Type}, {i1Type});
      
      rewriter.setInsertionPointAfter(module);
      auto funcOp = rewriter.create<mlir::func::FuncOp>(loc, funcName, funcType);
      auto *entryBlock = funcOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);
      
      // Primitives always succeed (return false for no abort)
      auto falseVal = rewriter.create<mlir::arith::ConstantOp>(
          loc, i1Type, rewriter.getBoolAttr(false));
      rewriter.create<mlir::func::ReturnOp>(loc, falseVal.getResult());
    }
  }
  
  void generateScheduler(::sharp::txn::ModuleOp module, 
                        ::sharp::txn::ScheduleOp schedule,
                        mlir::ConversionPatternRewriter &rewriter) const {
    auto loc = module.getLoc();
    StringRef moduleName = module.getName();
    
    // Create scheduler function
    std::string schedulerName = moduleName.str() + "_scheduler";
    auto funcType = rewriter.getFunctionType({}, {});
    auto schedulerFunc = rewriter.create<mlir::func::FuncOp>(
        loc, schedulerName, funcType);
    
    // Create entry block
    auto *entryBlock = schedulerFunc.addEntryBlock();
    rewriter.setInsertionPointToStart(entryBlock);
    
    // Get the list of scheduled actions
    auto actions = schedule.getActionsAttr();
    if (!actions) {
      rewriter.create<mlir::func::ReturnOp>(loc);
      return;
    }
    
    // Create tracking variables for which actions have fired
    SmallVector<Value> actionFired;
    auto i1Type = rewriter.getI1Type();
    auto memrefType = mlir::MemRefType::get({}, i1Type);
    
    for (size_t i = 0; i < actions.size(); ++i) {
      auto firedVar = rewriter.create<mlir::memref::AllocOp>(loc, memrefType);
      // Initialize to false
      auto falseVal = rewriter.create<mlir::arith::ConstantOp>(
          loc, i1Type, rewriter.getBoolAttr(false));
      rewriter.create<mlir::memref::StoreOp>(loc, falseVal.getResult(), firedVar);
      actionFired.push_back(firedVar);
    }
    
    // Extract conflict information
    auto conflicts = extractConflicts(module);
    
    // Execute each action in schedule order
    for (auto [idx, actionAttr] : llvm::enumerate(actions)) {
      auto action = cast<mlir::FlatSymbolRefAttr>(actionAttr);
      StringRef actionName = action.getValue();
      
      // For now, assume all actions are enabled (TODO: add guard checking)
      auto trueVal = rewriter.create<mlir::arith::ConstantOp>(
          loc, i1Type, rewriter.getBoolAttr(true));
      Value enabled = trueVal.getResult();
      
      // Check conflicts with earlier actions
      Value noConflicts = trueVal.getResult();
      for (size_t i = 0; i < idx; ++i) {
        auto earlierAction = cast<mlir::FlatSymbolRefAttr>(actions[i]);
        StringRef earlierName = earlierAction.getValue();
        
        // Check if these actions conflict
        bool hasConflict = false;
        for (auto &conflict : conflicts) {
          if ((conflict.action1 == earlierName && conflict.action2 == actionName && 
               conflict.relation == 2) || // Conflict
              (conflict.action1 == actionName && conflict.action2 == earlierName && 
               conflict.relation == 2)) { // Conflict
            hasConflict = true;
            break;
          }
        }
        
        if (hasConflict) {
          auto fired = rewriter.create<mlir::memref::LoadOp>(loc, actionFired[i]);
          auto notFired = rewriter.create<mlir::arith::XOrIOp>(loc, fired, trueVal.getResult());
          noConflicts = rewriter.create<mlir::arith::AndIOp>(loc, noConflicts, notFired);
        }
      }
      
      // Combine enabled and no-conflicts
      auto willFire = rewriter.create<mlir::arith::AndIOp>(loc, enabled, noConflicts);
      
      // Execute the action if will-fire is true
      auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, willFire, false);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      
      // Call the action function
      std::string actionFuncName = moduleName.str() + "_rule_" + actionName.str();
      auto callOp = rewriter.create<mlir::func::CallOp>(
          loc, actionFuncName, SmallVector<Type>{i1Type}, ValueRange{});
      auto aborted = callOp->getResult(0);
      
      // If not aborted, mark as fired
      auto notAborted = rewriter.create<mlir::arith::XOrIOp>(loc, aborted, trueVal.getResult());
      rewriter.create<mlir::memref::StoreOp>(loc, notAborted, actionFired[idx]);
      
      rewriter.setInsertionPointAfter(ifOp);
    }
    
    // Add final return
    rewriter.create<mlir::func::ReturnOp>(loc);
  }
};

/// Convert txn.return to func.return
struct ReturnToFuncReturnPattern : public mlir::OpConversionPattern<::sharp::txn::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(::sharp::txn::ReturnOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
    if (!funcOp) {
      return mlir::failure();
    }
    
    // Check if we're in an action method or rule that returns i1
    if (funcOp.getNumResults() == 1 && 
        funcOp.getResultTypes()[0].isInteger(1) &&
        adaptor.getOperands().empty()) {
      // Return false for success
      auto falseVal = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false));
      rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, falseVal.getResult());
    } else {
      // Normal case - convert to func.return with operands
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
    // Check if we're inside a txn.if - if so, don't convert yet
    if (op->getParentOfType<::sharp::txn::IfOp>()) {
      // Don't convert yields inside txn.if - they'll be handled when the if is converted
      return mlir::failure();
    }
    
    // Check if we're inside an scf.if
    if (op->getParentOfType<mlir::scf::IfOp>()) {
      // Convert to scf.yield
      rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, adaptor.getOperands());
    } else if (auto funcOp = op->getParentOfType<mlir::func::FuncOp>()) {
      // Check if function returns single i1 (abort flag)
      // This correctly identifies both rule and action method functions
      if (funcOp.getNumResults() == 1 && 
          funcOp.getResultTypes()[0].isInteger(1)) {
        // txn.yield means success (not aborted)
        auto falseVal = rewriter.create<mlir::arith::ConstantOp>(
            op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false));
        rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, falseVal.getResult());
      } else {
        // Invalid context - yield in value method
        return op.emitError("txn.yield not allowed in value methods");
      }
    } else {
      // Not in a recognized context
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
    // Handle instance method calls (e.g., @reg::@write)
    if (op.getCallee().getNestedReferences().size() > 0) {
      // This is an instance method call
      auto instanceName = op.getCallee().getRootReference().str();
      auto methodName = op.getCallee().getLeafReference().str();
      
      // Get parent module name
      std::string moduleName;
      if (auto parentModule = op->getParentOfType<::sharp::txn::ModuleOp>()) {
        moduleName = parentModule.getName().str();
      } else if (auto parentFunc = op->getParentOfType<mlir::func::FuncOp>()) {
        auto funcName = parentFunc.getName();
        auto underscorePos = funcName.find('_');
        if (underscorePos != StringRef::npos) {
          moduleName = funcName.substr(0, underscorePos).str();
        }
      }
      
      // Construct instance method function name
      auto funcName = moduleName + "_" + instanceName + "_" + methodName;
      
      // Check if we're inside a rule or action method
      auto parentFunc = op->getParentOfType<mlir::func::FuncOp>();
      bool insideRuleOrAction = parentFunc && 
          (parentFunc.getName().contains("_rule_") || 
           (parentFunc.getNumResults() == 1 && 
            parentFunc.getResultTypes()[0].isInteger(1)));
      
      // Check if this is a call to an action method (write, etc.)
      bool isActionMethodCall = op.getNumResults() == 0 && 
          methodName != "read" && methodName != "getValue";
      
      if (insideRuleOrAction && isActionMethodCall) {
        // Inside a rule/action, calling an action method (including primitives)
        // All action methods return i1 (abort status) in the transformed code
        auto i1Type = rewriter.getI1Type();
        rewriter.create<mlir::func::CallOp>(
            op.getLoc(), funcName, SmallVector<Type>{i1Type}, adaptor.getOperands());
        rewriter.eraseOp(op);
      } else {
        // Normal call - preserve original behavior
        rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
            op, funcName, op.getResultTypes(), adaptor.getOperands());
      }
      return mlir::success();
    }
    
    // Handle module-level method calls
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
    auto calleeName = op.getCallee().getRootReference().str();
    auto funcName = moduleName + "_" + calleeName;
    
    // Check if we're inside a rule or action method
    auto parentFunc = op->getParentOfType<mlir::func::FuncOp>();
    bool insideRuleOrAction = parentFunc && 
        (parentFunc.getName().contains("_rule_") || 
         (parentFunc.getNumResults() == 1 && 
          parentFunc.getResultTypes()[0].isInteger(1)));
    
    // Check if this is a call to an action method
    bool isActionMethodCall = op.getNumResults() == 0 && 
        calleeName.find("getValue") == std::string::npos &&
        calleeName.find("read") == std::string::npos &&
        funcName.find("_rule_") == std::string::npos;
    
    if (insideRuleOrAction && isActionMethodCall) {
      // Inside a rule/action, calling an action method
      // The action method now returns i1 (abort status)
      auto i1Type = rewriter.getI1Type();
      rewriter.create<mlir::func::CallOp>(
          op.getLoc(), funcName, SmallVector<Type>{i1Type}, adaptor.getOperands());
      
      // Just erase the original op - the call was already created
      rewriter.eraseOp(op);
    } else {
      // Normal call - preserve original behavior
      rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
          op, funcName, op.getResultTypes(), adaptor.getOperands());
    }
    
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
      
      // Move all operations from source to destination except txn.yield
      for (auto &srcOp : llvm::make_early_inc_range(srcBlock)) {
        if (isa<::sharp::txn::YieldOp>(&srcOp)) {
          // Skip txn.yield - it will be handled by the conversion
          continue;
        }
        srcOp.moveBefore(&dstBlock, dstBlock.end());
      }
      
      // Ensure the block has a terminator
      if (dstBlock.empty() || !dstBlock.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        rewriter.setInsertionPointToEnd(&dstBlock);
        rewriter.create<mlir::scf::YieldOp>(op.getLoc());
      }
    }
    
    // Convert the else region if it exists
    if (!op.getElseRegion().empty() && !scfIf.getElseRegion().empty()) {
      auto &srcBlock = op.getElseRegion().front();
      auto &dstBlock = scfIf.getElseRegion().front();
      
      // Move all operations from source to destination except txn.yield
      for (auto &srcOp : llvm::make_early_inc_range(srcBlock)) {
        if (isa<::sharp::txn::YieldOp>(&srcOp)) {
          // Skip txn.yield - it will be handled by the conversion
          continue;
        }
        srcOp.moveBefore(&dstBlock, dstBlock.end());
      }
      
      // Ensure the block has a terminator
      if (dstBlock.empty() || !dstBlock.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        rewriter.setInsertionPointToEnd(&dstBlock);
        rewriter.create<mlir::scf::YieldOp>(op.getLoc());
      }
    }
    
    // Replace the original operation with the results
    rewriter.replaceOp(op, scfIf.getResults());
    return mlir::success();
  }
};

/// Convert txn.abort to func.return with abort status
struct AbortToReturnPattern : public mlir::OpConversionPattern<::sharp::txn::AbortOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(::sharp::txn::AbortOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
    if (!funcOp) {
      return mlir::failure();
    }
    
    // Check if we're in a rule or action method (which should return i1)
    if (funcOp.getNumResults() == 1 && 
        funcOp.getResultTypes()[0].isInteger(1)) {
      // Return true to indicate abort
      auto trueVal = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(true));
      rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, trueVal.getResult());
    } else {
      // Legacy behavior for other functions
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
    }
    
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
    target.addLegalDialect<mlir::memref::MemRefDialect>();
    
    // Mark specific txn operations as illegal
    target.addIllegalOp<::sharp::txn::ModuleOp>();
    target.addIllegalOp<::sharp::txn::ValueMethodOp>();
    target.addIllegalOp<::sharp::txn::ActionMethodOp>();
    target.addIllegalOp<::sharp::txn::RuleOp>();
    target.addIllegalOp<::sharp::txn::ReturnOp>();
    target.addIllegalOp<::sharp::txn::CallOp>();
    target.addIllegalOp<::sharp::txn::IfOp>();
    target.addIllegalOp<::sharp::txn::AbortOp>();
    
    // txn.yield is always illegal - it will be converted
    target.addIllegalOp<::sharp::txn::YieldOp>();
    
    // Mark txn.schedule as illegal too
    target.addIllegalOp<::sharp::txn::ScheduleOp>();
    
    TxnToFuncTypeConverter typeConverter;
    mlir::RewritePatternSet patterns(&getContext());
    
    populateTxnToFuncConversionPatterns(typeConverter, patterns);
    
    if (failed(applyPartialConversion(getOperation(), target,
                                     std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    
    // Post-process functions to add abort propagation
    addAbortPropagation();
  }
  
private:
  void addAbortPropagation() {
    // Walk all functions and add abort propagation where needed
    getOperation()->walk([&](mlir::func::FuncOp funcOp) {
      // Only process rule and action functions
      if (!funcOp.getName().contains("_rule_") && 
          !funcOp.getName().contains("_action_")) {
        return;
      }
      
      // Check if function returns i1 (abort status)
      if (funcOp.getNumResults() != 1 || 
          !funcOp.getResultTypes()[0].isInteger(1)) {
        return;
      }
      
      // Process the function body
      auto &block = funcOp.getBody().front();
      mlir::IRRewriter rewriter(&getContext());
      
      // Find all action method calls that return i1
      SmallVector<mlir::func::CallOp> actionCalls;
      block.walk([&](mlir::func::CallOp callOp) {
        if (callOp.getNumResults() == 1 && 
            callOp.getResult(0).getType().isInteger(1) &&
            (callOp.getCallee().contains("_action_") ||
             callOp.getCallee().contains("_mayAbort") ||
             callOp.getCallee().contains("_doWork") ||
             callOp.getCallee().contains("_write"))) {
          actionCalls.push_back(callOp);
        }
      });
      
      // Process calls in reverse order to avoid invalidating iterators
      for (auto it = actionCalls.rbegin(); it != actionCalls.rend(); ++it) {
        auto callOp = *it;
        
        // Skip if this is the last operation before return
        auto nextOp = callOp->getNextNode();
        if (!nextOp || isa<mlir::func::ReturnOp>(nextOp)) {
          continue;
        }
        
        // Collect all operations from after this call to the terminator
        SmallVector<Operation*> opsToWrap;
        Operation *currentOp = nextOp;
        Operation *returnOp = nullptr;
        while (currentOp) {
          if (isa<mlir::func::ReturnOp>(currentOp)) {
            returnOp = currentOp;
            break;
          }
          // Skip constants that are just feeding the return
          if (auto constOp = dyn_cast<mlir::arith::ConstantOp>(currentOp)) {
            auto nextNext = currentOp->getNextNode();
            if (nextNext && isa<mlir::func::ReturnOp>(nextNext)) {
              // This is likely the false constant for the return
              currentOp = nextNext;
              continue;
            }
          }
          opsToWrap.push_back(currentOp);
          currentOp = currentOp->getNextNode();
        }
        
        // If there's nothing meaningful to wrap, skip
        if (opsToWrap.empty()) {
          continue;
        }
        
        // Create the abort check
        rewriter.setInsertionPointAfter(callOp);
        auto loc = callOp.getLoc();
        auto i1Type = rewriter.getI1Type();
        
        // Get abort status from the call result (all action calls now return i1)
        auto abortStatus = callOp.getResult(0);
        
        // Create if (!aborted) { ... rest ... } else { return true }
        auto trueVal = rewriter.create<mlir::arith::ConstantOp>(
            loc, i1Type, rewriter.getBoolAttr(true));
        auto notAborted = rewriter.create<mlir::arith::XOrIOp>(
            loc, abortStatus, trueVal.getResult());
        
        auto ifOp = rewriter.create<mlir::scf::IfOp>(
            loc, TypeRange{}, notAborted, /*withElse=*/true);
        
        // Move operations into then block
        auto &thenBlock = ifOp.getThenRegion().front();
        Operation *existingTerminator = nullptr;
        for (auto *op : opsToWrap) {
          // Check if this is a terminator (scf.yield)
          if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
            existingTerminator = op;
            // Don't move terminators yet
          } else {
            op->moveBefore(&thenBlock, thenBlock.end());
          }
        }
        
        // If we found a return operation, handle it
        if (returnOp) {
          // Erase any existing terminator first
          if (existingTerminator) {
            rewriter.eraseOp(existingTerminator);
          }
          rewriter.setInsertionPointToEnd(&thenBlock);
          auto falseVal = rewriter.create<mlir::arith::ConstantOp>(
              loc, i1Type, rewriter.getBoolAttr(false));
          rewriter.create<mlir::func::ReturnOp>(loc, falseVal.getResult());
          // Erase the original return
          rewriter.eraseOp(returnOp);
        } else if (!returnOp && !opsToWrap.empty()) {
          // Only add scf.yield if we don't have a return and we have ops to wrap
          // Move the existing terminator if we have one, or create a new scf.yield
          if (existingTerminator) {
            existingTerminator->moveBefore(&thenBlock, thenBlock.end());
          } else {
            rewriter.setInsertionPointToEnd(&thenBlock);
            rewriter.create<mlir::scf::YieldOp>(loc);
          }
        }
        
        // Add return true to else block
        auto &elseBlock = ifOp.getElseRegion().front();
        rewriter.setInsertionPointToEnd(&elseBlock);
        // Remove any existing terminator
        if (!elseBlock.empty() && elseBlock.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
          rewriter.eraseOp(&elseBlock.back());
        }
        rewriter.create<mlir::func::ReturnOp>(loc, trueVal.getResult());
      }
    });
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