//===- LowerOpToFIRRTLPass.cpp - Lower Txn Bodies to FIRRTL --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the first pass of the two-phase Txn-to-FIRRTL conversion.
// It converts operations within the guards and bodies of Txn value methods,
// action methods, and rules to FIRRTL operations while preserving the Txn 
// module structure.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/BuiltinTypes.h"
#include "sharp/Conversion/Passes.h"
#include "sharp/Dialect/Txn/TxnDialect.h"
#include "sharp/Dialect/Txn/TxnOps.h"
#include "sharp/Dialect/Txn/TxnTypes.h"
#include "sharp/Analysis/AnalysisError.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lower-op-to-firrtl"

namespace mlir {
namespace sharp {

#define GEN_PASS_DEF_LOWEROPTOFIRRTLPASS
#include "sharp/Conversion/Passes.h.inc"

namespace {

// Use explicit namespaces to avoid ambiguity
using ::sharp::txn::ActionMethodOp;
using ::sharp::txn::ValueMethodOp;
using ::sharp::txn::CallOp;
using ::sharp::txn::ReturnOp;
using ::sharp::txn::IfOp;
using ::sharp::txn::PrimitiveOp;
// Explicitly qualify InstanceOp to avoid ambiguity with FIRRTL's InstanceOp
namespace txn = ::sharp::txn;

using namespace ::circt::firrtl;


//===----------------------------------------------------------------------===//
// TypeConverter for Txn to FIRRTL
//===----------------------------------------------------------------------===//

class TxnToFIRRTLTypeConverter : public mlir::TypeConverter {
public:
  TxnToFIRRTLTypeConverter() {
    // Convert integer types to FIRRTL types
    addConversion([](IntegerType type) -> std::optional<Type> {
      LLVM_DEBUG(llvm::dbgs() << "Running TxnToFIRRTLTypeConverter::conversion for type: " << type << "\n");
     
      if (type.isSigned()) {
        return SIntType::get(type.getContext(), type.getWidth());
      } else {
        return UIntType::get(type.getContext(), type.getWidth());
      }
    });

    // Firrtl types are legal
    addConversion([](FIRRTLType type) -> std::optional<Type> {
      return type;
    });
    
    // Add target materialization to handle conversions
    addTargetMaterialization([](OpBuilder &builder, Type resultType,
                               ValueRange inputs, Location loc) -> Value {

      LLVM_DEBUG(llvm::dbgs() << "Running TxnToFIRRTLTypeConverter::target materialization\n");
      // debug, print inputs and resultType
      llvm::errs() << "inputs: ";
      for (auto input : inputs) {
        llvm::errs() << input.getType() << " ";
      }
      llvm::errs() << " =>  resultType: " << resultType << "\n";

      if (inputs.size() != 1)
        return nullptr;
      
      // If types match, return the input directly
      if (inputs[0].getType() == resultType)
        return inputs[0];

      // If the firrtl type has unmatched width, use bits for truncation or pad for extension
      if (auto result_ty = dyn_cast<UIntType>(resultType)) {
        if (auto input_ty = dyn_cast<UIntType>(inputs[0].getType())) {
          auto result_width = result_ty.getWidth();
          auto input_width = input_ty.getWidth();
          if (result_width.value() > input_width.value()) {
            return builder.create<PadPrimOp>(loc, result_ty, inputs[0]);
          } else if (result_width.value() < input_width.value()) {
            return builder.create<BitsPrimOp>(loc, result_ty, inputs[0], result_width.value() - 1, 0);
          }
        } else if (auto input_ty = dyn_cast<SIntType>(inputs[0].getType())) {
          // use bitcast to convert to UIntType
          return builder.create<BitCastOp>(loc, result_ty, inputs[0]);
        } 
      } else if (auto result_ty = dyn_cast<SIntType>(resultType)) {
        if (auto input_ty = dyn_cast<SIntType>(inputs[0].getType())) {
          auto result_width = result_ty.getWidth();
          auto input_width = input_ty.getWidth();
          if (result_width.value() > input_width.value()) {
            return builder.create<PadPrimOp>(loc, result_ty, inputs[0]);
          } else if (result_width.value() < input_width.value()) {
            return builder.create<BitsPrimOp>(loc, result_ty, inputs[0], result_width.value() - 1, 0);
          }
        } else if (auto input_ty = dyn_cast<UIntType>(inputs[0].getType())) {
          // use bitcast to convert to SIntType
          return builder.create<BitCastOp>(loc, result_ty, inputs[0]);
        }
      }
      
      // Otherwise, create an unrealized conversion cast
      auto cast = builder.create<mlir::UnrealizedConversionCastOp>(
          loc, resultType, inputs[0]);
      return cast.getResult(0);
    });
    
    // Add source materialization
    addSourceMaterialization([](OpBuilder &builder, Type resultType,
                               ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1)
        return nullptr;
      
      // If types match, return the input directly
      if (inputs[0].getType() == resultType)
        return inputs[0];
      
      // Otherwise, create an unrealized conversion cast
      auto cast = builder.create<mlir::UnrealizedConversionCastOp>(
          loc, resultType, inputs[0]);
      return cast.getResult(0);
    });
  }
};


//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Convert arith.constant operations to firrtl.constant
class ArithConstantToFIRRTLPattern : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {

    LLVM_DEBUG(llvm::dbgs() << "ArithConstantToFIRRTLPattern: converting constant: " << op << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Original type: " << op.getType() << "\n");
    auto firrtlType = getTypeConverter()->convertType(op.getType());
    if (!firrtlType)
      return failure();
    
    auto intAttr = dyn_cast<IntegerAttr>(adaptor.getValue());
    if (!intAttr)
      return failure();
    
    bool isUnsigned = isa<UIntType>(firrtlType);
    llvm::APSInt value(intAttr.getValue(), isUnsigned);
    
    // Create FIRRTL constant with the converted type
    auto firrtlConstant = rewriter.create<ConstantOp>(
        op.getLoc(), firrtlType, value);
    
    rewriter.replaceOp(op, firrtlConstant.getResult());
    return success();
  }
};

/// Convert arith binary operations to FIRRTL primitives
template<typename ArithOp, typename FIRRTLOp>
class ArithBinaryToFIRRTLPattern : public OpConversionPattern<ArithOp> {
public:
  using OpConversionPattern<ArithOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(ArithOp op,  ArithBinaryToFIRRTLPattern::OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    // Convert operands to FIRRTL types
    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];
    
    // For FIRRTL operations, let them infer their own result types
    // Don't force the result type - let FIRRTL type inference handle it
    auto firrtlOp = rewriter.create<FIRRTLOp>(op.getLoc(), lhs, rhs);
    
    rewriter.replaceOp(op, firrtlOp.getResult());
    return success();
  }
};

/// Convert arith.cmpi operations to FIRRTL comparison primitives
class ArithCmpIToFIRRTLPattern : public OpConversionPattern<arith::CmpIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    // Convert operands to FIRRTL types
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    
    // Map arith comparison predicates to FIRRTL operations
    Value firrtlResult;
    switch (op.getPredicate()) {
    case arith::CmpIPredicate::eq:
      firrtlResult = rewriter.create<EQPrimOp>(op.getLoc(), lhs, rhs).getResult();
      break;
    case arith::CmpIPredicate::ne:
      firrtlResult = rewriter.create<NEQPrimOp>(op.getLoc(), lhs, rhs).getResult();
      break;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      firrtlResult = rewriter.create<LTPrimOp>(op.getLoc(), lhs, rhs).getResult();
      break;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      firrtlResult = rewriter.create<LEQPrimOp>(op.getLoc(), lhs, rhs).getResult();
      break;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      firrtlResult = rewriter.create<GTPrimOp>(op.getLoc(), lhs, rhs).getResult();
      break;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      firrtlResult = rewriter.create<GEQPrimOp>(op.getLoc(), lhs, rhs).getResult();
      break;
    }
    
    // Replace with the FIRRTL result - type materialization will handle conversion if needed
    rewriter.replaceOp(op, firrtlResult);
    return success();
  }
};



//===----------------------------------------------------------------------===//
// Conversion Patterns for Txn Operations
//===----------------------------------------------------------------------===//

/// Convert txn.action_method operation
class ActionMethodOpConversion : public OpConversionPattern<ActionMethodOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(ActionMethodOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    // Get the current function type and convert it
    auto funcType = op.getFunctionType();
    SmallVector<Type> convertedInputs;
    SmallVector<Type> convertedResults;
    
    if (failed(getTypeConverter()->convertTypes(funcType.getInputs(), convertedInputs)))
      return failure();
    if (failed(getTypeConverter()->convertTypes(funcType.getResults(), convertedResults)))
      return failure();
    
    auto newFuncType = FunctionType::get(op.getContext(), convertedInputs, convertedResults);


    // Clone the operation without regions
    auto newOp = rewriter.cloneWithoutRegions(op);
    
    // Update the function type attribute
    newOp->setAttr("function_type", TypeAttr::get(newFuncType));
    
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                               newOp.getBody().end());
    
    // Convert the region types (block arguments)
    if (failed(rewriter.convertRegionTypes(&newOp.getBody(), *getTypeConverter()))) {
      LLVM_DEBUG(llvm::dbgs() << "ActionMethodOpConversion: failed to convert region types\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "ActionMethodOpConversion: converted " << newOp << "\n");
    
      
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

/// Convert txn.value_method operation
class ValueMethodOpConversion : public OpConversionPattern<ValueMethodOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(ValueMethodOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    // Get the current function type and convert it
    auto funcType = op.getFunctionType();
    SmallVector<Type> convertedInputs;
    SmallVector<Type> convertedResults;
    
    if (failed(getTypeConverter()->convertTypes(funcType.getInputs(), convertedInputs)))
      return failure();
    if (failed(getTypeConverter()->convertTypes(funcType.getResults(), convertedResults)))
      return failure();
    
    auto newFuncType = FunctionType::get(op.getContext(), convertedInputs, convertedResults);
    
    // Clone the operation without regions
    auto newOp = rewriter.cloneWithoutRegions(op);
    
    // Update the function type attribute
    newOp->setAttr("function_type", TypeAttr::get(newFuncType));
    
    // Inline the body region
    rewriter.inlineRegionBefore(op.getBody(), cast<ValueMethodOp>(newOp).getBody(),
                               cast<ValueMethodOp>(newOp).getBody().end());
    
    // Convert the region types (block arguments)
    if (failed(rewriter.convertRegionTypes(&cast<ValueMethodOp>(newOp).getBody(), *getTypeConverter()))) {
      LLVM_DEBUG(llvm::dbgs() << "ValueMethodOpConversion: failed to convert region types\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "ValueMethodOpConversion: converted " << newOp << "\n");
      
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

/// Convert txn.instance operation
class InstanceOpConversion : public OpConversionPattern<txn::InstanceOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(txn::InstanceOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    // Convert type arguments if present
    SmallVector<Attribute> convertedTypeAttrs;
    if (auto typeArgs = op.getTypeArguments()) {
      for (auto typeAttr : typeArgs.value()) {
        if (auto type = dyn_cast<TypeAttr>(typeAttr)) {
          auto convertedType = getTypeConverter()->convertType(type.getValue());
          if (!convertedType)
            return failure();
          convertedTypeAttrs.push_back(TypeAttr::get(convertedType));
        } else {
          convertedTypeAttrs.push_back(typeAttr);
        }
      }
    }
    
    // Create new instance with converted type arguments
    ArrayAttr typeArgs = convertedTypeAttrs.empty() ? nullptr : ArrayAttr::get(op.getContext(), convertedTypeAttrs);
  
    // replace
    auto newOp = rewriter.create<txn::InstanceOp>(op.getLoc(), op.getSymName(), op.getModuleName(), typeArgs, op.getConstArgumentsAttr());
    rewriter.replaceOp(op, newOp);
    LLVM_DEBUG(llvm::dbgs() << "InstanceOpConversion: replaced with " << newOp << "\n");

    // // In-place replacement
    // rewriter.startOpModification(op);
    // op.setTypeArgumentsAttr(typeArgs);
    // rewriter.finalizeOpModification(op);
    // LLVM_DEBUG(llvm::dbgs() << "InstanceOpConversion: in-place modified into" << op << "\n");

    return success();
  }
};

/// Convert txn.primitive operation
class PrimitiveOpConversion : public OpConversionPattern<PrimitiveOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(PrimitiveOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    // Convert type parameters if present
    SmallVector<Attribute> convertedTypeAttrs;
    if (auto typeParams = op.getTypeParameters()) {
      for (auto typeAttr : typeParams.value()) {
        if (auto type = dyn_cast<TypeAttr>(typeAttr)) {
          auto convertedType = getTypeConverter()->convertType(type.getValue());
          if (!convertedType)
            return failure();
          convertedTypeAttrs.push_back(TypeAttr::get(convertedType));
        } else {
          convertedTypeAttrs.push_back(typeAttr);
        }
      }
    }
    
    // Clone the primitive with converted type parameters
    auto newOp = rewriter.cloneWithoutRegions(op);
    auto newPrimOp = cast<PrimitiveOp>(newOp);
    
    if (!convertedTypeAttrs.empty()) {
      newOp->setAttr("type_parameters", ArrayAttr::get(op.getContext(), convertedTypeAttrs));
    }
    
    // Inline the body region
    rewriter.inlineRegionBefore(op.getBody(), newPrimOp.getBody(), newPrimOp.getBody().end());
    
    // Convert the body region types (method definitions inside primitive)
    // Iterate through operations in the body and convert method signatures
    for (auto &nestedOp : newPrimOp.getBody().front()) {
      if (auto methodOp = dyn_cast<::sharp::txn::FirValueMethodOp>(&nestedOp)) {
        // Get the function type from the attribute
        auto funcTypeAttr = methodOp.getFunctionTypeAttr();
        auto funcType = cast<FunctionType>(funcTypeAttr.getValue());
        
        // Convert method argument and result types
        SmallVector<Type> convertedArgTypes;
        for (Type type : funcType.getInputs()) {
          if (auto convertedType = getTypeConverter()->convertType(type))
            convertedArgTypes.push_back(convertedType);
          else
            return failure();
        }
        
        SmallVector<Type> convertedResultTypes;
        for (Type type : funcType.getResults()) {
          if (auto convertedType = getTypeConverter()->convertType(type))
            convertedResultTypes.push_back(convertedType);
          else
            return failure();
        }
        
        // Update the method signature
        auto newFuncType = FunctionType::get(op.getContext(), convertedArgTypes, convertedResultTypes);
        methodOp->setAttr("function_type", TypeAttr::get(newFuncType));
      } else if (auto methodOp = dyn_cast<::sharp::txn::FirActionMethodOp>(&nestedOp)) {
        // Similar conversion for action methods
        auto funcTypeAttr = methodOp.getFunctionTypeAttr();
        auto funcType = cast<FunctionType>(funcTypeAttr.getValue());
        
        SmallVector<Type> convertedArgTypes;
        for (Type type : funcType.getInputs()) {
          if (auto convertedType = getTypeConverter()->convertType(type))
            convertedArgTypes.push_back(convertedType);
          else
            return failure();
        }
        
        SmallVector<Type> convertedResultTypes;
        for (Type type : funcType.getResults()) {
          if (auto convertedType = getTypeConverter()->convertType(type))
            convertedResultTypes.push_back(convertedType);
          else
            return failure();
        }
        
        // Update the method signature
        auto newFuncType = FunctionType::get(op.getContext(), convertedArgTypes, convertedResultTypes);
        methodOp->setAttr("function_type", TypeAttr::get(newFuncType));
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "PrimitiveOpConversion: converted " << newOp << "\n");

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

/// Convert txn.call operation
class CallOpConversion : public OpConversionPattern<CallOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(CallOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
                                
    LLVM_DEBUG(llvm::dbgs() << "CallOpConversion: converting call of operand types " << adaptor.getOperands().getTypes() << ", original result types " << op.getResultTypes() << "\n");

    // Convert result types
    SmallVector<Type> convertedResults;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), convertedResults)))
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "CallOpConversion: converted result types " << convertedResults << "\n");
    
    // Create new call with converted operands and result types
    auto newOp = rewriter.create<CallOp>(
        op.getLoc(), op.getCalleeAttr(), adaptor.getOperands(), convertedResults);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

/// Convert txn.return operation
class ReturnOpConversion : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    // Create new return with converted operands
    rewriter.replaceOpWithNewOp<ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

/// Convert txn.abort operation
class AbortOpConversion : public OpConversionPattern<txn::AbortOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(txn::AbortOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    // Create new abort with converted operands
    auto emptyTypeRange = TypeRange();
    rewriter.replaceOpWithNewOp<txn::AbortOp>(op, emptyTypeRange, adaptor.getOperands());
    return success();
  }
};

/// Convert txn.if operation
class IfOpConversion : public OpConversionPattern<IfOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(IfOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    // Convert result types
    SmallVector<Type> convertedResults;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), convertedResults)))
      return failure();
    
    // Create new if operation
    auto newIf = rewriter.create<IfOp>(op.getLoc(), convertedResults, adaptor.getCondition());
    
    // Move regions
    rewriter.inlineRegionBefore(op.getThenRegion(), newIf.getThenRegion(), newIf.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), newIf.getElseRegion(), newIf.getElseRegion().end());
    
    // Convert region types
    if (failed(rewriter.convertRegionTypes(&newIf.getThenRegion(), *getTypeConverter()))) {
      LLVM_DEBUG(llvm::dbgs() << "IfOpConversion: failed to convert then region types\n");
      return failure();
    }
    if (failed(rewriter.convertRegionTypes(&newIf.getElseRegion(), *getTypeConverter()))) {
      LLVM_DEBUG(llvm::dbgs() << "IfOpConversion: failed to convert else region types\n");
      return failure();
    }
      
    rewriter.replaceOp(op, newIf.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

class LowerOpToFIRRTLPass : public impl::LowerOpToFIRRTLPassBase<LowerOpToFIRRTLPass> {
public:
  using impl::LowerOpToFIRRTLPassBase<LowerOpToFIRRTLPass>::LowerOpToFIRRTLPassBase;
  
  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "Running LowerOpToFIRRTLPass\n");
    
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    
    // Set up type converter
    TxnToFIRRTLTypeConverter typeConverter;
    
    // Set up conversion target
    ConversionTarget target(*ctx);
    
    // FIRRTL dialect is legal
    target.addLegalDialect<::circt::firrtl::FIRRTLDialect>();
    
    // Other Txn operations are legal by default
    target.addLegalDialect<::sharp::txn::TxnDialect>();
    
    // Arith operations are illegal
    target.addIllegalDialect<arith::ArithDialect>();
    
    // Txn operations are legal if their types are converted
    target.addDynamicallyLegalOp<ActionMethodOp>([&](ActionMethodOp op) {
      // Check if block arguments are converted
      for (auto arg : op.getFunctionType().getInputs()) {
        if (!typeConverter.isLegal(arg))
          return false;
      }
      // check if the return type is converted
      for (auto result : op.getFunctionType().getResults()) {
        if (!typeConverter.isLegal(result))
          return false;
      }
      return true;
    });
    
    target.addDynamicallyLegalOp<ValueMethodOp>([&](ValueMethodOp op) {
      // Check result types
      for (auto result : op.getFunctionType().getResults()) {
        if (!typeConverter.isLegal(result))
          return false;
      }
      // Check block arguments
      for (auto arg : op.getFunctionType().getInputs()) {
        if (!typeConverter.isLegal(arg))
          return false;
      }
      return true;
    });
    
    target.addDynamicallyLegalOp<txn::InstanceOp>([&](txn::InstanceOp op) {
      // Check type arguments
      if (auto typeArgs = op.getTypeArguments()) {
        for (auto typeAttr : typeArgs.value()) {
          if (auto type = dyn_cast<TypeAttr>(typeAttr)) {
            if (!typeConverter.isLegal(type.getValue()))
              return false;
          }
        }
      }
      return true;
    });
    
    target.addDynamicallyLegalOp<PrimitiveOp>([&](PrimitiveOp op) {
      // Check type parameters
      if (auto typeParams = op.getTypeParameters()) {
        for (auto typeAttr : typeParams.value()) {
          if (auto type = dyn_cast<TypeAttr>(typeAttr)) {
            if (!typeConverter.isLegal(type.getValue()))
              return false;
          }
        }
      }
      // Also check method signatures inside primitive
      for (auto &nestedOp : op.getBody().front()) {
        if (auto methodOp = dyn_cast<::sharp::txn::FirValueMethodOp>(&nestedOp)) {
          auto funcType = cast<FunctionType>(methodOp.getFunctionTypeAttr().getValue());
          for (Type type : funcType.getInputs()) {
            if (!typeConverter.isLegal(type))
              return false;
          }
          for (Type type : funcType.getResults()) {
            if (!typeConverter.isLegal(type))
              return false;
          }
        } else if (auto methodOp = dyn_cast<::sharp::txn::FirActionMethodOp>(&nestedOp)) {
          auto funcType = cast<FunctionType>(methodOp.getFunctionTypeAttr().getValue());
          for (Type type : funcType.getInputs()) {
            if (!typeConverter.isLegal(type))
              return false;
          }
          for (Type type : funcType.getResults()) {
            if (!typeConverter.isLegal(type))
              return false;
          }
        }
      }
      return true;
    });
    
    target.addDynamicallyLegalOp<txn::CallOp>([&](txn::CallOp op) {
      // Check operand and result types
      for (Value operand : op.getOperands()) {
        if (!typeConverter.isLegal(operand.getType()))
          return false;
      }
      for (Type type : op.getResultTypes()) {
        if (!typeConverter.isLegal(type))
          return false;
      }
      return true;
    });
    
    target.addDynamicallyLegalOp<ReturnOp>([&](ReturnOp op) {
      // Check operand types
      for (Value operand : op.getOperands()) {
        if (!typeConverter.isLegal(operand.getType()))
          return false;
      }
      return true;
    });

    target.addDynamicallyLegalOp<txn::AbortOp>([&](txn::AbortOp op) {
      // Check operand types
      for (Value operand : op.getOperands()) {
        if (!typeConverter.isLegal(operand.getType()))
          return false;
      }
      return true;
    });
    
    target.addDynamicallyLegalOp<IfOp>([&](IfOp op) {
      Type conditionType = op.getCondition().getType();
      if (!typeConverter.isLegal(conditionType))
        return false;
      
      // results must be converted
      for (Type type : op.getResultTypes()) {
        if (!typeConverter.isLegal(type))
          return false;
      }
      
      return true;
    });
    
    // Set up conversion patterns
    RewritePatternSet patterns(ctx);
    
    // Add Txn operation conversions
    patterns.add<ActionMethodOpConversion, ValueMethodOpConversion,
                 CallOpConversion, ReturnOpConversion, IfOpConversion, AbortOpConversion>(typeConverter, ctx);
    // Instance and Primitive need special handling due to namespace conflicts
    patterns.add<InstanceOpConversion>(typeConverter, ctx);
    patterns.add<PrimitiveOpConversion>(typeConverter, ctx);
    
    // Add arith operation conversions
    patterns.add<ArithConstantToFIRRTLPattern>(typeConverter, ctx);
    patterns.add<ArithBinaryToFIRRTLPattern<arith::AddIOp, AddPrimOp>>(typeConverter, ctx);
    patterns.add<ArithBinaryToFIRRTLPattern<arith::SubIOp, SubPrimOp>>(typeConverter, ctx);
    patterns.add<ArithBinaryToFIRRTLPattern<arith::MulIOp, MulPrimOp>>(typeConverter, ctx);
    
    // Add bitwise logical operations
    patterns.add<ArithBinaryToFIRRTLPattern<arith::XOrIOp, XorPrimOp>>(typeConverter, ctx);
    patterns.add<ArithBinaryToFIRRTLPattern<arith::AndIOp, AndPrimOp>>(typeConverter, ctx);
    patterns.add<ArithBinaryToFIRRTLPattern<arith::OrIOp, OrPrimOp>>(typeConverter, ctx);
    
    // Add comparison operations
    patterns.add<ArithCmpIToFIRRTLPattern>(typeConverter, ctx);
    
    // Apply partial conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "LowerOpToFIRRTLPass failed\n");
      signalPassFailure();
      return;
    }
    
    
    LLVM_DEBUG(llvm::dbgs() << "LowerOpToFIRRTLPass completed successfully\n");
  }
};

} // end anonymous namespace

// Pass creation function is generated by tablegen

} // end namespace sharp
} // end namespace mlir