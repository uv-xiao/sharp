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
// Type Conversion Utilities
//===----------------------------------------------------------------------===//

/// Convert Sharp types to FIRRTL types
static FIRRTLType convertType(Type type) {
  MLIRContext *ctx = type.getContext();
  
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.isSigned()) {
      return SIntType::get(ctx, intType.getWidth());
    } else {
      return UIntType::get(ctx, intType.getWidth());
    }
  }
  
  if (auto clockType = dyn_cast<::circt::firrtl::ClockType>(type)) {
    return clockType;
  }
  
  if (auto resetType = dyn_cast<::circt::firrtl::ResetType>(type)) {
    return resetType;
  }
  
  // For other types, return as-is (they might already be FIRRTL types)
  if (auto firrtlType = dyn_cast<FIRRTLType>(type)) {
    return firrtlType;
  }
  
  // Default to UInt<1> for unknown types
  return UIntType::get(ctx, 1);
}

/// Convert a value to FIRRTL type if needed
static Value convertValueToFIRRTL(Value value, PatternRewriter &rewriter) {
  // If already FIRRTL type, return as-is
  if (isa<FIRRTLType>(value.getType())) {
    return value;
  }
  
  // If it's a constant, convert it to FIRRTL constant
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      FIRRTLType firrtlType = convertType(constOp.getType());
      bool isUnsigned = isa<UIntType>(firrtlType);
      llvm::APSInt constValue(intAttr.getValue(), isUnsigned);
      auto firrtlConstant = rewriter.create<ConstantOp>(
          constOp.getLoc(), firrtlType, constValue);
      return firrtlConstant.getResult();
    }
  }
  
  // For other values, create an unrealized cast for now
  // This will be handled properly by subsequent passes
  FIRRTLType firrtlType = convertType(value.getType());
  auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
      value.getLoc(), firrtlType, value);
  return cast.getResult(0);
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Convert arith.constant operations to firrtl.constant
class ArithConstantToFIRRTLPattern : public OpRewritePattern<arith::ConstantOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  
  LogicalResult matchAndRewrite(arith::ConstantOp op,
                               PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "Converting arith.constant to firrtl.constant\n");
    
    auto intAttr = dyn_cast<IntegerAttr>(op.getValue());
    if (!intAttr) {
      return rewriter.notifyMatchFailure(op, "Non-integer constant not supported");
    }
    
    // Convert to FIRRTL type directly
    Type originalType = op.getType();
    FIRRTLType firrtlType = convertType(originalType);
    
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
class ArithBinaryToFIRRTLPattern : public OpRewritePattern<ArithOp> {
public:
  using OpRewritePattern<ArithOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(ArithOp op,
                               PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "Converting arith binary op to FIRRTL\n");
    
    // Convert operands to FIRRTL types
    Value lhs = convertValueToFIRRTL(op.getLhs(), rewriter);
    Value rhs = convertValueToFIRRTL(op.getRhs(), rewriter);
    
    // For FIRRTL operations, let them infer their own result types
    // Don't force the result type - let FIRRTL type inference handle it
    auto firrtlOp = rewriter.create<FIRRTLOp>(
        op.getLoc(), lhs, rhs);
    
    // If the inferred result type is different from the expected type,
    // we need to cast it to the expected type
    Type expectedType = convertType(op.getType());
    Value result = firrtlOp.getResult();
    
    // If the types don't match, add a cast operation
    if (result.getType() != expectedType) {
      // Use bits operation to cast to the expected width
      if (auto expectedUInt = dyn_cast<UIntType>(expectedType)) {
        result = rewriter.create<BitsPrimOp>(
            op.getLoc(), expectedUInt, result, 
            expectedUInt.getWidth().value() - 1, 0);
      } else if (auto expectedSInt = dyn_cast<SIntType>(expectedType)) {
        result = rewriter.create<BitsPrimOp>(
            op.getLoc(), expectedSInt, result, 
            expectedSInt.getWidth().value() - 1, 0);
      }
    }
    
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Pattern to eliminate redundant unrealized conversion casts
/// Handles pattern: A -> cast -> B -> cast -> A => A
class EliminateRedundantCastsPattern : public OpRewritePattern<::mlir::UnrealizedConversionCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  
  LogicalResult matchAndRewrite(::mlir::UnrealizedConversionCastOp op,
                               PatternRewriter &rewriter) const override {
    // Check if this cast has a single operand and result
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
      return failure();
      
    Value input = op.getOperand(0);
    Value output = op.getResult(0);
    
    // Check if the input is also from a cast
    auto inputCast = input.getDefiningOp<::mlir::UnrealizedConversionCastOp>();
    if (!inputCast || inputCast.getNumOperands() != 1 || inputCast.getNumResults() != 1)
      return failure();
      
    Value originalValue = inputCast.getOperand(0);
    
    // Check if we have A -> B -> A pattern
    if (originalValue.getType() == output.getType()) {
      rewriter.replaceOp(op, originalValue);
      return success();
    }
    
    return failure();
  }
};

// Removed unused conversion patterns since we're using simpler rewrite patterns now

//===----------------------------------------------------------------------===//
// TypeConverter for Txn to FIRRTL
//===----------------------------------------------------------------------===//

class TxnToFIRRTLTypeConverter : public mlir::TypeConverter {
public:
  TxnToFIRRTLTypeConverter() {
    // Convert integer types to FIRRTL types
    addConversion([](Type type) -> Type {
      if (auto intType = dyn_cast<IntegerType>(type)) {
        if (intType.isSigned()) {
          return SIntType::get(type.getContext(), intType.getWidth());
        } else {
          return UIntType::get(type.getContext(), intType.getWidth());
        }
      }
      
      // FIRRTL types are already legal
      if (isa<FIRRTLType>(type)) {
        return type;
      }
      
      // Preserve other types (like txn.module type)
      return type;
    });
    
    // Add target materialization to handle conversions
    addTargetMaterialization([](OpBuilder &builder, Type resultType,
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
    if (failed(rewriter.convertRegionTypes(&newOp.getBody(), *getTypeConverter())))
      return failure();
      
    rewriter.replaceOp(op, newOp->getResults());
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
    if (failed(rewriter.convertRegionTypes(&cast<ValueMethodOp>(newOp).getBody(), *getTypeConverter())))
      return failure();
      
    rewriter.replaceOp(op, newOp->getResults());
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
    
    auto newOp = rewriter.create<txn::InstanceOp>(
        op.getLoc(), op.getResult().getType(), op.getSymName(), op.getModuleName(), typeArgs);
    
    rewriter.replaceOp(op, newOp.getResult());
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
    
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

/// Convert txn.call operation
class CallOpConversion : public OpConversionPattern<CallOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(CallOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    // Convert result types
    SmallVector<Type> convertedResults;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), convertedResults)))
      return failure();
    
    // Create new call with converted operands and result types
    auto newCall = rewriter.create<CallOp>(
        op.getLoc(), op.getCalleeAttr(), adaptor.getOperands(), convertedResults);
    
    rewriter.replaceOp(op, newCall.getResults());
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
    
    // Unrealized conversion casts are temporarily legal
    target.addLegalOp<::mlir::UnrealizedConversionCastOp>();
    
    // Arith operations are illegal
    target.addIllegalDialect<arith::ArithDialect>();
    
    // Txn operations are legal if their types are converted
    target.addDynamicallyLegalOp<ActionMethodOp>([&](ActionMethodOp op) {
      // Check if block arguments are converted
      for (auto &block : op.getBody()) {
        for (auto arg : block.getArguments()) {
          if (!typeConverter.isLegal(arg.getType()))
            return false;
        }
      }
      return true;
    });
    
    target.addDynamicallyLegalOp<ValueMethodOp>([&](ValueMethodOp op) {
      // Check result types
      for (Type type : op.getResultTypes()) {
        if (!typeConverter.isLegal(type))
          return false;
      }
      // Check block arguments
      for (auto &block : op.getBody()) {
        for (auto arg : block.getArguments()) {
          if (!typeConverter.isLegal(arg.getType()))
            return false;
        }
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
    
    target.addDynamicallyLegalOp<CallOp>([&](CallOp op) {
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
    
    // Other Txn operations are legal by default
    target.addLegalDialect<::sharp::txn::TxnDialect>();
    
    // Set up conversion patterns
    RewritePatternSet patterns(ctx);
    
    // Add Txn operation conversions
    patterns.add<ActionMethodOpConversion, ValueMethodOpConversion,
                 CallOpConversion, ReturnOpConversion>(typeConverter, ctx);
    // Instance and Primitive need special handling due to namespace conflicts
    patterns.insert<InstanceOpConversion>(typeConverter, ctx);
    patterns.insert<PrimitiveOpConversion>(typeConverter, ctx);
    
    // Add arith operation conversions
    patterns.add<ArithConstantToFIRRTLPattern>(ctx);
    patterns.add<ArithBinaryToFIRRTLPattern<arith::AddIOp, AddPrimOp>>(ctx);
    patterns.add<ArithBinaryToFIRRTLPattern<arith::SubIOp, SubPrimOp>>(ctx);
    patterns.add<ArithBinaryToFIRRTLPattern<arith::MulIOp, MulPrimOp>>(ctx);
    
    // Apply partial conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "LowerOpToFIRRTLPass failed\n");
      signalPassFailure();
      return;
    }
    
    // Run a cleanup pass to eliminate redundant casts
    RewritePatternSet cleanupPatterns(ctx);
    cleanupPatterns.add<EliminateRedundantCastsPattern>(ctx);
    
    if (failed(applyPatternsGreedily(module, std::move(cleanupPatterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to eliminate redundant casts\n");
      // Don't fail the pass for this - it's just a cleanup
    }
    
    LLVM_DEBUG(llvm::dbgs() << "LowerOpToFIRRTLPass completed successfully\n");
  }
};

} // end anonymous namespace

// Pass creation function is generated by tablegen

} // end namespace sharp
} // end namespace mlir