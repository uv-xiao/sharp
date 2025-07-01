//===- TxnToFIRRTLPass.cpp - Txn to FIRRTL Conversion Pass -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Txn to FIRRTL conversion pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "sharp/Conversion/Passes.h"
#include "sharp/Dialect/Txn/TxnDialect.h"
#include "sharp/Dialect/Txn/TxnOps.h"
#include "sharp/Dialect/Txn/TxnTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "txn-to-firrtl"

namespace mlir {
namespace sharp {

#define GEN_PASS_DEF_TXNTOFIRRTLCONVERSION
#include "sharp/Conversion/Passes.h.inc"

namespace {

using namespace ::sharp::txn;
using namespace ::circt::firrtl;

/// Conversion context to track state during translation
struct ConversionContext {
  /// Track generated FIRRTL values
  DenseMap<Value, Value> txnToFirrtl;
  
  /// Track will-fire signals for actions
  llvm::StringMap<Value> willFireSignals;
  
  /// Track which actions call which methods
  llvm::StringMap<SmallVector<StringRef>> methodCallers;
  
  /// Conflict matrix for current module
  llvm::StringMap<int> conflictMatrix;
  
  /// Reachability conditions for method calls
  DenseMap<Operation*, Value> reachabilityConditions;
  
  /// Track instance ports for method call connections
  DenseMap<StringRef, DenseMap<StringRef, Value>> instancePorts;
  
  /// Current FIRRTL module being built
  FModuleOp currentFIRRTLModule;
  
  /// Builder positioned inside FIRRTL module
  OpBuilder firrtlBuilder;
  
  ConversionContext(MLIRContext *ctx) : firrtlBuilder(ctx) {}
};

/// Helper to convert Sharp types to FIRRTL types
static FIRRTLType convertType(Type type) {
  MLIRContext *ctx = type.getContext();
  
  // Handle integer types
  if (auto intType = dyn_cast<IntegerType>(type)) {
    // Check if it's signed or unsigned
    // For now, treat i1 as unsigned (boolean), others as signed
    if (intType.getWidth() == 1) {
      return UIntType::get(ctx, 1);
    } else {
      return SIntType::get(ctx, intType.getWidth());
    }
  }
  
  // Handle vector types
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    // Convert to FIRRTL vector type
    auto elementType = convertType(vectorType.getElementType());
    if (!elementType) {
      return nullptr;
    }
    // FIRRTL uses FVectorType for vectors
    // Need to cast to FIRRTLBaseType
    if (auto baseType = dyn_cast<FIRRTLBaseType>(elementType)) {
      return FVectorType::get(baseType, vectorType.getNumElements());
    }
    llvm::errs() << "Vector element type must be a FIRRTL base type\n";
    return nullptr;
  }
  
  // Handle module types - not directly converted
  if (isa<ModuleType>(type)) {
    return nullptr;
  }
  
  // Default: try to use as-is if it's already a FIRRTL type
  if (auto firrtlType = dyn_cast<FIRRTLType>(type)) {
    return firrtlType;
  }
  
  llvm::errs() << "Unsupported type for FIRRTL conversion: " << type << "\n";
  return nullptr;
}

/// Helper to get or create a port in the FIRRTL module
static Value getOrCreatePort(FModuleOp module, StringRef portName, 
                            FIRRTLType type, Direction dir) {
  // Search for existing port
  for (auto [name, result] : llvm::zip(module.getPortNames(), 
                                       module.getBodyBlock()->getArguments())) {
    if (cast<StringAttr>(name).getValue() == portName) {
      return result;
    }
  }
  
  // Port doesn't exist - this shouldn't happen if module structure is correct
  llvm::errs() << "Port not found: " << portName << "\n";
  return nullptr;
}

/// Populate the conflict matrix from schedule operation
static void populateConflictMatrix(ScheduleOp schedule, ConversionContext &ctx) {
  ctx.conflictMatrix.clear();
  
  if (auto cmAttr = schedule.getConflictMatrix()) {
    auto cmDict = cast<DictionaryAttr>(*cmAttr);
    for (auto entry : cmDict) {
      auto key = entry.getName().str();
      auto value = cast<IntegerAttr>(entry.getValue()).getInt();
      ctx.conflictMatrix[key] = value;
    }
  }
}

/// Get conflict relation between two actions
static ConflictRelation getConflictRelation(StringRef a1, StringRef a2,
                                           const ConversionContext &ctx) {
  // Check both orderings
  std::string key1 = (a1 + "," + a2).str();
  std::string key2 = (a2 + "," + a1).str();
  
  auto it1 = ctx.conflictMatrix.find(key1);
  if (it1 != ctx.conflictMatrix.end()) {
    return static_cast<ConflictRelation>(it1->second);
  }
  
  auto it2 = ctx.conflictMatrix.find(key2);
  if (it2 != ctx.conflictMatrix.end()) {
    // Swap SB and SA when reversing order
    auto rel = static_cast<ConflictRelation>(it2->second);
    if (rel == ConflictRelation::SequenceBefore) return ConflictRelation::SequenceAfter;
    if (rel == ConflictRelation::SequenceAfter) return ConflictRelation::SequenceBefore;
    return rel;
  }
  
  // Default to conflict-free
  return ConflictRelation::ConflictFree;
}

/// Generate will-fire logic for an action
static Value generateWillFire(StringRef actionName, Value enabled,
                            ArrayRef<StringRef> schedule,
                            ConversionContext &ctx) {
  auto &builder = ctx.firrtlBuilder;
  auto loc = builder.getUnknownLoc();
  [[maybe_unused]] MLIRContext *mlirCtx = builder.getContext();
  
  // Start with enabled signal
  Value wf = enabled;
  
  if (!wf) {
    llvm::errs() << "generateWillFire: enabled signal is null for " << actionName << "\n";
    return nullptr;
  }
  
  // Check conflicts with earlier actions
  for (StringRef earlier : schedule) {
    if (earlier == actionName) break;
    
    auto rel = getConflictRelation(earlier, actionName, ctx);
    
    // Generate conflict check if needed
    if (rel == ConflictRelation::Conflict || rel == ConflictRelation::SequenceBefore) {
      Value earlierWF = ctx.willFireSignals[earlier];
      if (!earlierWF) {
        // Skip - this is likely a value method which doesn't have will-fire
        continue;
      }
      
      // Create: wf = wf & !earlier_wf
      auto notEarlier = builder.create<NotPrimOp>(loc, earlierWF);
      wf = builder.create<AndPrimOp>(loc, wf, notEarlier);
    }
  }
  
  // TODO: Add conflict_inside calculation once reachability is integrated
  
  // Create node for will-fire signal
  auto wfNode = builder.create<NodeOp>(loc, wf,
                                      (actionName.str() + "_wf"),
                                      NameKindEnum::DroppableName);
  
  // Store in context
  ctx.willFireSignals[actionName] = wfNode.getResult();
  
  return wfNode.getResult();
}

/// Analyze method calls in an action to track reachability
static void analyzeActionReachability(Operation *action, ConversionContext &ctx) {
  // Track current path condition (starts as true)
  auto loc = action->getLoc();
  auto intType = IntType::get(ctx.firrtlBuilder.getContext(), false, 1);
  Value trueVal = ctx.firrtlBuilder.create<ConstantOp>(loc, Type(intType), APSInt(APInt(1, 1), true));
  
  struct ReachabilityState {
    Value pathCondition;
    Region *region;
  };
  SmallVector<ReachabilityState> workList;
  
  // Helper to process a region with given path condition
  auto processRegion = [&](Region &region, Value pathCond) {
    for (auto &op : region.getOps()) {
      if (auto ifOp = dyn_cast<IfOp>(&op)) {
        // Get the converted FIRRTL condition
        Value firrtlCond = ctx.txnToFirrtl.lookup(ifOp.getCondition());
        if (!firrtlCond) {
          // If condition hasn't been converted yet, skip this if
          continue;
        }
        
        // Update path condition for then branch
        Value thenCond = ctx.firrtlBuilder.create<AndPrimOp>(
            ifOp.getLoc(), pathCond, firrtlCond);
        workList.push_back({thenCond, &ifOp.getThenRegion()});
        
        // Update path condition for else branch if exists
        if (!ifOp.getElseRegion().empty()) {
          Value notCond = ctx.firrtlBuilder.create<NotPrimOp>(
              ifOp.getLoc(), firrtlCond);
          Value elseCond = ctx.firrtlBuilder.create<AndPrimOp>(
              ifOp.getLoc(), pathCond, notCond);
          workList.push_back({elseCond, &ifOp.getElseRegion()});
        }
      } else if (auto callOp = dyn_cast<CallOp>(&op)) {
        // Record reachability condition for this method call
        ctx.reachabilityConditions[callOp] = pathCond;
      }
    }
  };
  
  // Start with the body of the action
  if (auto rule = dyn_cast<RuleOp>(action)) {
    processRegion(rule.getBody(), trueVal);
  } else if (auto method = dyn_cast<ActionMethodOp>(action)) {
    processRegion(method.getBody(), trueVal);
  }
  
  // Process all regions in the worklist
  while (!workList.empty()) {
    auto state = workList.pop_back_val();
    processRegion(*state.region, state.pathCondition);
  }
}

/// Calculate conflict_inside for an action
static Value calculateConflictInside(Operation *action, ConversionContext &ctx) {
  // First analyze reachability
  analyzeActionReachability(action, ctx);
  
  // Collect all method calls in this action
  SmallVector<CallOp> methodCalls;
  action->walk([&](CallOp call) {
    methodCalls.push_back(call);
  });
  
  // Start with false (no conflicts)
  auto loc = action->getLoc();
  auto intType = IntType::get(ctx.firrtlBuilder.getContext(), false, 1);
  Value conflictInside = ctx.firrtlBuilder.create<ConstantOp>(loc, Type(intType), APSInt(APInt(1, 0), true));
  
  // Check each pair of method calls
  for (size_t i = 0; i < methodCalls.size(); ++i) {
    for (size_t j = i + 1; j < methodCalls.size(); ++j) {
      auto call1 = methodCalls[i];
      auto call2 = methodCalls[j];
      
      // Get the called methods from the callee symbol reference
      auto callee1 = call1.getCallee();
      auto callee2 = call2.getCallee();
      
      // Extract instance and method from nested symbol ref
      StringRef inst1, method1, inst2, method2;
      if (callee1.getNestedReferences().size() == 1) {
        inst1 = callee1.getRootReference().getValue();
        method1 = callee1.getNestedReferences()[0].getValue();
      }
      if (callee2.getNestedReferences().size() == 1) {
        inst2 = callee2.getRootReference().getValue();
        method2 = callee2.getNestedReferences()[0].getValue();
      }
      
      // Build the conflict key
      std::string key = (inst1 + "::" + method1 + "," + inst2 + "::" + method2).str();
      
      // Check if methods conflict
      auto it = ctx.conflictMatrix.find(key);
      if (it != ctx.conflictMatrix.end() && it->second == static_cast<int>(ConflictRelation::Conflict)) {
        // Get reachability conditions
        Value reach1 = ctx.reachabilityConditions[call1];
        Value reach2 = ctx.reachabilityConditions[call2];
        
        if (reach1 && reach2) {
          // Both methods can be reached if both conditions can be true
          Value bothReachable = ctx.firrtlBuilder.create<AndPrimOp>(
              action->getLoc(), reach1, reach2);
          
          // Add to conflict_inside
          conflictInside = ctx.firrtlBuilder.create<OrPrimOp>(
              action->getLoc(), conflictInside, bothReachable);
        }
      }
    }
  }
  
  return conflictInside;
}

/// Convert method/rule body operations to FIRRTL
static LogicalResult convertBodyOps(Region &region, ConversionContext &ctx) {
  // Process all blocks in the region
  for (auto &block : region) {
    // Process operations in order
    for (auto &op : block.getOperations()) {
    if (auto ifOp = dyn_cast<IfOp>(&op)) {
      // Get the converted condition
      Value firrtlCond = ctx.txnToFirrtl.lookup(ifOp.getCondition());
      if (!firrtlCond) {
        // Condition not converted yet - this shouldn't happen
        return ifOp.emitError("condition not converted to FIRRTL");
      }
      
      // Convert to FIRRTL when
      bool hasElse = !ifOp.getElseRegion().empty();
      ctx.firrtlBuilder.create<WhenOp>(ifOp.getLoc(), firrtlCond, 
                                      hasElse, [&]() {
        if (failed(convertBodyOps(ifOp.getThenRegion(), ctx)))
          return;
      }, [&]() {
        if (hasElse) {
          if (failed(convertBodyOps(ifOp.getElseRegion(), ctx)))
            return;
        }
      });
    } else if (auto callOp = dyn_cast<CallOp>(&op)) {
      // Handle method calls
      auto callee = callOp.getCallee();
      
      if (callee.getNestedReferences().size() == 0) {
        // Local method call (within same module)
        StringRef methodName = callee.getRootReference().getValue();
        
        if (callOp.getNumResults() > 0) {
          // Value method call - for local calls in rules, we just use the output
          // The method is always enabled (combinational)
          auto portNames = ctx.currentFIRRTLModule.getPortNames();
          auto blockArgs = ctx.currentFIRRTLModule.getBodyBlock()->getArguments();
          
          // Find output port and map result
          auto outputPortName = (methodName + "OUT").str();
          for (size_t i = 0; i < portNames.size(); ++i) {
            if (cast<StringAttr>(portNames[i]).getValue() == outputPortName) {
              // Create a node to read the value
              auto node = ctx.firrtlBuilder.create<NodeOp>(
                  callOp.getLoc(), blockArgs[i], 
                  ctx.firrtlBuilder.getStringAttr(methodName + "_call"));
              ctx.txnToFirrtl[callOp.getResult(0)] = node.getResult();
              break;
            }
          }
        }
      } else if (callee.getNestedReferences().size() == 1) {
        StringRef instName = callee.getRootReference().getValue();
        StringRef methodName = callee.getNestedReferences()[0].getValue();
        ctx.methodCallers[methodName].push_back(instName);
        
        // Connect to instance ports
        auto instPorts = ctx.instancePorts.find(instName);
        if (instPorts != ctx.instancePorts.end()) {
          // Handle value method calls
          if (callOp.getNumResults() > 0) {
            // Find the output port for this method
            auto outputPortName = (methodName + "OUT").str();
            auto outputPort = instPorts->second.find(outputPortName);
            
            if (outputPort != instPorts->second.end()) {
              // Enable the method
              auto enablePortName = (methodName + "_EN").str();
              auto enablePort = instPorts->second.find(enablePortName);
              if (enablePort != instPorts->second.end()) {
                auto intTy = IntType::get(ctx.firrtlBuilder.getContext(), false, 1);
                auto trueVal = ctx.firrtlBuilder.create<ConstantOp>(
                    callOp.getLoc(), Type(intTy), APSInt(APInt(1, 1), true));
                ctx.firrtlBuilder.create<ConnectOp>(callOp.getLoc(),
                                                   enablePort->second, trueVal);
              }
              
              // Map the result
              ctx.txnToFirrtl[callOp.getResult(0)] = outputPort->second;
            }
          } else {
            // Action method call
            // Set input arguments
            for (size_t i = 0; i < callOp.getNumOperands(); ++i) {
              auto argValue = ctx.txnToFirrtl.lookup(callOp.getOperand(i));
              if (argValue) {
                auto argPortName = (methodName + "OUT").str(); // For single arg
                auto argPort = instPorts->second.find(argPortName);
                if (argPort != instPorts->second.end()) {
                  ctx.firrtlBuilder.create<ConnectOp>(callOp.getLoc(),
                                                     argPort->second, argValue);
                }
              }
            }
            
            // Enable the action
            auto enablePortName = (methodName + "EN").str();
            auto enablePort = instPorts->second.find(enablePortName);
            if (enablePort != instPorts->second.end()) {
              auto intTy = IntType::get(ctx.firrtlBuilder.getContext(), false, 1);
              auto trueVal = ctx.firrtlBuilder.create<ConstantOp>(
                  callOp.getLoc(), Type(intTy), APSInt(APInt(1, 1), true));
              ctx.firrtlBuilder.create<ConnectOp>(callOp.getLoc(),
                                                 enablePort->second, trueVal);
            }
          }
        }
        
        // For action methods with arguments (like write)
        if (callOp.getNumOperands() > 0 && methodName == "write") {
          // Convert the argument
          Value arg = callOp.getOperand(0);
          Value firrtlArg = ctx.txnToFirrtl.lookup(arg);
          if (firrtlArg) {
            // In a real implementation, this would connect to the instance's input port
            // For now, just track that we have the converted argument
          }
        }
      }
    } else if (auto returnOp = dyn_cast<ReturnOp>(&op)) {
      // Handle return values if any
      // The return operand should already be converted, so nothing to do here
      // The caller will look up the converted value using ctx.txnToFirrtl
    } else if (auto yieldOp = dyn_cast<YieldOp>(&op)) {
      // Yield operations don't generate FIRRTL code, they just terminate blocks
    } else if (auto constOp = dyn_cast<arith::ConstantOp>(&op)) {
      // Convert constants to FIRRTL
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        auto firrtlType = convertType(constOp.getType());
        if (firrtlType) {
          auto apInt = intAttr.getValue();
          // FIRRTL constants need to know signedness
          // Check if the FIRRTL type is signed or unsigned
          bool isUnsigned = isa<UIntType>(firrtlType);
          auto firrtlConst = ctx.firrtlBuilder.create<ConstantOp>(
              constOp.getLoc(), Type(firrtlType), 
              APSInt(apInt, isUnsigned));
          ctx.txnToFirrtl[constOp.getResult()] = firrtlConst;
        }
      } else if (auto boolAttr = dyn_cast<BoolAttr>(constOp.getValue())) {
        // Convert boolean constants
        auto intType = IntType::get(ctx.firrtlBuilder.getContext(), false, 1);
        auto value = boolAttr.getValue() ? 1 : 0;
        auto firrtlConst = ctx.firrtlBuilder.create<ConstantOp>(
            constOp.getLoc(), Type(intType), 
            APSInt(APInt(1, value), true));
        ctx.txnToFirrtl[constOp.getResult()] = firrtlConst;
      }
    } else if (auto addOp = dyn_cast<arith::AddIOp>(&op)) {
      // Convert integer addition
      Value lhs = ctx.txnToFirrtl.lookup(addOp.getLhs());
      Value rhs = ctx.txnToFirrtl.lookup(addOp.getRhs());
      if (lhs && rhs) {
        auto sum = ctx.firrtlBuilder.create<AddPrimOp>(addOp.getLoc(), lhs, rhs);
        // FIRRTL add increases bit width by 1, so we need to truncate
        auto targetType = convertType(addOp.getType());
        if (targetType && isa<IntType>(targetType)) {
          auto intType = cast<IntType>(targetType);
          // Use bits operation to extract the lower bits
          int targetWidth = intType.getWidth().value();
          auto truncated = ctx.firrtlBuilder.create<BitsPrimOp>(
              addOp.getLoc(), sum.getResult(), targetWidth - 1, 0);
          
          // Check if we need signed or unsigned result
          if (isa<SIntType>(targetType)) {
            // Convert to signed
            auto asSigned = ctx.firrtlBuilder.create<AsSIntPrimOp>(
                addOp.getLoc(), truncated.getResult());
            ctx.txnToFirrtl[addOp.getResult()] = asSigned;
          } else {
            // Keep as unsigned
            ctx.txnToFirrtl[addOp.getResult()] = truncated;
          }
        } else {
          ctx.txnToFirrtl[addOp.getResult()] = sum;
        }
      }
    } else if (auto cmpOp = dyn_cast<arith::CmpIOp>(&op)) {
      // Convert integer comparison
      Value lhs = ctx.txnToFirrtl.lookup(cmpOp.getLhs());
      Value rhs = ctx.txnToFirrtl.lookup(cmpOp.getRhs());
      if (lhs && rhs) {
        Value result;
        switch (cmpOp.getPredicate()) {
          case arith::CmpIPredicate::eq:
            result = ctx.firrtlBuilder.create<EQPrimOp>(cmpOp.getLoc(), lhs, rhs);
            break;
          case arith::CmpIPredicate::ne:
            result = ctx.firrtlBuilder.create<NEQPrimOp>(cmpOp.getLoc(), lhs, rhs);
            break;
          case arith::CmpIPredicate::slt:
          case arith::CmpIPredicate::ult:
            result = ctx.firrtlBuilder.create<LTPrimOp>(cmpOp.getLoc(), lhs, rhs);
            break;
          case arith::CmpIPredicate::sle:
          case arith::CmpIPredicate::ule:
            result = ctx.firrtlBuilder.create<LEQPrimOp>(cmpOp.getLoc(), lhs, rhs);
            break;
          case arith::CmpIPredicate::sgt:
          case arith::CmpIPredicate::ugt:
            result = ctx.firrtlBuilder.create<GTPrimOp>(cmpOp.getLoc(), lhs, rhs);
            break;
          case arith::CmpIPredicate::sge:
          case arith::CmpIPredicate::uge:
            result = ctx.firrtlBuilder.create<GEQPrimOp>(cmpOp.getLoc(), lhs, rhs);
            break;
        }
        if (result)
          ctx.txnToFirrtl[cmpOp.getResult()] = result;
      }
    } else if (auto subOp = dyn_cast<arith::SubIOp>(&op)) {
      // Convert integer subtraction
      Value lhs = ctx.txnToFirrtl.lookup(subOp.getLhs());
      Value rhs = ctx.txnToFirrtl.lookup(subOp.getRhs());
      if (lhs && rhs) {
        auto diff = ctx.firrtlBuilder.create<SubPrimOp>(subOp.getLoc(), lhs, rhs);
        // FIRRTL sub increases bit width by 1, so we need to truncate
        auto targetType = convertType(subOp.getType());
        if (targetType && isa<IntType>(targetType)) {
          auto intType = cast<IntType>(targetType);
          int targetWidth = intType.getWidth().value();
          auto truncated = ctx.firrtlBuilder.create<BitsPrimOp>(
              subOp.getLoc(), diff.getResult(), targetWidth - 1, 0);
          
          if (isa<SIntType>(targetType)) {
            auto asSigned = ctx.firrtlBuilder.create<AsSIntPrimOp>(
                subOp.getLoc(), truncated.getResult());
            ctx.txnToFirrtl[subOp.getResult()] = asSigned;
          } else {
            ctx.txnToFirrtl[subOp.getResult()] = truncated;
          }
        } else {
          ctx.txnToFirrtl[subOp.getResult()] = diff;
        }
      }
    } else if (auto mulOp = dyn_cast<arith::MulIOp>(&op)) {
      // Convert integer multiplication
      Value lhs = ctx.txnToFirrtl.lookup(mulOp.getLhs());
      Value rhs = ctx.txnToFirrtl.lookup(mulOp.getRhs());
      if (lhs && rhs) {
        auto prod = ctx.firrtlBuilder.create<MulPrimOp>(mulOp.getLoc(), lhs, rhs);
        // FIRRTL mul doubles bit width, so we need to truncate
        auto targetType = convertType(mulOp.getType());
        if (targetType && isa<IntType>(targetType)) {
          auto intType = cast<IntType>(targetType);
          int targetWidth = intType.getWidth().value();
          auto truncated = ctx.firrtlBuilder.create<BitsPrimOp>(
              mulOp.getLoc(), prod.getResult(), targetWidth - 1, 0);
          
          if (isa<SIntType>(targetType)) {
            auto asSigned = ctx.firrtlBuilder.create<AsSIntPrimOp>(
                mulOp.getLoc(), truncated.getResult());
            ctx.txnToFirrtl[mulOp.getResult()] = asSigned;
          } else {
            ctx.txnToFirrtl[mulOp.getResult()] = truncated;
          }
        } else {
          ctx.txnToFirrtl[mulOp.getResult()] = prod;
        }
      }
    } else if (auto selectOp = dyn_cast<arith::SelectOp>(&op)) {
      // Convert select (ternary) operation
      Value cond = ctx.txnToFirrtl.lookup(selectOp.getCondition());
      Value trueVal = ctx.txnToFirrtl.lookup(selectOp.getTrueValue());
      Value falseVal = ctx.txnToFirrtl.lookup(selectOp.getFalseValue());
      if (cond && trueVal && falseVal) {
        auto mux = ctx.firrtlBuilder.create<MuxPrimOp>(
            selectOp.getLoc(), cond, trueVal, falseVal);
        ctx.txnToFirrtl[selectOp.getResult()] = mux;
      }
    }
    // Add more operation conversions as needed
  }
  }
  return success();
}

/// Convert a Txn module to FIRRTL
static LogicalResult convertModule(::sharp::txn::ModuleOp txnModule, 
                                 ConversionContext &ctx,
                                 ::circt::firrtl::CircuitOp circuit) {
  MLIRContext *mlirCtx = txnModule->getContext();
  OpBuilder builder(mlirCtx);
  
  // Find schedule to get action ordering and conflict matrix
  ScheduleOp schedule;
  txnModule.walk([&](ScheduleOp op) {
    schedule = op;
    return WalkResult::interrupt();
  });
  
  if (!schedule) {
    return txnModule.emitError("Txn module missing schedule operation");
  }
  
  populateConflictMatrix(schedule, ctx);
  
  // Create FIRRTL module structure
  SmallVector<PortInfo> ports;
  
  // Add clock and reset
  ports.push_back({builder.getStringAttr("clock"), 
                   ClockType::get(mlirCtx), Direction::In});
  ports.push_back({builder.getStringAttr("reset"), 
                   UIntType::get(mlirCtx, 1), Direction::In});
  
  // Process methods to add ports
  txnModule.walk([&](Operation *op) {
    if (auto valueMethod = dyn_cast<ValueMethodOp>(op)) {
      // Get method attributes
      StringRef prefix = valueMethod.getPrefix().value_or(valueMethod.getSymName());
      StringRef resultPostfix = valueMethod.getResult().value_or("OUT");
      
      // Add data output port
      auto returnType = valueMethod.getFunctionType().getResult(0);
      if (auto firrtlType = convertType(returnType)) {
        ports.push_back({builder.getStringAttr((prefix + resultPostfix).str()),
                        firrtlType, Direction::Out});
      }
      
      // Add enable input port for value methods
      StringRef enablePostfix = "_EN";
      ports.push_back({builder.getStringAttr((prefix + enablePostfix).str()),
                      UIntType::get(mlirCtx, 1), Direction::In});
      
    } else if (auto actionMethod = dyn_cast<ActionMethodOp>(op)) {
      // Get method attributes
      StringRef prefix = actionMethod.getPrefix().value_or(actionMethod.getSymName());
      StringRef resultPostfix = actionMethod.getResult().value_or("OUT");
      StringRef enablePostfix = actionMethod.getEnable().value_or("EN");
      StringRef readyPostfix = actionMethod.getReady().value_or("RDY");
      
      // Add data input ports for arguments
      auto funcType = actionMethod.getFunctionType();
      if (funcType.getNumInputs() > 0) {
        // For simplicity, assume single argument
        if (auto firrtlType = convertType(funcType.getInput(0))) {
          ports.push_back({builder.getStringAttr((prefix + resultPostfix).str()),
                          firrtlType, Direction::In});
        }
      }
      
      // Add enable port (unless always_enable)
      if (!actionMethod.getAlwaysEnable()) {
        ports.push_back({builder.getStringAttr((prefix + enablePostfix).str()),
                        UIntType::get(mlirCtx, 1), Direction::In});
      }
      
      // Add ready port (unless always_ready)
      if (!actionMethod.getAlwaysReady()) {
        ports.push_back({builder.getStringAttr((prefix + readyPostfix).str()),
                        UIntType::get(mlirCtx, 1), Direction::Out});
      }
    }
  });
  
  // Create FIRRTL module
  builder.setInsertionPointToEnd(circuit.getBodyBlock());
  auto moduleName = txnModule.getSymName();
  if (moduleName.empty()) {
    return txnModule.emitError("Txn module missing symbol name");
  }
  
  auto firrtlModule = builder.create<FModuleOp>(
      txnModule.getLoc(), 
      builder.getStringAttr(moduleName),
      ConventionAttr::get(mlirCtx, Convention::Internal),
      ports);
  
  // Set up conversion context
  ctx.currentFIRRTLModule = firrtlModule;
  ctx.firrtlBuilder.setInsertionPointToStart(firrtlModule.getBodyBlock());
  
  // Get clock and reset signals
  Value clock = firrtlModule.getBodyBlock()->getArgument(0);
  Value reset = firrtlModule.getBodyBlock()->getArgument(1);
  
  // Create submodule instances
  DenseMap<StringRef, ::circt::firrtl::InstanceOp> firrtlInstances;
  DenseMap<StringRef, DenseMap<StringRef, Value>> instancePorts;
  
  txnModule.walk([&](::sharp::txn::InstanceOp txnInst) {
    // Find the target module to get its interface
    auto targetModuleName = txnInst.getModuleName();
    ::circt::firrtl::FModuleOp targetFIRRTLModule;
    
    // Look for the FIRRTL module in the circuit
    circuit.walk([&](::circt::firrtl::FModuleOp fmodule) {
      if (fmodule.getName() == targetModuleName) {
        targetFIRRTLModule = fmodule;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    
    if (!targetFIRRTLModule) {
      // Module not yet converted - this shouldn't happen with proper ordering
      return;
    }
    
    // Create the instance - use the simpler FModuleLike constructor
    auto firrtlInst = ctx.firrtlBuilder.create<::circt::firrtl::InstanceOp>(
        txnInst.getLoc(),
        targetFIRRTLModule,
        txnInst.getName(),
        NameKindEnum::InterestingName);
    
    // Connect clock and reset
    auto clock = firrtlModule.getBodyBlock()->getArgument(0);
    auto reset = firrtlModule.getBodyBlock()->getArgument(1);
    
    // Find clock and reset ports by name and connect them
    auto targetPorts = targetFIRRTLModule.getPorts();
    for (size_t i = 0; i < targetPorts.size(); ++i) {
      auto portName = cast<StringAttr>(targetPorts[i].name).getValue();
      auto portDir = targetPorts[i].direction;
      
      if (portName == "clock" && portDir == Direction::In) {
        ctx.firrtlBuilder.create<ConnectOp>(txnInst.getLoc(), 
                                           firrtlInst.getResult(i), clock);
      } else if (portName == "reset" && portDir == Direction::In) {
        ctx.firrtlBuilder.create<ConnectOp>(txnInst.getLoc(), 
                                           firrtlInst.getResult(i), reset);
      }
      
      // Track all ports for later use
      instancePorts[txnInst.getName()][portName] = firrtlInst.getResult(i);
    }
    
    firrtlInstances[txnInst.getName()] = firrtlInst;
  });
  
  // Store instance ports in context for method call connections
  ctx.instancePorts = instancePorts;
  
  // Generate will-fire logic for each action in schedule order
  auto scheduleArrayAttr = schedule.getActions();
  SmallVector<StringRef> scheduleOrder;
  for (auto attr : scheduleArrayAttr) {
    auto symRef = cast<SymbolRefAttr>(attr);
    scheduleOrder.push_back(symRef.getRootReference().getValue());
  }
  
  // First pass: generate enabled signals and conflict_inside
  DenseMap<StringRef, Value> enabledSignals;
  DenseMap<StringRef, Value> conflictInsideSignals;
  
  for (StringRef name : scheduleOrder) {
    // Find the action (rule or method)
    Operation *action = nullptr;
    bool isValueMethod = false;
    
    txnModule.walk([&](Operation *op) {
      if (auto rule = dyn_cast<RuleOp>(op)) {
        if (rule.getSymName() == name) {
          action = op;
          return WalkResult::interrupt();
        }
      } else if (auto method = dyn_cast<ActionMethodOp>(op)) {
        if (method.getSymName() == name) {
          action = op;
          return WalkResult::interrupt();
        }
      } else if (auto valueMethod = dyn_cast<ValueMethodOp>(op)) {
        if (valueMethod.getSymName() == name) {
          // Value methods shouldn't be in schedule, but handle gracefully
          isValueMethod = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    
    if (isValueMethod) {
      // Skip value methods in schedule - they don't have will-fire logic
      continue;
    }
    
    if (!action) {
      return txnModule.emitError("Action not found in schedule: ") << name;
    }
    
    // Generate enabled signal
    Value enabled;
    if (auto rule = dyn_cast<RuleOp>(action)) {
      // Evaluate rule guard
      // TODO: Properly evaluate guard condition
      // For now, rules are always enabled
      auto intTy = IntType::get(mlirCtx, false, 1);
      enabled = ctx.firrtlBuilder.create<ConstantOp>(
          action->getLoc(), Type(intTy), APSInt(APInt(1, 1), true));
    } else if (auto method = dyn_cast<ActionMethodOp>(action)) {
      // Methods use their enable port
      StringRef prefix = method.getPrefix().value_or(method.getSymName());
      StringRef enablePostfix = method.getEnable().value_or("EN");
      
      if (method.getAlwaysEnable()) {
        // Always enabled
        auto intTy = IntType::get(mlirCtx, false, 1);
        enabled = ctx.firrtlBuilder.create<ConstantOp>(
            action->getLoc(), Type(intTy), APSInt(APInt(1, 1), true));
      } else {
        // Get enable port
        std::string portName = (prefix + enablePostfix).str();
        enabled = getOrCreatePort(firrtlModule, portName, 
                                UIntType::get(mlirCtx, 1), Direction::In);
      }
    }
    
    enabledSignals[name] = enabled;
    
    // Don't calculate conflict_inside yet - will do after body conversion
    conflictInsideSignals[name] = nullptr;
  }
  
  // Second pass: generate will-fire signals with proper conflict checking
  for (StringRef name : scheduleOrder) {
    Value enabled = enabledSignals[name];
    if (!enabled) {
      // Skip value methods (they don't have enabled signals)
      continue;
    }
    
    Value conflictInside = conflictInsideSignals[name];
    
    // Check for conflicts with earlier actions and conflict_inside
    Value wf = enabled;
    
    // Add conflict_inside check
    if (conflictInside) {
      Value noConflictInside = ctx.firrtlBuilder.create<NotPrimOp>(
          conflictInside.getLoc(), conflictInside);
      wf = ctx.firrtlBuilder.create<AndPrimOp>(wf.getLoc(), wf, noConflictInside);
    }
    
    // Generate final will-fire signal
    wf = generateWillFire(name, wf, scheduleOrder, ctx);
  }
  
  // Convert value methods
  txnModule.walk([&](ValueMethodOp valueMethod) {
    StringRef prefix = valueMethod.getPrefix().value_or(valueMethod.getSymName());
    StringRef resultPostfix = valueMethod.getResult().value_or("OUT");
    StringRef enablePostfix = "_EN";
    
    // Find the output port
    Value outputPort;
    Value enablePort;
    std::string outputName = (prefix + resultPostfix).str();
    std::string enableName = (prefix + enablePostfix).str();
    
    auto portNames = firrtlModule.getPortNames();
    auto blockArgs = firrtlModule.getBodyBlock()->getArguments();
    for (size_t i = 0; i < portNames.size(); ++i) {
      if (cast<StringAttr>(portNames[i]).getValue() == outputName) {
        outputPort = blockArgs[i];
      } else if (cast<StringAttr>(portNames[i]).getValue() == enableName) {
        enablePort = blockArgs[i];
      }
    }
    
    if (!outputPort) return;
    
    // Create wire for method result
    auto resultWire = ctx.firrtlBuilder.create<WireOp>(
        valueMethod.getLoc(), outputPort.getType(), (prefix + "_result").str());
    
    // Convert method body to compute result
    ctx.txnToFirrtl.clear(); // Clear conversions for this method
    
    // Map method arguments to FIRRTL ports
    for (unsigned i = 0; i < valueMethod.getNumArguments(); ++i) {
      Value methodArg = valueMethod.getArgument(i);
      // For value methods with args, we created ports with pattern: methodOUT_a, methodOUT_b
      std::string argPortName = (prefix + "OUT_" + 
                                 std::string(1, 'a' + i)).str();
      
      for (size_t j = 0; j < portNames.size(); ++j) {
        if (cast<StringAttr>(portNames[j]).getValue() == argPortName) {
          ctx.txnToFirrtl[methodArg] = blockArgs[j];
          break;
        }
      }
    }
    
    if (failed(convertBodyOps(valueMethod.getBody(), ctx))) {
      valueMethod.emitError("Failed to convert value method body");
      return;
    }
    
    // Get the return value
    Value returnValue;
    valueMethod.walk([&](ReturnOp ret) {
      if (ret.getNumOperands() > 0) {
        returnValue = ctx.txnToFirrtl.lookup(ret.getOperand(0));
      }
    });
    
    if (returnValue) {
      ctx.firrtlBuilder.create<ConnectOp>(valueMethod.getLoc(), 
                                         resultWire.getResult(), returnValue);
    } else {
      // No return value found, use invalid
      auto invalid = ctx.firrtlBuilder.create<InvalidValueOp>(
          valueMethod.getLoc(), outputPort.getType());
      ctx.firrtlBuilder.create<ConnectOp>(valueMethod.getLoc(), 
                                         resultWire.getResult(), invalid.getResult());
    }
    
    // Always connect result to output port (value methods are combinational)
    ctx.firrtlBuilder.create<ConnectOp>(valueMethod.getLoc(), 
                                       outputPort, resultWire.getResult());
  });
  
  // Convert action methods
  txnModule.walk([&](ActionMethodOp actionMethod) {
    StringRef prefix = actionMethod.getPrefix().value_or(actionMethod.getSymName());
    [[maybe_unused]] StringRef enablePostfix = actionMethod.getEnable().value_or("EN");
    StringRef readyPostfix = actionMethod.getReady().value_or("RDY");
    
    // Get will-fire signal for this method
    Value wf = ctx.willFireSignals[actionMethod.getSymName()];
    if (!wf) return;
    
    // Find ready port if exists and generate ready logic
    if (!actionMethod.getAlwaysReady()) {
      std::string readyName = (prefix + readyPostfix).str();
      auto portNames = firrtlModule.getPortNames();
      auto blockArgs = firrtlModule.getBodyBlock()->getArguments();
      for (size_t i = 0; i < portNames.size(); ++i) {
        if (cast<StringAttr>(portNames[i]).getValue() == readyName) {
          // Ready signal indicates no conflicts prevent execution
          // Calculate based on conflicts with other actions
          auto intTy = IntType::get(mlirCtx, false, 1);
          Value ready = ctx.firrtlBuilder.create<ConstantOp>(
              actionMethod.getLoc(), Type(intTy), APSInt(APInt(1, 1), true));
          
          // Check conflicts with all other actions
          for (StringRef other : scheduleOrder) {
            if (other == actionMethod.getSymName()) continue;
            
            // Skip value methods - they don't have will-fire signals
            if (!ctx.willFireSignals.count(other)) continue;
            
            auto rel = getConflictRelation(other, actionMethod.getSymName(), ctx);
            if (rel == ConflictRelation::Conflict || rel == ConflictRelation::SequenceBefore) {
              // If other action will fire, this method is not ready
              Value otherWF = ctx.willFireSignals[other];
              if (!otherWF) {
                continue;
              }
              Value notOther = ctx.firrtlBuilder.create<NotPrimOp>(
                  actionMethod.getLoc(), otherWF);
              ready = ctx.firrtlBuilder.create<AndPrimOp>(
                  actionMethod.getLoc(), ready, notOther);
            }
          }
          
          ctx.firrtlBuilder.create<ConnectOp>(actionMethod.getLoc(), blockArgs[i], ready);
          break;
        }
      }
    }
    
    // Execute method body when will-fire is true
    ctx.firrtlBuilder.create<WhenOp>(actionMethod.getLoc(), wf, false, [&]() {
      // Convert method body
      ctx.txnToFirrtl.clear();
      
      // Map method arguments to FIRRTL ports
      auto methodPortNames = firrtlModule.getPortNames();
      auto methodBlockArgs = firrtlModule.getBodyBlock()->getArguments();
      for (unsigned i = 0; i < actionMethod.getNumArguments(); ++i) {
        Value methodArg = actionMethod.getArgument(i);
        // Find corresponding FIRRTL input port
        std::string argPortName = (prefix + "OUT").str();
        if (actionMethod.getNumArguments() > 1) {
          argPortName = (prefix + "OUT_arg" + std::to_string(i)).str();
        }
        
        for (size_t j = 0; j < methodPortNames.size(); ++j) {
          if (cast<StringAttr>(methodPortNames[j]).getValue() == argPortName) {
            ctx.txnToFirrtl[methodArg] = methodBlockArgs[j];
            break;
          }
        }
      }
      
      if (failed(convertBodyOps(actionMethod.getBody(), ctx))) {
        actionMethod.emitError("Failed to convert action method body");
        return;
      }
    });
  });
  
  // Convert rules
  for (StringRef ruleName : scheduleOrder) {
    txnModule.walk([&](RuleOp rule) {
      if (rule.getSymName() != ruleName) return;
      
      Value wf = ctx.willFireSignals[ruleName];
      if (!wf) return;
      
      // Execute rule body when will-fire is true
      ctx.firrtlBuilder.create<WhenOp>(rule.getLoc(), wf, false, [&]() {
        // Convert rule body
        ctx.txnToFirrtl.clear();
        if (failed(convertBodyOps(rule.getBody(), ctx))) {
          rule.emitError("Failed to convert rule body");
          return;
        }
      });
    });
  }
  
  // TODO: Add conflict_inside calculation in a future enhancement
  // For now, we assume no internal conflicts within actions
  
  return success();
}

/// Module dependency analysis to determine processing order
static LogicalResult analyzeModuleDependencies(
    ModuleOp topModule,
    SmallVectorImpl<::sharp::txn::ModuleOp> &sortedModules) {
  
  // Build dependency graph
  DenseMap<StringRef, SmallVector<StringRef>> dependencies;
  DenseMap<StringRef, ::sharp::txn::ModuleOp> moduleMap;
  
  topModule.walk([&](::sharp::txn::ModuleOp txnMod) {
    auto symName = txnMod.getSymName();
    if (!symName.empty()) {
      moduleMap[symName] = txnMod;
      auto &deps = dependencies[symName];
      
      // Find instance dependencies
      txnMod.walk([&](::sharp::txn::InstanceOp inst) {
        deps.push_back(inst.getModuleName());
      });
    }
  });
  
  // Topological sort
  DenseSet<StringRef> visited;
  DenseSet<StringRef> visiting;
  SmallVector<StringRef> sorted;
  
  std::function<LogicalResult(StringRef)> visit = [&](StringRef name) -> LogicalResult {
    if (visited.count(name)) return success();
    if (visiting.count(name)) {
      return topModule.emitError("Circular dependency detected involving module: ") << name;
    }
    
    visiting.insert(name);
    
    if (auto deps = dependencies.find(name); deps != dependencies.end()) {
      for (StringRef dep : deps->second) {
        if (failed(visit(dep))) return failure();
      }
    }
    
    visiting.erase(name);
    visited.insert(name);
    sorted.push_back(name);
    return success();
  };
  
  // Visit all modules
  for (auto &entry : moduleMap) {
    if (failed(visit(entry.first))) return failure();
  }
  
  // Build sorted module list
  for (StringRef name : sorted) {
    if (auto mod = moduleMap[name]) {
      sortedModules.push_back(mod);
    }
  }
  
  return success();
}

/// Main conversion pass
struct TxnToFIRRTLConversionPass 
    : public impl::TxnToFIRRTLConversionBase<TxnToFIRRTLConversionPass> {
  
  using TxnToFIRRTLConversionBase::TxnToFIRRTLConversionBase;
  
  void runOnOperation() override {
    auto module = getOperation();
    MLIRContext *ctx = &getContext();
    ConversionContext convCtx(ctx);
    
    // Get sorted module list
    SmallVector<::sharp::txn::ModuleOp> sortedModules;
    if (failed(analyzeModuleDependencies(module, sortedModules))) {
      signalPassFailure();
      return;
    }
    
    // Create a FIRRTL circuit to hold converted modules
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(module.getBody());
    
    // Find the top-level module name (first module in dependency order)
    StringRef topModuleName = "Top";
    if (!sortedModules.empty()) {
      topModuleName = sortedModules.back().getName();
    }
    
    auto firrtlCircuit = builder.create<::circt::firrtl::CircuitOp>(
        module.getLoc(), builder.getStringAttr(topModuleName));
    
    // Convert each module bottom-up
    for (auto txnModule : sortedModules) {
      // Only convert Txn modules, not the top-level builtin module
      if (!isa<::sharp::txn::ModuleOp>(txnModule)) continue;
      if (txnModule.getName() == "firrtl_generated") continue;
      
      if (failed(convertModule(txnModule, convCtx, firrtlCircuit))) {
        signalPassFailure();
        return;
      }
      ++numModulesConverted;
    }
    
    // Remove original Txn modules after successful conversion
    for (auto txnModule : sortedModules) {
      txnModule.erase();
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Converted " << numModulesConverted 
                           << " modules to FIRRTL\n");
  }
};

} // namespace


} // namespace sharp
} // namespace mlir