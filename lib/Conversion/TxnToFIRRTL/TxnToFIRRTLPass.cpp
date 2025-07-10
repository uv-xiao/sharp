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
#include "llvm/ADT/StringMap.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "sharp/Conversion/Passes.h"
#include "sharp/Dialect/Txn/TxnDialect.h"
#include "sharp/Dialect/Txn/TxnOps.h"
#include "sharp/Dialect/Txn/TxnTypes.h"
#include "sharp/Dialect/Txn/TxnPrimitives.h"
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
/// 
/// Following Sharp's execution model:
/// - Value methods are computed once per cycle (combinational logic)
/// - Only actions (rules and action methods) participate in scheduling
/// - Actions cannot call other actions in the same module
/// - Schedules must only contain actions, not value methods
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
  
  /// Reachability conditions for abort operations
  DenseMap<Operation*, Value> abortReachabilityConditions;
  
  /// Track instance ports for method call connections
  DenseMap<StringRef, DenseMap<StringRef, Value>> instancePorts;
  
  /// Current FIRRTL module being built
  FModuleOp currentFIRRTLModule;
  
  /// Current Txn module being converted
  ::sharp::txn::ModuleOp currentTxnModule;
  
  /// Builder positioned inside FIRRTL module
  OpBuilder firrtlBuilder;
  
  ConversionContext(MLIRContext *ctx) : firrtlBuilder(ctx) {}
};

/// Helper to convert Sharp types to FIRRTL types
static FIRRTLType convertType(Type type) {
  MLIRContext *ctx = type.getContext();
  
  // Handle integer types
  if (auto intType = dyn_cast<IntegerType>(type)) {
    // MLIR ui/i types -> FIRRTL uint
    // MLIR si types -> FIRRTL sint
    if (intType.isSigned()) {
      return SIntType::get(ctx, intType.getWidth());
    } else {
      // Treat signless integers as unsigned
      return UIntType::get(ctx, intType.getWidth());
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

/// Get conflict relation between two method calls (for dynamic mode)
static ConflictRelation getConflictRelationFromString(const std::string &method1, 
                                                     const std::string &method2,
                                                     const ConversionContext &ctx) {
  // Check both orderings
  std::string key1 = method1 + "," + method2;
  std::string key2 = method2 + "," + method1;
  
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

/// Helper to walk an action and track reachability conditions
static void walkWithPathConditions(
    Operation *op, Value pathCondition, ConversionContext &ctx,
    llvm::function_ref<void(Operation*, Value)> callback) {
  
  // Handle regions
  for (auto &region : op->getRegions()) {
    for (auto &block : region) {
      for (auto &nested : block) {
        if (auto ifOp = dyn_cast<IfOp>(&nested)) {
          // Get the FIRRTL condition
          Value cond = ctx.txnToFirrtl.lookup(ifOp.getCondition());
          if (!cond) {
            // If condition not converted yet, skip
            continue;
          }
          
          // Build path condition for then branch
          Value thenCond = pathCondition;
          if (pathCondition) {
            thenCond = ctx.firrtlBuilder.create<AndPrimOp>(ifOp.getLoc(), pathCondition, cond);
          } else {
            thenCond = cond;
          }
          
          // Walk then region
          for (auto &thenBlock : ifOp.getThenRegion()) {
            for (auto &thenOp : thenBlock) {
              walkWithPathConditions(&thenOp, thenCond, ctx, callback);
            }
          }
          
          // Build path condition for else branch if exists
          if (!ifOp.getElseRegion().empty()) {
            Value notCond = ctx.firrtlBuilder.create<NotPrimOp>(ifOp.getLoc(), cond);
            Value elseCond = pathCondition;
            if (pathCondition) {
              elseCond = ctx.firrtlBuilder.create<AndPrimOp>(ifOp.getLoc(), pathCondition, notCond);
            } else {
              elseCond = notCond;
            }
            
            // Walk else region
            for (auto &elseBlock : ifOp.getElseRegion()) {
              for (auto &elseOp : elseBlock) {
                walkWithPathConditions(&elseOp, elseCond, ctx, callback);
              }
            }
          }
        } else {
          // For non-if operations, pass through the path condition
          callback(&nested, pathCondition);
          // Recursively walk nested operations
          walkWithPathConditions(&nested, pathCondition, ctx, callback);
        }
      }
    }
  }
}

/// Calculate reach_abort for an action
/// reach_abort[action] = OR(reach(abort_i, action) for every abort_i in action) 
///                     || OR(reach(call_i, action) && reach_abort(method[call_i]) for every call_i in action)
static Value calculateReachAbort(Operation *action, ConversionContext &ctx, 
                               const DenseMap<StringRef, Operation*> &actionMap) {
  auto &builder = ctx.firrtlBuilder;
  auto loc = action->getLoc();
  auto boolType = IntType::get(builder.getContext(), false, 1);
  
  // Start with false (no abort)
  Value reachAbort = builder.create<ConstantOp>(loc, Type(boolType), 
                                                APSInt(APInt(1, 0), true));
  
  // Walk the action body with path conditions
  Region *bodyRegion = nullptr;
  if (auto rule = dyn_cast<RuleOp>(action)) {
    bodyRegion = &rule.getBody();
  } else if (auto method = dyn_cast<ActionMethodOp>(action)) {
    bodyRegion = &method.getBody();
  }
  
  if (!bodyRegion) return reachAbort;
  
  // Walk through the body tracking path conditions
  for (auto &block : *bodyRegion) {
    for (auto &op : block) {
      walkWithPathConditions(&op, nullptr, ctx, [&](Operation *nested, Value pathCond) {
        if (isa<::sharp::txn::AbortOp>(nested)) {
          // Found an abort - use its path condition
          Value abortReach = pathCond;
          if (!abortReach) {
            // No path condition means always reachable
            abortReach = builder.create<ConstantOp>(loc, Type(boolType), 
                                                   APSInt(APInt(1, 1), true));
          }
          // OR with existing reach_abort
          reachAbort = builder.create<OrPrimOp>(loc, reachAbort, abortReach);
        } else if (auto callOp = dyn_cast<CallOp>(nested)) {
          // For method calls, use path condition combined with the call's own condition
          Value callReach = pathCond;
          
          // Check if ReachabilityAnalysis provided a condition 
          auto condIt = ctx.reachabilityConditions.find(nested);
          if (condIt != ctx.reachabilityConditions.end()) {
            Value analysisCondition = ctx.txnToFirrtl.lookup(condIt->second);
            if (analysisCondition) {
              if (callReach) {
                callReach = builder.create<AndPrimOp>(loc, callReach, analysisCondition);
              } else {
                callReach = analysisCondition;
              }
            }
          } else if (callOp.getCondition()) {
            // Fall back to the condition operand if present
            Value callCond = ctx.txnToFirrtl.lookup(callOp.getCondition());
            if (callCond) {
              if (callReach) {
                callReach = builder.create<AndPrimOp>(loc, callReach, callCond);
              } else {
                callReach = callCond;
              }
            }
          }
          
          if (!callReach) {
            // Default: always reachable
            callReach = builder.create<ConstantOp>(loc, Type(boolType), 
                                                  APSInt(APInt(1, 1), true));
          }
          
          // Check if the called method can abort
          // For instance method calls, check the method type
          if (callOp.getCallee().getNestedReferences().size() == 1) {
            StringRef methodName = callOp.getCallee().getNestedReferences()[0].getValue();
            
            // For primitive methods, we know their abort behavior
            // Register/Wire read/write methods don't abort
            // FIFO dequeue can abort if empty, enqueue can abort if full
            // TODO: In a full implementation, we'd check the instance type and method
            
            // For now, assume primitive methods don't abort except FIFO operations
            bool methodCanAbort = false;
            if (methodName == "dequeue" || methodName == "enqueue") {
              // FIFO operations can abort
              methodCanAbort = true;
            }
            
            if (methodCanAbort) {
              // Create abort condition: callReach && method_aborts
              // For FIFO, the abort condition depends on empty/full status
              // This is a simplification - in reality we'd need the FIFO's status signals
              Value methodAbortCond = builder.create<ConstantOp>(loc, Type(boolType), 
                                                                APSInt(APInt(1, 0), true)); // TODO: Get actual abort condition
              Value callAborts = builder.create<AndPrimOp>(loc, callReach, methodAbortCond);
              reachAbort = builder.create<OrPrimOp>(loc, reachAbort, callAborts);
            }
          } else {
            // For local method calls, we'd need to recursively analyze the called method
            // This requires inter-procedural analysis which is complex
            // For now, conservatively assume local action methods might abort
            StringRef methodName = callOp.getCallee().getRootReference().getValue();
            
            // Look up if this is an action method
            bool isActionMethod = false;
            ctx.currentTxnModule.walk([&](ActionMethodOp am) {
              if (am.getSymName() == methodName) {
                isActionMethod = true;
              }
            });
            
            if (isActionMethod) {
              // Conservatively assume action methods can abort
              // In reality, we'd analyze the method body
              Value methodAbortCond = builder.create<ConstantOp>(loc, Type(boolType), 
                                                                APSInt(APInt(1, 1), true)); // Conservative: assume can abort
              Value callAborts = builder.create<AndPrimOp>(loc, callReach, methodAbortCond);
              reachAbort = builder.create<OrPrimOp>(loc, reachAbort, callAborts);
            }
          }
        }
      });
    }
  }
  
  return reachAbort;
}

/// Get conflict relation between two actions by examining their method calls
static ConflictRelation getConflictRelation(StringRef a1, StringRef a2,
                                           const ConversionContext &ctx,
                                           const DenseMap<StringRef, Operation*> &actionMap) {
  // First check if there's a direct action-to-action conflict in the matrix
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
  
  // Infer conflict from method calls: if action a1 calls method m1 and action a2 calls method m2,
  // and m1 conflicts with m2, then a1 conflicts with a2
  auto it_a1 = actionMap.find(a1);
  auto it_a2 = actionMap.find(a2);
  if (it_a1 == actionMap.end() || it_a2 == actionMap.end()) {
    return ConflictRelation::ConflictFree;
  }
  
  Operation *action1 = it_a1->second;
  Operation *action2 = it_a2->second;
  
  // Collect method calls from both actions
  SmallVector<std::string> methods1, methods2;
  action1->walk([&](CallOp call) {
    auto callee = call.getCallee();
    if (callee.getNestedReferences().size() == 1) {
      // Build full method name: instance::method
      StringRef inst = callee.getRootReference().getValue();
      StringRef method = callee.getNestedReferences()[0].getValue();
      std::string fullName = (inst + "::" + method).str();
      methods1.push_back(fullName);
    }
  });
  action2->walk([&](CallOp call) {
    auto callee = call.getCallee();
    if (callee.getNestedReferences().size() == 1) {
      // Build full method name: instance::method
      StringRef inst = callee.getRootReference().getValue();
      StringRef method = callee.getNestedReferences()[0].getValue();
      std::string fullName = (inst + "::" + method).str();
      methods2.push_back(fullName);
    }
  });
  
  // Check for conflicts between any method pairs
  ConflictRelation maxConflict = ConflictRelation::ConflictFree;
  for (const std::string &m1 : methods1) {
    for (const std::string &m2 : methods2) {
      auto rel = getConflictRelationFromString(m1, m2, ctx);
      // Prioritize conflicts: C > SA/SB > CF
      if (rel == ConflictRelation::Conflict) {
        return ConflictRelation::Conflict; // Highest priority
      } else if (rel == ConflictRelation::SequenceBefore || rel == ConflictRelation::SequenceAfter) {
        maxConflict = rel; // Remember sequence constraints
      }
    }
  }
  
  return maxConflict;
}

/// Generate static mode will-fire logic for an action
/// wf[action] = enabled[action] && !reach_abort[action] && !conflicts_with_earlier[action] && !conflict_inside[action]
static Value generateStaticWillFire(StringRef actionName, Value enabled,
                                  ArrayRef<StringRef> schedule,
                                  ConversionContext &ctx,
                                  const DenseMap<StringRef, Operation*> &actionMap) {
  auto &builder = ctx.firrtlBuilder;
  auto loc = builder.getUnknownLoc();
  
  // Start with enabled signal
  Value wf = enabled;
  
  if (!wf) {
    llvm::errs() << "generateStaticWillFire: enabled signal is null for " << actionName << "\n";
    return nullptr;
  }
  
  // Calculate reach_abort for this action
  auto actionIt = actionMap.find(actionName);
  if (actionIt != actionMap.end()) {
    Value reachAbort = calculateReachAbort(actionIt->second, ctx, actionMap);
    if (reachAbort) {
      // wf = enabled && !reach_abort
      Value notAbort = builder.create<NotPrimOp>(loc, reachAbort);
      wf = builder.create<AndPrimOp>(loc, wf, notAbort);
    }
  }
  
  // Check conflicts with earlier actions (static mode)
  // conflicts_with_earlier[a] = OR(wf[a1] && conflict(a1, a) for all a1 scheduled before a)
  for (StringRef earlier : schedule) {
    if (earlier == actionName) break;
    
    auto rel = getConflictRelation(earlier, actionName, ctx, actionMap);
    
    // Generate conflict check if needed
    // conflicts(a1, a2) = (CM[a1,a2] == C) || (CM[a1,a2] == SA && wf[a1])
    if (rel == ConflictRelation::Conflict) {
      auto wfIt = ctx.willFireSignals.find(earlier);
      if (wfIt == ctx.willFireSignals.end()) {
        // Earlier action hasn't been processed yet or is a value method
        llvm::errs() << "Warning: no will-fire signal found for earlier action " << earlier << "\n";
        continue;
      }
      Value earlierWF = wfIt->second;
      
      // Create: wf = wf & !earlier_wf (static conflict)
      auto notEarlier = builder.create<NotPrimOp>(loc, earlierWF);
      wf = builder.create<AndPrimOp>(loc, wf, notEarlier);
    } else if (rel == ConflictRelation::SequenceBefore) {
      // SA: a1 must happen after a2, but a1 has already happened
      auto wfIt = ctx.willFireSignals.find(earlier);
      if (wfIt == ctx.willFireSignals.end()) {
        llvm::errs() << "Warning: no will-fire signal found for earlier action " << earlier << "\n";
        continue;
      }
      Value earlierWF = wfIt->second;
      
      // Create: wf = wf & !earlier_wf (sequence violation)
      auto notEarlier = builder.create<NotPrimOp>(loc, earlierWF);
      wf = builder.create<AndPrimOp>(loc, wf, notEarlier);
    }
  }
  
  // conflict_inside is already handled in the main logic
  
  // Return the will-fire value - node creation is handled in the main loop
  return wf;
}

/// Generate dynamic mode will-fire logic for an action
/// wf[action] = enabled[action] && !reach_abort[action] && AND{for every m in action, NOT(reach(m, action) && conflict_with_earlier(m))}
static Value generateDynamicWillFire(StringRef actionName, Value enabled,
                                   ArrayRef<StringRef> schedule,
                                   ConversionContext &ctx,
                                   Operation *action,
                                   const DenseMap<StringRef, Operation*> &actionMap) {
  auto &builder = ctx.firrtlBuilder;
  auto loc = builder.getUnknownLoc();
  auto intType = IntType::get(builder.getContext(), false, 1);
  
  // Start with enabled signal
  Value wf = enabled;
  
  if (!wf) {
    llvm::errs() << "generateDynamicWillFire: enabled signal is null for " << actionName << "\n";
    return nullptr;
  }
  
  // Calculate reach_abort for this action
  Value reachAbort = calculateReachAbort(action, ctx, actionMap);
  if (reachAbort) {
    // wf = enabled && !reach_abort
    Value notAbort = builder.create<NotPrimOp>(loc, reachAbort);
    wf = builder.create<AndPrimOp>(loc, wf, notAbort);
  }
  
  // Track method calls that have been made by earlier actions
  // method_called[M] = OR{wf[a] && OR(reach(m, action) for m in M) for every earlier action}
  llvm::StringMap<Value> methodCalled;
  
  // Analyze earlier actions to see what methods they might call
  for (StringRef earlierName : schedule) {
    if (earlierName == actionName) break;
    
    Value earlierWF = ctx.willFireSignals[earlierName];
    if (!earlierWF) continue;
    
    // Find the earlier action operation from actionMap
    auto it = actionMap.find(earlierName);
    if (it == actionMap.end()) continue;
    Operation *earlierAction = it->second;
    
    // Collect method calls from earlier action
    earlierAction->walk([&](CallOp call) {
      StringRef methodKey = call.getCallee().getRootReference().getValue();
      
      // Get reachability condition for this call
      Value condition;
      if (call.getCondition()) {
        condition = ctx.txnToFirrtl.lookup(call.getCondition());
        if (!condition) {
          condition = builder.create<ConstantOp>(loc, Type(intType), 
                                                APSInt(APInt(1, 1), true));
        }
      } else {
        condition = builder.create<ConstantOp>(loc, Type(intType), 
                                              APSInt(APInt(1, 1), true));
      }
      
      // method_called[M] |= wf[earlier] && reach(m, earlier)
      Value callCondition = builder.create<AndPrimOp>(loc, earlierWF, condition);
      if (methodCalled.count(methodKey)) {
        methodCalled[methodKey] = builder.create<OrPrimOp>(loc, methodCalled[methodKey], callCondition);
      } else {
        methodCalled[methodKey] = callCondition;
      }
    });
  }
  
  // For each method call in current action, check for conflicts with earlier calls
  action->walk([&](CallOp call) {
    StringRef methodKey = call.getCallee().getRootReference().getValue();
    
    // Get reachability condition for this call
    Value condition;
    if (call.getCondition()) {
      condition = ctx.txnToFirrtl.lookup(call.getCondition());
      if (!condition) {
        condition = builder.create<ConstantOp>(loc, Type(intType), 
                                              APSInt(APInt(1, 1), true));
      }
    } else {
      condition = builder.create<ConstantOp>(loc, Type(intType), 
                                            APSInt(APInt(1, 1), true));
    }
    
    // Check for conflicts with any earlier method calls
    for (auto &[earlierMethodKey, earlierCalled] : methodCalled) {
      // Check if these methods conflict
      auto rel = getConflictRelationFromString(earlierMethodKey.str(), methodKey.str(), ctx);
      
      if (rel == ConflictRelation::Conflict) {
        // conflict_with_earlier(m) = method_called[M'] && conflict(M', M)
        Value conflictCondition = builder.create<AndPrimOp>(loc, earlierCalled, condition);
        
        // wf = wf && !(reach(m) && conflict_with_earlier(m))
        auto notConflict = builder.create<NotPrimOp>(loc, conflictCondition);
        wf = builder.create<AndPrimOp>(loc, wf, notConflict);
      }
    }
  });
  
  // conflict_inside is already handled in the main logic
  
  // Return the will-fire value - node creation is handled in the main loop
  return wf;
}

/// Generate most-dynamic mode will-fire logic for an action
/// wf[action] = enabled[action] && !reach_abort[action] && 
///              AND{for every direct/indirect method call m by action: NOT(reach(m, action) && conflict_with_earlier(m))}
static Value generateMostDynamicWillFire(StringRef actionName, Value enabled,
                                       ArrayRef<StringRef> schedule,
                                       ConversionContext &ctx,
                                       Operation *action,
                                       const DenseMap<StringRef, Operation*> &actionMap) {
  auto &builder = ctx.firrtlBuilder;
  auto loc = builder.getUnknownLoc();
  
  // Start with enabled signal
  Value wf = enabled;
  
  if (!wf) {
    llvm::errs() << "generateMostDynamicWillFire: enabled signal is null for " << actionName << "\n";
    return nullptr;
  }
  
  // Calculate reach_abort for this action
  Value reachAbort = calculateReachAbort(action, ctx, actionMap);
  if (reachAbort) {
    // wf = enabled && !reach_abort
    Value notAbort = builder.create<NotPrimOp>(loc, reachAbort);
    wf = builder.create<AndPrimOp>(loc, wf, notAbort);
  }
  
  // Track primitive actions that have been called by earlier actions
  llvm::StringMap<Value> primitiveCalled;
  
  // Get primitive calls for current action from the attribute
  SmallVector<StringRef> currentPrimitiveCalls;
  if (auto attr = action->getAttrOfType<ArrayAttr>("primitive_calls")) {
    for (auto elem : attr) {
      if (auto strAttr = dyn_cast<StringAttr>(elem)) {
        currentPrimitiveCalls.push_back(strAttr.getValue());
      }
    }
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Most-dynamic: Action " << actionName 
             << " has primitive calls: ";
             for (auto call : currentPrimitiveCalls) {
               llvm::dbgs() << call << " ";
             }
             llvm::dbgs() << "\n");
  
  // Analyze earlier actions to see what primitive actions they might call
  for (StringRef earlierName : schedule) {
    if (earlierName == actionName) break;
    
    Value earlierWF = ctx.willFireSignals[earlierName];
    if (!earlierWF) {
      LLVM_DEBUG(llvm::dbgs() << "Most-dynamic: No will-fire signal for earlier action " << earlierName << "\n");
      continue;
    }
    
    // Find the earlier action operation
    auto it = actionMap.find(earlierName);
    if (it == actionMap.end()) continue;
    Operation *earlierAction = it->second;
    
    // Get primitive calls from the earlier action's attribute
    if (auto attr = earlierAction->getAttrOfType<ArrayAttr>("primitive_calls")) {
      LLVM_DEBUG(llvm::dbgs() << "Most-dynamic: Earlier action " << earlierName 
                 << " has primitive calls\n");
      for (auto elem : attr) {
        if (auto strAttr = dyn_cast<StringAttr>(elem)) {
          StringRef primitiveCall = strAttr.getValue();
          LLVM_DEBUG(llvm::dbgs() << "  - " << primitiveCall << "\n");
          
          // Mark this primitive as called
          if (primitiveCalled.count(primitiveCall) == 0) {
            primitiveCalled[primitiveCall] = earlierWF;
          } else {
            // OR with existing signal
            Value existing = primitiveCalled[primitiveCall];
            primitiveCalled[primitiveCall] = builder.create<OrPrimOp>(loc, existing, earlierWF);
          }
        }
      }
    }
  }
  
  // Check if any of current action's primitive calls conflict with earlier ones
  for (StringRef primitiveCall : currentPrimitiveCalls) {
    // Parse the primitive call format: instance::method
    // For nested paths like counter1::reg::write, we need the last :: split
    size_t lastDoubleColon = primitiveCall.rfind("::");
    if (lastDoubleColon == StringRef::npos) continue;
    
    StringRef instancePath = primitiveCall.substr(0, lastDoubleColon);
    StringRef methodName = primitiveCall.substr(lastDoubleColon + 2);
    
    // Check all earlier primitive calls for conflicts
    for (const auto &entry : primitiveCalled) {
      StringRef earlierCall = entry.first();
      Value earlierCalled = entry.second;
      
      // Parse the earlier call format: instance::method
      // For nested paths like counter1::reg::write, we need the last :: split
      size_t earlierLastDoubleColon = earlierCall.rfind("::");
      if (earlierLastDoubleColon == StringRef::npos) continue;
      
      StringRef earlierInstance = earlierCall.substr(0, earlierLastDoubleColon);
      StringRef earlierMethod = earlierCall.substr(earlierLastDoubleColon + 2);
      
      // Check if same instance
      if (instancePath == earlierInstance) {
        // Get conflict relation between methods
        // For primitives, we need to check their conflict matrix
        // This is a simplified check - in reality we'd need the primitive's conflict matrix
        ConflictRelation conflict = ConflictRelation::ConflictFree;
        
        LLVM_DEBUG(llvm::dbgs() << "Most-dynamic: Checking conflict between " 
                   << instancePath << "::" << methodName << " and " 
                   << earlierInstance << "::" << earlierMethod << "\n");
        LLVM_DEBUG(llvm::dbgs() << "  Full primitive calls: " << primitiveCall << " vs " << earlierCall << "\n");
        
        // Common primitive conflicts:
        // Register: read CF write, write C write
        // Wire: read SA write, write C write
        // FIFO: enqueue C dequeue, enqueue C enqueue, dequeue C dequeue
        
        // Check primitive type by looking at the instance path
        // The instance might be named "reg", "wire", etc., or the path might contain the type
        bool isRegister = instancePath.ends_with("reg") || instancePath.contains("Register");
        bool isWire = instancePath.ends_with("wire") || instancePath.contains("Wire");
        bool isFIFO = instancePath.ends_with("fifo") || instancePath.contains("FIFO");
        
        LLVM_DEBUG(llvm::dbgs() << "  Instance path: " << instancePath 
                   << " isRegister: " << isRegister 
                   << " methodName: " << methodName 
                   << " earlierMethod: " << earlierMethod << "\n");
        
        if (isRegister) {
          LLVM_DEBUG(llvm::dbgs() << "  Detected Register primitive\n");
          if (methodName == "write" && earlierMethod == "write") {
            conflict = ConflictRelation::Conflict;
          }
        } else if (isWire) {
          if (methodName == "read" && earlierMethod == "write") {
            conflict = ConflictRelation::SequenceAfter;
          } else if (methodName == "write" && earlierMethod == "write") {
            conflict = ConflictRelation::Conflict;
          }
        } else if (isFIFO) {
          if ((methodName == "enqueue" && earlierMethod == "dequeue") ||
              (methodName == "dequeue" && earlierMethod == "enqueue") ||
              (methodName == "enqueue" && earlierMethod == "enqueue") ||
              (methodName == "dequeue" && earlierMethod == "dequeue")) {
            conflict = ConflictRelation::Conflict;
          }
        }
        
        // If there's a conflict, block this action
        if (conflict != ConflictRelation::ConflictFree) {
          LLVM_DEBUG(llvm::dbgs() << "Most-dynamic: Found conflict! Blocking " 
                     << actionName << " when primitive " << earlierCall << " is called\n");
          LLVM_DEBUG(llvm::dbgs() << "  - primitiveCall: " << primitiveCall << "\n");
          LLVM_DEBUG(llvm::dbgs() << "  - earlierCall: " << earlierCall << "\n");
          LLVM_DEBUG(llvm::dbgs() << "  - conflict type: " << static_cast<int>(conflict) << "\n");
          Value notEarlierCalled = builder.create<NotPrimOp>(loc, earlierCalled);
          wf = builder.create<AndPrimOp>(loc, wf, notEarlierCalled);
          LLVM_DEBUG(llvm::dbgs() << "  - updated wf: " << wf << "\n");
        }
      }
    }
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Most-dynamic: Final wf for " << actionName << ": " << wf << "\n");
  return wf;
}

/// Generate will-fire logic for an action (dispatcher for static/dynamic/most-dynamic modes)
static Value generateWillFire(StringRef actionName, Value enabled,
                            ArrayRef<StringRef> schedule,
                            ConversionContext &ctx,
                            const std::string &willFireMode = "static",
                            Operation *action = nullptr,
                            const DenseMap<StringRef, Operation*> *actionMap = nullptr) {
  if (!actionMap) {
    llvm::errs() << "generateWillFire: actionMap is required for all modes\n";
    return nullptr;
  }
  
  if (willFireMode == "dynamic") {
    if (!action) {
      llvm::errs() << "generateWillFire: dynamic mode requires action operation\n";
      return generateStaticWillFire(actionName, enabled, schedule, ctx, *actionMap);
    }
    return generateDynamicWillFire(actionName, enabled, schedule, ctx, action, *actionMap);
  } else if (willFireMode == "most-dynamic") {
    if (!action) {
      llvm::errs() << "generateWillFire: most-dynamic mode requires action operation\n";
      return generateStaticWillFire(actionName, enabled, schedule, ctx, *actionMap);
    }
    return generateMostDynamicWillFire(actionName, enabled, schedule, ctx, action, *actionMap);
  } else {
    return generateStaticWillFire(actionName, enabled, schedule, ctx, *actionMap);
  }
}

// Forward declaration
static LogicalResult convertBodyOps(Region &region, ConversionContext &ctx);
static LogicalResult convertOp(Operation *op, ConversionContext &ctx);

// Note: This function is currently unused but kept for potential future use
// when implementing more sophisticated reachability analysis
#if 0
/// Build reachability conditions for method calls in an action's body
static void buildReachabilityConditions(Region &region, Value currentCond, 
                                       ConversionContext &ctx,
                                       DenseMap<CallOp, Value> &reachMap) {
  for (auto &op : region.front()) {
    if (auto ifOp = dyn_cast<IfOp>(&op)) {
      // Convert the condition
      Value cond = ifOp.getCondition();
      if (ctx.txnToFirrtl.count(cond)) {
        cond = ctx.txnToFirrtl[cond];
      } else {
        // Convert the condition if not already converted
        if (auto cmpOp = cond.getDefiningOp<arith::CmpIOp>()) {
          Value lhs = ctx.txnToFirrtl.lookup(cmpOp.getLhs());
          Value rhs = ctx.txnToFirrtl.lookup(cmpOp.getRhs());
          if (lhs && rhs) {
            switch (cmpOp.getPredicate()) {
              case arith::CmpIPredicate::eq:
                cond = ctx.firrtlBuilder.create<EQPrimOp>(cmpOp.getLoc(), lhs, rhs);
                break;
              case arith::CmpIPredicate::ne:
                cond = ctx.firrtlBuilder.create<NEQPrimOp>(cmpOp.getLoc(), lhs, rhs);
                break;
              case arith::CmpIPredicate::slt:
              case arith::CmpIPredicate::ult:
                cond = ctx.firrtlBuilder.create<LTPrimOp>(cmpOp.getLoc(), lhs, rhs);
                break;
              case arith::CmpIPredicate::sle:
              case arith::CmpIPredicate::ule:
                cond = ctx.firrtlBuilder.create<LEQPrimOp>(cmpOp.getLoc(), lhs, rhs);
                break;
              case arith::CmpIPredicate::sgt:
              case arith::CmpIPredicate::ugt:
                cond = ctx.firrtlBuilder.create<GTPrimOp>(cmpOp.getLoc(), lhs, rhs);
                break;
              case arith::CmpIPredicate::sge:
              case arith::CmpIPredicate::uge:
                cond = ctx.firrtlBuilder.create<GEQPrimOp>(cmpOp.getLoc(), lhs, rhs);
                break;
            }
            ctx.txnToFirrtl[ifOp.getCondition()] = cond;
          }
        }
      }
      
      // Process then branch
      Value thenCond = ctx.firrtlBuilder.create<AndPrimOp>(
          ifOp.getLoc(), currentCond, cond);
      buildReachabilityConditions(ifOp.getThenRegion(), thenCond, ctx, reachMap);
      
      // Process else branch if exists
      if (!ifOp.getElseRegion().empty()) {
        Value notCond = ctx.firrtlBuilder.create<NotPrimOp>(ifOp.getLoc(), cond);
        Value elseCond = ctx.firrtlBuilder.create<AndPrimOp>(
            ifOp.getLoc(), currentCond, notCond);
        buildReachabilityConditions(ifOp.getElseRegion(), elseCond, ctx, reachMap);
      }
    } else if (auto callOp = dyn_cast<CallOp>(&op)) {
      // Record reachability condition for this call
      reachMap[callOp] = currentCond;
    }
  }
}
#endif

/// Check if action has potential internal conflicts (simple check)
static bool hasConflictingCalls(Operation *action, ConversionContext &ctx) {
  // Collect all method calls
  SmallVector<CallOp> methodCalls;
  action->walk([&](CallOp call) {
    methodCalls.push_back(call);
  });
  
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
      
      // Build the conflict key - check both orderings
      std::string key1 = (inst1 + "::" + method1 + "," + inst2 + "::" + method2).str();
      std::string key2 = (inst2 + "::" + method2 + "," + inst1 + "::" + method1).str();
      
      // Check if methods conflict
      auto it1 = ctx.conflictMatrix.find(key1);
      auto it2 = ctx.conflictMatrix.find(key2);
      
      if (it1 != ctx.conflictMatrix.end()) {
        auto rel = static_cast<ConflictRelation>(it1->second);
        if (rel == ConflictRelation::Conflict) return true;
      } else if (it2 != ctx.conflictMatrix.end()) {
        auto rel = static_cast<ConflictRelation>(it2->second);
        if (rel == ConflictRelation::Conflict) return true;
      }
    }
  }
  
  return false;
}

/// Check if a module is a known primitive (Register, Wire, etc.)
static bool isKnownPrimitive(StringRef moduleName) {
  return moduleName == "Register" || moduleName == "Wire" || 
         moduleName == "FIFO" || moduleName == "Memory" ||
         moduleName == "SpecFIFO" || moduleName == "SpecMemory";
}

/// Get the data type for a parametric primitive instance
static Type getInstanceDataType(::sharp::txn::InstanceOp inst) {
  // Check if the instance has type arguments
  auto typeArgsOpt = inst.getTypeArguments();
  if (typeArgsOpt && !typeArgsOpt.value().empty()) {
    auto typeArgs = typeArgsOpt.value();
    // For now, assume single type parameter
    if (auto typeAttr = dyn_cast<TypeAttr>(typeArgs[0]))
      return typeAttr.getValue();
  }
  
  // If no type arguments, try to infer from primitive definition
  auto moduleName = inst.getModuleName();
  if (auto primitiveOp = inst->getParentOfType<mlir::ModuleOp>().lookupSymbol<::sharp::txn::PrimitiveOp>(moduleName)) {
    // Look for the first method to infer type
    for (auto& op : primitiveOp.getBody().front()) {
      if (auto methodOp = dyn_cast<::sharp::txn::FirValueMethodOp>(op)) {
        auto methodType = methodOp.getFunctionType();
        if (methodType.getNumResults() > 0) {
          return methodType.getResult(0);
        }
      } else if (auto methodOp = dyn_cast<::sharp::txn::FirActionMethodOp>(op)) {
        auto methodType = methodOp.getFunctionType();
        if (methodType.getNumInputs() > 0) {
          return methodType.getInput(0);
        }
      }
    }
  }
    
  return nullptr;
}

/// Create a primitive FIRRTL module if it doesn't exist
static circt::firrtl::FModuleOp getOrCreatePrimitiveFIRRTLModule(
    StringRef primitiveType,
    Type dataType,
    ::circt::firrtl::CircuitOp circuit,
    OpBuilder &builder) {
  
  // Generate a unique module name based on primitive type and data type
  std::string baseName;
  llvm::raw_string_ostream nameStream(baseName);
  nameStream << primitiveType << "_";
  dataType.print(nameStream);
  nameStream.flush();
  // Replace invalid characters in module name
  std::replace(baseName.begin(), baseName.end(), '<', '_');
  std::replace(baseName.begin(), baseName.end(), '>', '_');
  
  // The primitive constructors add "_impl" suffix, so check for that name
  std::string moduleName = baseName + "_impl";
  
  // Check if module already exists
  circt::firrtl::FModuleOp existingModule;
  circuit.walk([&](circt::firrtl::FModuleOp fmodule) {
    if (fmodule.getName() == moduleName) {
      existingModule = fmodule;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  if (existingModule) {
    return existingModule;
  }
  
  // Create the primitive module
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(circuit.getBodyBlock());
  
  if (primitiveType == "Register") {
    return ::mlir::sharp::txn::createRegisterFIRRTLModule(builder, builder.getUnknownLoc(), baseName, dataType);
  } else if (primitiveType == "Wire") {
    return ::mlir::sharp::txn::createWireFIRRTLModule(builder, builder.getUnknownLoc(), baseName, dataType);
  } else if (primitiveType == "FIFO") {
    // Use default depth of 16 for FIFOs
    return ::mlir::sharp::txn::createFIFOFIRRTLModule(builder, builder.getUnknownLoc(), baseName, dataType, 16);
  } else if (primitiveType == "Memory") {
    // Use default address width of 32 bits for Memory
    return ::mlir::sharp::txn::createMemoryFIRRTLModule(builder, builder.getUnknownLoc(), baseName, dataType, 32);
  }
  // TODO: Add support for SpecFIFO, SpecMemory when their create functions are implemented
  
  return nullptr;
}

/// Convert method/rule body operations to FIRRTL
static LogicalResult convertBodyOps(Region &region, ConversionContext &ctx) {
  // Process all blocks in the region
  for (auto &block : region) {
    // First pass: convert all non-control-flow operations
    for (auto &op : block.getOperations()) {
      // Skip control flow operations in the first pass
      if (isa<IfOp, YieldOp, ReturnOp, AbortOp>(op)) {
        continue;
      }
      
      // Convert the operation
      if (failed(convertOp(&op, ctx))) {
        op.emitError("Failed to convert operation");
        return failure();
      }
    }
    
    // Second pass: handle control flow operations
    for (auto &op : block.getOperations()) {
    if (auto ifOp = dyn_cast<IfOp>(&op)) {
      // The condition should now be converted
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
      
      // If the call has a condition, ensure it's mapped to FIRRTL
      if (callOp.getCondition()) {
        Value condition = callOp.getCondition();
        if (!ctx.txnToFirrtl.count(condition)) {
          // The condition should already be converted, but if not, map it
          ctx.txnToFirrtl[condition] = condition;
        }
      }
      
      if (callee.getNestedReferences().size() == 0) {
        // Local method call (within same module)
        StringRef methodName = callee.getRootReference().getValue();
        
        // Check if this violates execution model constraints
        if (ctx.currentTxnModule) {
          // Check if the caller is an action
          Operation *caller = callOp->getParentOp();
          while (caller && !isa<ActionMethodOp>(caller) && !isa<RuleOp>(caller) && 
                 !isa<ValueMethodOp>(caller)) {
            caller = caller->getParentOp();
          }
          
          bool isCallerAction = caller && (isa<ActionMethodOp>(caller) || isa<RuleOp>(caller));
          
          // Check if the callee is an action
          bool isCalleeAction = false;
          ctx.currentTxnModule.walk([&](Operation *op) {
            if (auto actionMethod = dyn_cast<ActionMethodOp>(op)) {
              if (actionMethod.getSymName() == methodName) {
                isCalleeAction = true;
              }
            } else if (auto rule = dyn_cast<RuleOp>(op)) {
              if (rule.getSymName() == methodName) {
                isCalleeAction = true;
              }
            }
          });
          
          if (isCallerAction && isCalleeAction) {
            return callOp.emitError("Action cannot call another action '") << methodName 
                   << "' in the same module. Actions can only call value methods in the same module "
                   << "or methods of child module instances.";
          }
        }
        
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
            for (size_t i = 0; i < callOp.getArgs().size(); ++i) {
              auto argValue = ctx.txnToFirrtl.lookup(callOp.getArgs()[i]);
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
        if (callOp.getArgs().size() > 0 && methodName == "write") {
          // Convert the argument
          Value arg = callOp.getArgs()[0];
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
        auto lhsType = dyn_cast<IntType>(lhs.getType());
        auto rhsType = dyn_cast<IntType>(rhs.getType());

        // Unify signedness of operands before the operation.
        if (lhsType && rhsType && lhsType.isSigned() != rhsType.isSigned()) {
          // The result type of addi dictates the target signedness.
          auto resultType = convertType(addOp.getType());
          bool targetIsSigned = isa<SIntType>(resultType);

          if (targetIsSigned) {
            if (!isa<SIntType>(lhsType))
              lhs = ctx.firrtlBuilder.create<AsSIntPrimOp>(addOp.getLoc(), lhs);
            if (!isa<SIntType>(rhsType))
              rhs = ctx.firrtlBuilder.create<AsSIntPrimOp>(addOp.getLoc(), rhs);
          } else { // target is unsigned
            if (isa<SIntType>(lhsType))
              lhs = ctx.firrtlBuilder.create<AsUIntPrimOp>(addOp.getLoc(), lhs);
            if (isa<SIntType>(rhsType))
              rhs = ctx.firrtlBuilder.create<AsUIntPrimOp>(addOp.getLoc(), rhs);
          }
        }

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
    } else if (auto andOp = dyn_cast<arith::AndIOp>(&op)) {
      // Convert bitwise AND operation
      Value lhs = ctx.txnToFirrtl.lookup(andOp.getLhs());
      Value rhs = ctx.txnToFirrtl.lookup(andOp.getRhs());
      if (lhs && rhs) {
        auto result = ctx.firrtlBuilder.create<AndPrimOp>(andOp.getLoc(), lhs, rhs);
        ctx.txnToFirrtl[andOp.getResult()] = result;
      }
    } else if (auto xorOp = dyn_cast<arith::XOrIOp>(&op)) {
      // Convert bitwise XOR operation
      Value lhs = ctx.txnToFirrtl.lookup(xorOp.getLhs());
      Value rhs = ctx.txnToFirrtl.lookup(xorOp.getRhs());
      if (lhs && rhs) {
        auto result = ctx.firrtlBuilder.create<XorPrimOp>(xorOp.getLoc(), lhs, rhs);
        ctx.txnToFirrtl[xorOp.getResult()] = result;
      }
    } else if (auto futureOp = dyn_cast<FutureOp>(&op)) {
      // Future operations contain launch operations
      // Just convert the body - the launch operations inside will create the timing logic
      if (failed(convertBodyOps(futureOp.getBody(), ctx))) {
        return futureOp.emitError("failed to convert future body");
      }
    } else if (auto launchOp = dyn_cast<LaunchOp>(&op)) {
      // Convert launch operation to FIRRTL
      // Launch operations represent multi-cycle execution
      auto boolType = IntType::get(ctx.firrtlBuilder.getContext(), false, 1);
      
      // First, convert the body of the launch operation
      // The body contains the operations that execute during the multi-cycle period
      if (failed(convertBodyOps(launchOp.getBody(), ctx))) {
        return launchOp.emitError("failed to convert launch body");
      }
      
      // Create a wire to represent the launch done signal
      std::string doneName = "launch_done_" + std::to_string(reinterpret_cast<uintptr_t>(&op));
      auto doneWire = ctx.firrtlBuilder.create<WireOp>(
          launchOp.getLoc(), boolType, 
          ctx.firrtlBuilder.getStringAttr(doneName));
      
      // For static timing (after N cycles), we would need to create a counter
      // For dynamic timing (until condition), we would check the condition
      // For now, just connect it to true as a placeholder
      auto trueVal = ctx.firrtlBuilder.create<ConstantOp>(
          launchOp.getLoc(), Type(boolType), APSInt(APInt(1, 1)));
      ctx.firrtlBuilder.create<ConnectOp>(launchOp.getLoc(), 
                                         doneWire.getResult(), trueVal);
      
      // Map the launch result to the done wire
      ctx.txnToFirrtl[launchOp.getResult()] = doneWire.getResult();
    }
    // Add more operation conversions as needed
  }
  }
  return success();
}

/// Convert a single operation to FIRRTL
static LogicalResult convertOp(Operation *op, ConversionContext &ctx) {
  // Handle dependency operations first
  for (Value operand : op->getOperands()) {
    if (ctx.txnToFirrtl.lookup(operand)) {
      // Already converted
      continue;
    }
    
    // Find the defining operation
    if (auto defOp = operand.getDefiningOp()) {
      // Recursively convert the defining operation
      if (failed(convertOp(defOp, ctx))) {
        return failure();
      }
    }
  }
  
  // Now convert this operation
  if (auto callOp = dyn_cast<CallOp>(op)) {
    // Handle method calls
    auto callee = callOp.getCallee();
    
    if (callee.getNestedReferences().size() == 0) {
      // Local method call
      // Check if this is a value method call
      bool isValueMethod = false;
      if (auto symbolOp = SymbolTable::lookupNearestSymbolFrom(callOp, callee)) {
        isValueMethod = isa<ValueMethodOp>(symbolOp);
      }
      
      if (isValueMethod) {
        // For value methods, we need to read the output port
        StringRef methodName = callee.getLeafReference();
        std::string portName = (methodName + "OUT").str();
        
        // Find the FIRRTL module
        auto firrtlModule = dyn_cast<FModuleOp>(ctx.firrtlBuilder.getBlock()->getParentOp());
        if (!firrtlModule) return failure();
        
        Value outputPort = getOrCreatePort(firrtlModule, portName, 
                                         convertType(callOp.getType(0)), Direction::Out);
        
        // Create a node to read the value
        auto nodeOp = ctx.firrtlBuilder.create<NodeOp>(
            callOp.getLoc(), outputPort, 
            ctx.firrtlBuilder.getStringAttr(methodName + "_call"));
        ctx.txnToFirrtl[callOp.getResult(0)] = nodeOp.getResult();
      }
    } else if (callee.getNestedReferences().size() == 1) {
      // Instance method call
      StringRef instName = callee.getRootReference().getValue();
      StringRef methodName = callee.getNestedReferences()[0].getValue();
      
      // For value method calls that produce results
      if (callOp.getNumResults() > 0) {
        // Find the instance ports
        auto instPorts = ctx.instancePorts.find(instName);
        if (instPorts != ctx.instancePorts.end()) {
          // Find the output port for this method
          auto outputPortName = (methodName + "_data").str();
          auto outputPort = instPorts->second.find(outputPortName);
          
          if (outputPort != instPorts->second.end()) {
            // Map the result to the output port
            ctx.txnToFirrtl[callOp.getResult(0)] = outputPort->second;
          } else {
            // If the port wasn't found with _data suffix, try without
            outputPortName = methodName.str();
            outputPort = instPorts->second.find(outputPortName);
            if (outputPort != instPorts->second.end()) {
              ctx.txnToFirrtl[callOp.getResult(0)] = outputPort->second;
            } else {
              // Port not found - this is an error
              return callOp.emitError("Could not find output port for instance method: ") 
                     << instName << "::" << methodName;
            }
          }
        } else {
          return callOp.emitError("Instance not found in ports map: ") << instName;
        }
      }
    }
  } else if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
    // Convert constants to FIRRTL
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      auto firrtlType = convertType(constOp.getType());
      if (!firrtlType) return failure();
      
      APInt value = intAttr.getValue();
      if (value.getBitWidth() == 1) {
        value = value.zext(std::max(1u, value.getBitWidth()));
      }
      APSInt apSInt(value, !isa<SIntType>(firrtlType));
      
      auto firrtlConst = ctx.firrtlBuilder.create<ConstantOp>(
          constOp.getLoc(), firrtlType, apSInt);
      ctx.txnToFirrtl[constOp.getResult()] = firrtlConst.getResult();
    }
  } else if (auto cmpOp = dyn_cast<arith::CmpIOp>(op)) {
    // Make sure operands are converted
    Value lhs = ctx.txnToFirrtl.lookup(cmpOp.getLhs());
    Value rhs = ctx.txnToFirrtl.lookup(cmpOp.getRhs());
    if (!lhs || !rhs) return failure();
    
    Value result;
    switch (cmpOp.getPredicate()) {
      case arith::CmpIPredicate::eq:
        result = ctx.firrtlBuilder.create<EQPrimOp>(cmpOp.getLoc(), lhs, rhs);
        break;
      case arith::CmpIPredicate::ne:
        result = ctx.firrtlBuilder.create<NEQPrimOp>(cmpOp.getLoc(), lhs, rhs);
        break;
      case arith::CmpIPredicate::sgt:
      case arith::CmpIPredicate::ugt:
        result = ctx.firrtlBuilder.create<GTPrimOp>(cmpOp.getLoc(), lhs, rhs);
        break;
      case arith::CmpIPredicate::sge:
      case arith::CmpIPredicate::uge:
        result = ctx.firrtlBuilder.create<GEQPrimOp>(cmpOp.getLoc(), lhs, rhs);
        break;
      case arith::CmpIPredicate::slt:
      case arith::CmpIPredicate::ult:
        result = ctx.firrtlBuilder.create<LTPrimOp>(cmpOp.getLoc(), lhs, rhs);
        break;
      case arith::CmpIPredicate::sle:
      case arith::CmpIPredicate::ule:
        result = ctx.firrtlBuilder.create<LEQPrimOp>(cmpOp.getLoc(), lhs, rhs);
        break;
    }
    
    if (result) {
      ctx.txnToFirrtl[cmpOp.getResult()] = result;
    }
  }
  // Add more operation types as needed
  
  return success();
}

/// Convert a Txn module to FIRRTL
static LogicalResult convertModule(::sharp::txn::ModuleOp txnModule, 
                                 ConversionContext &ctx,
                                 ::circt::firrtl::CircuitOp circuit,
                                 const std::string &willFireMode = "static") {
  MLIRContext *mlirCtx = txnModule->getContext();
  OpBuilder builder(mlirCtx);
  
  // Set current module in context
  ctx.currentTxnModule = txnModule;
  
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
      for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
        if (auto firrtlType = convertType(funcType.getInput(i))) {
          std::string portName;
          if (funcType.getNumInputs() == 1) {
            portName = (prefix + resultPostfix).str();
          } else {
            portName = (prefix + resultPostfix + "_arg" + std::to_string(i)).str();
          }
          ports.push_back({builder.getStringAttr(portName),
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
  
  // Get clock and reset signals - they'll be used when connecting instances
  
  // Create submodule instances
  DenseMap<StringRef, ::circt::firrtl::InstanceOp> firrtlInstances;
  DenseMap<StringRef, DenseMap<StringRef, Value>> instancePorts;
  
  // Track conversion failure
  bool instanceConversionFailed = false;
  
  txnModule.walk([&](::sharp::txn::InstanceOp txnInst) {
    // Find the target module to get its interface
    auto targetModuleName = txnInst.getModuleName();
    ::circt::firrtl::FModuleOp targetFIRRTLModule;
    
    // Look for the FIRRTL module in the circuit
    // For primitives, the FIRRTL module might have "_impl" suffix
    circuit.walk([&](::circt::firrtl::FModuleOp fmodule) {
      if (fmodule.getName() == targetModuleName || 
          fmodule.getName() == (targetModuleName + "_impl").str()) {
        targetFIRRTLModule = fmodule;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    
    if (!targetFIRRTLModule) {
      // Check if this is a primitive module with type arguments
      if (isKnownPrimitive(targetModuleName)) {
        // Get the data type from instance type arguments
        Type dataType = getInstanceDataType(txnInst);
        if (!dataType) {
          txnInst.emitError("Parametric primitive instance missing type arguments: ") << targetModuleName;
          instanceConversionFailed = true;
          return;
        }
        
        targetFIRRTLModule = getOrCreatePrimitiveFIRRTLModule(targetModuleName, dataType, circuit, ctx.firrtlBuilder);
        if (!targetFIRRTLModule) {
          txnInst.emitError("Failed to create primitive FIRRTL module for: ") << targetModuleName;
          instanceConversionFailed = true;
          return;
        }
      } else {
        // Module not yet converted - this shouldn't happen with proper ordering
        return;
      }
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
  
  // Check if instance conversion failed
  if (instanceConversionFailed) {
    return failure();
  }
  
  // Store instance ports in context for method call connections
  ctx.instancePorts = instancePorts;
  
  // First step: Map all block arguments to FIRRTL ports for all action methods
  txnModule.walk([&](ActionMethodOp actionMethod) {
    StringRef prefix = actionMethod.getPrefix().value_or(actionMethod.getSymName());
    auto methodPortNames = firrtlModule.getPortNames();
    auto methodBlockArgs = firrtlModule.getBodyBlock()->getArguments();
    
    // Map method arguments to FIRRTL ports
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
  });
  
  // Pre-pass: Convert all arith operations that might be used as conditions
  // This ensures reachability conditions are available in FIRRTL
  // We need multiple passes to handle dependencies
  bool changed = true;
  while (changed) {
    changed = false;
    txnModule.walk([&](Operation *op) {
      // Skip if already converted
      if (op->getNumResults() > 0 && ctx.txnToFirrtl.count(op->getResult(0))) {
        return;
      }
      
      if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
        // Convert constants to FIRRTL
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
          auto firrtlType = convertType(constOp.getType());
          if (firrtlType) {
            auto apInt = intAttr.getValue();
            bool isUnsigned = isa<UIntType>(firrtlType);
            auto firrtlConst = ctx.firrtlBuilder.create<ConstantOp>(
                constOp.getLoc(), Type(firrtlType), 
                APSInt(apInt, isUnsigned));
            ctx.txnToFirrtl[constOp.getResult()] = firrtlConst;
            changed = true;
          }
        } else if (auto boolAttr = dyn_cast<BoolAttr>(constOp.getValue())) {
          // Convert boolean constants
          auto intType = IntType::get(ctx.firrtlBuilder.getContext(), false, 1);
          auto value = boolAttr.getValue() ? 1 : 0;
          auto firrtlConst = ctx.firrtlBuilder.create<ConstantOp>(
              constOp.getLoc(), Type(intType), 
              APSInt(APInt(1, value), true));
          ctx.txnToFirrtl[constOp.getResult()] = firrtlConst;
          changed = true;
        }
      } else if (auto andOp = dyn_cast<arith::AndIOp>(op)) {
        // Convert AND operation
        Value lhs = ctx.txnToFirrtl.lookup(andOp.getLhs());
        Value rhs = ctx.txnToFirrtl.lookup(andOp.getRhs());
        if (lhs && rhs) {
          auto result = ctx.firrtlBuilder.create<AndPrimOp>(andOp.getLoc(), lhs, rhs);
          ctx.txnToFirrtl[andOp.getResult()] = result;
          changed = true;
        }
      } else if (auto xorOp = dyn_cast<arith::XOrIOp>(op)) {
        // Convert XOR operation
        Value lhs = ctx.txnToFirrtl.lookup(xorOp.getLhs());
        Value rhs = ctx.txnToFirrtl.lookup(xorOp.getRhs());
        if (lhs && rhs) {
          auto result = ctx.firrtlBuilder.create<XorPrimOp>(xorOp.getLoc(), lhs, rhs);
          ctx.txnToFirrtl[xorOp.getResult()] = result;
          changed = true;
        }
      } else if (auto cmpOp = dyn_cast<arith::CmpIOp>(op)) {
        // Convert comparison operations
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
          if (result) {
            ctx.txnToFirrtl[cmpOp.getResult()] = result;
            changed = true;
          }
        }
      }
    });
  }
  
  // Generate will-fire logic for each action in schedule order
  auto scheduleArrayAttr = schedule.getActions();
  SmallVector<StringRef> scheduleOrder;
  for (auto attr : scheduleArrayAttr) {
    auto symRef = cast<SymbolRefAttr>(attr);
    scheduleOrder.push_back(symRef.getRootReference().getValue());
  }
  
  // First pass: Calculate conflict_inside for each action and generate enabled signals
  DenseMap<StringRef, Value> enabledSignals;
  DenseMap<StringRef, Value> conflictInsideSignals;
  DenseMap<StringRef, Operation*> actionMap;
  
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
          // Value methods must not be in schedule according to execution model
          isValueMethod = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    
    if (isValueMethod) {
      // Error: value methods should never be in schedules
      return txnModule.emitError("Value method '") << name 
             << "' found in schedule. Only actions (rules and action methods) are allowed in schedules.";
    }
    
    if (!action) {
      return txnModule.emitError("Action not found in schedule: ") << name;
    }
    
    actionMap[name] = action;
    
    // Check if this action has potential conflicts
    // We'll calculate the actual conflict_inside logic later
    if (hasConflictingCalls(action, ctx)) {
      // Mark this action as needing conflict_inside calculation
      conflictInsideSignals[name] = nullptr; // Placeholder - will calculate later
    }
    
    // Generate enabled signal
    Value enabled;
    if (auto rule = dyn_cast<RuleOp>(action)) {
      // Evaluate rule guard by examining the rule body
      // A rule's guard is determined by conditions that affect whether
      // the rule executes its main actions
      
      // Look for the first conditional in the rule body
      Value guardCondition;
      bool foundGuard = false;
      
      rule.walk([&](IfOp ifOp) {
        if (!foundGuard && ifOp->getParentOp() == rule) {
          // This is a top-level if in the rule - likely the guard
          guardCondition = ifOp.getCondition();
          foundGuard = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      
      if (foundGuard && guardCondition) {
        // Convert the guard condition to FIRRTL
        // First, we need to convert all operations in the rule body that the guard depends on
        // Walk through all operations and convert them recursively
        for (auto &op : rule.getBody().front()) {
          // Skip operations that are already converted or don't produce values
          if (op.getNumResults() == 0) continue;
          
          // Check if any result is already converted
          bool alreadyConverted = true;
          for (auto result : op.getResults()) {
            if (!ctx.txnToFirrtl.count(result)) {
              alreadyConverted = false;
              break;
            }
          }
          
          if (!alreadyConverted) {
            // Convert this operation and its dependencies
            if (failed(convertOp(&op, ctx))) {
              rule.emitError("Failed to convert guard condition operation");
              continue;
            }
          }
        }
        
        // Look up the converted condition
        Value firrtlGuard = ctx.txnToFirrtl.lookup(guardCondition);
        if (firrtlGuard) {
          enabled = firrtlGuard;
        } else {
          // If we can't find the guard, default to always enabled
          auto intTy = IntType::get(mlirCtx, false, 1);
          enabled = ctx.firrtlBuilder.create<ConstantOp>(
              action->getLoc(), Type(intTy), APSInt(APInt(1, 1), true));
        }
      } else {
        // No guard found - rule is always enabled
        auto intTy = IntType::get(mlirCtx, false, 1);
        enabled = ctx.firrtlBuilder.create<ConstantOp>(
            action->getLoc(), Type(intTy), APSInt(APInt(1, 1), true));
      }
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
  }
  
  // Second pass: generate will-fire signals with simplified conflict_inside
  // In most-dynamic mode, we need to collect primitive information first,
  // but we create the will-fire signals in a single pass to maintain SSA form
  
  for (StringRef name : scheduleOrder) {
    Value enabled = enabledSignals[name];
    if (!enabled) {
      // Skip value methods (they don't have enabled signals)
      continue;
    }
    
    Value effectiveEnabled = enabled;
    
    // If this action has conflicting calls, add conflict_inside check using reachability
    if (conflictInsideSignals.count(name)) {
      auto action = actionMap[name];
      if (action) {
        auto loc = action->getLoc();
        auto intType = IntType::get(mlirCtx, false, 1);
        
        // Block arguments are already mapped globally at the top of the function
        
        // Now collect method calls with their reachability conditions
        SmallVector<std::pair<CallOp, Value>> methodCallsWithConditions;
        action->walk([&](CallOp call) {
          // Get the reachability condition if available
          Value condition;
          if (call.getCondition()) {
            // Look up the condition in FIRRTL - should be converted in pre-pass
            condition = ctx.txnToFirrtl.lookup(call.getCondition());
            if (!condition) {
              call.emitError("Condition not converted to FIRRTL");
              condition = ctx.firrtlBuilder.create<ConstantOp>(loc, Type(intType), 
                                                              APSInt(APInt(1, 1), true));
            }
          } else {
            // No condition means always reachable
            condition = ctx.firrtlBuilder.create<ConstantOp>(loc, Type(intType), 
                                                            APSInt(APInt(1, 1), true));
          }
          methodCallsWithConditions.push_back({call, condition});
        });
        
        // Check each pair of method calls for conflicts
        Value conflictInside;
        bool hasConflicts = false;
        
        for (size_t i = 0; i < methodCallsWithConditions.size(); ++i) {
          for (size_t j = i + 1; j < methodCallsWithConditions.size(); ++j) {
            auto [call1, cond1] = methodCallsWithConditions[i];
            auto [call2, cond2] = methodCallsWithConditions[j];
            
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
            
            // Build the conflict key - check both orderings
            std::string key1 = (inst1 + "::" + method1 + "," + inst2 + "::" + method2).str();
            std::string key2 = (inst2 + "::" + method2 + "," + inst1 + "::" + method1).str();
            
            // Check if methods conflict
            auto it1 = ctx.conflictMatrix.find(key1);
            auto it2 = ctx.conflictMatrix.find(key2);
            
            bool pairConflicts = false;
            if (it1 != ctx.conflictMatrix.end()) {
              auto rel = static_cast<ConflictRelation>(it1->second);
              if (rel == ConflictRelation::Conflict) pairConflicts = true;
            } else if (it2 != ctx.conflictMatrix.end()) {
              auto rel = static_cast<ConflictRelation>(it2->second);
              if (rel == ConflictRelation::Conflict) pairConflicts = true;
            }
            
            if (pairConflicts) {
              // These methods conflict - create: conflict(m1,m2) && reach(m1) && reach(m2)
              Value bothReachable = ctx.firrtlBuilder.create<AndPrimOp>(loc, cond1, cond2);
              
              if (!hasConflicts) {
                conflictInside = bothReachable;
                hasConflicts = true;
              } else {
                // OR with previous conflicts
                conflictInside = ctx.firrtlBuilder.create<OrPrimOp>(loc, conflictInside, bothReachable);
              }
            }
          }
        }
        
        // If we found conflicts, negate and include in enabled calculation
        if (hasConflicts) {
          // conflict_inside = OR of all (conflict(m1,m2) && reach(m1) && reach(m2))
          // noConflictInside = !conflict_inside
          Value noConflictInside = ctx.firrtlBuilder.create<NotPrimOp>(loc, conflictInside);
          
          // Include in enabled calculation
          effectiveEnabled = ctx.firrtlBuilder.create<AndPrimOp>(loc, enabled, noConflictInside);
        }
      }
    }
    
    // Generate will-fire signal
    Operation *actionForWF = actionMap[name];
    Value wf = generateWillFire(name, effectiveEnabled, scheduleOrder, ctx, willFireMode, actionForWF, &actionMap);
    
    // Create wire for will-fire signal and store it
    if (wf) {
      std::string wfName = name.str() + "_wf";
      auto wfNode = ctx.firrtlBuilder.create<NodeOp>(actionForWF->getLoc(), wf, wfName);
      ctx.willFireSignals[name] = wfNode.getResult();
    }
  }
  
  // Track conversion failures
  bool conversionFailed = false;
  
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
      conversionFailed = true;
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
            
            auto rel = getConflictRelation(other, actionMethod.getSymName(), ctx, actionMap);
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
    
    // Check if method body has any operations that generate FIRRTL
    bool hasNonTerminatorOps = false;
    for (auto &block : actionMethod.getBody()) {
      for (auto &op : block) {
        // Skip terminators and operations that don't generate FIRRTL
        if (!isa<ReturnOp, YieldOp, AbortOp>(op)) {
          hasNonTerminatorOps = true;
          break;
        }
      }
      if (hasNonTerminatorOps) break;
    }
    
    // Only create when block if there are operations to convert
    if (hasNonTerminatorOps) {
      // Execute method body when will-fire is true
      ctx.firrtlBuilder.create<WhenOp>(actionMethod.getLoc(), wf, false, [&]() {
        // Map method arguments to FIRRTL ports
        auto portNames = firrtlModule.getPortNames();
        auto blockArgs = firrtlModule.getBodyBlock()->getArguments();
        for (unsigned i = 0; i < actionMethod.getNumArguments(); ++i) {
          Value methodArg = actionMethod.getArgument(i);
          // For action methods with args, we created ports with pattern: methodOUT
          std::string argPortName = (prefix + "OUT").str();
          if (actionMethod.getNumArguments() > 1) {
            argPortName = (prefix + "OUT_arg" + std::to_string(i)).str();
          }
          
          for (size_t j = 0; j < portNames.size(); ++j) {
            if (cast<StringAttr>(portNames[j]).getValue() == argPortName) {
              ctx.txnToFirrtl[methodArg] = blockArgs[j];
              break;
            }
          }
        }
        
        // Convert method body
        // Note: Don't clear ctx.txnToFirrtl here to preserve block argument mappings
        // Block arguments and pre-converted conditions are preserved
          
        if (failed(convertBodyOps(actionMethod.getBody(), ctx))) {
          actionMethod.emitError("Failed to convert action method body");
          conversionFailed = true;
          return;
        }
      });
    }
  });
  
  // Third pass: Convert rules
  for (StringRef ruleName : scheduleOrder) {
    txnModule.walk([&](RuleOp rule) {
      if (rule.getSymName() != ruleName) return;
      
      Value wf = ctx.willFireSignals[ruleName];
      if (!wf) return;
      
      // Check if rule body has any operations that generate FIRRTL
      bool hasNonTerminatorOps = false;
      for (auto &block : rule.getBody()) {
        for (auto &op : block) {
          // Skip terminators and operations that don't generate FIRRTL
          if (!isa<ReturnOp, YieldOp, AbortOp>(op)) {
            hasNonTerminatorOps = true;
            break;
          }
        }
        if (hasNonTerminatorOps) break;
      }
      
      // Only create when block if there are operations to convert
      if (hasNonTerminatorOps) {
        // Execute rule body when will-fire is true
        ctx.firrtlBuilder.create<WhenOp>(rule.getLoc(), wf, false, [&]() {
          // Convert rule body
          // Note: Don't clear ctx.txnToFirrtl here to preserve pre-converted conditions
          if (failed(convertBodyOps(rule.getBody(), ctx))) {
            rule.emitError("Failed to convert rule body");
            conversionFailed = true;
            return;
          }
        });
      }
    });
  }
  
  // For now, skip conflict_inside calculation to avoid dominance issues
  // This will be addressed in a future enhancement
  
  // Return failure if any conversion failed
  if (conversionFailed) {
    return failure();
  }
  
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
    
    // First, collect reachability analysis results if available
    // The ReachabilityAnalysis pass adds condition operands to txn.call operations
    module.walk([&](::sharp::txn::ModuleOp txnModule) {
      txnModule.walk([&](Operation *op) {
        // Collect reachability conditions from CallOps
        if (auto callOp = dyn_cast<CallOp>(op)) {
          if (auto cond = callOp.getCondition()) {
            convCtx.reachabilityConditions[op] = cond;
          }
        }
        // Also need to collect abort reachability conditions
        // Since ReachabilityAnalysis now tracks aborts, we need to extract that info
        // For now, we'll handle this during the conversion phase
      });
    });
    
    // Get sorted module list
    SmallVector<::sharp::txn::ModuleOp> sortedModules;
    if (failed(analyzeModuleDependencies(module, sortedModules))) {
      signalPassFailure();
      return;
    }
    
    // Create a FIRRTL circuit to hold converted modules
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(module.getBody());
    
    // Find the top-level module name (module that is not instantiated by others)
    StringRef topModuleName = "Top";
    
    // Build a set of modules that are instantiated by others
    DenseSet<StringRef> instantiatedModules;
    for (auto txnModule : sortedModules) {
      txnModule.walk([&](::sharp::txn::InstanceOp inst) {
        instantiatedModules.insert(inst.getModuleName());
      });
    }
    
    // Find the first module that is not instantiated by others (true top module)
    for (auto txnModule : sortedModules) {
      if (!instantiatedModules.count(txnModule.getName())) {
        topModuleName = txnModule.getName();
        break;
      }
    }
    
    // If all modules are instantiated (circular), use the last one in dependency order
    if (topModuleName == "Top" && !sortedModules.empty()) {
      topModuleName = sortedModules.back().getName();
    }
    
    auto firrtlCircuit = builder.create<::circt::firrtl::CircuitOp>(
        module.getLoc(), builder.getStringAttr(topModuleName));
    
    // Convert each module bottom-up
    for (auto txnModule : sortedModules) {
      // Only convert Txn modules, not the top-level builtin module
      if (!isa<::sharp::txn::ModuleOp>(txnModule)) continue;
      if (txnModule.getName() == "firrtl_generated") continue;
      
      if (failed(convertModule(txnModule, convCtx, firrtlCircuit, willFireMode))) {
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