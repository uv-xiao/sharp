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
#include "sharp/Analysis/AnalysisError.h"
#include "sharp/Analysis/ConflictDebugger.h"
#include "sharp/Analysis/ScheduleDebugger.h"

#define DEBUG_TYPE "txn-to-firrtl"

namespace mlir {
namespace sharp {

#define GEN_PASS_DEF_TXNTOFIRRTLCONVERSION
#include "sharp/Conversion/Passes.h.inc"

namespace {

using namespace ::sharp::txn;
using namespace ::circt::firrtl;

/// Conflict relations from the execution model
enum class ConflictRelation {
  SequenceBefore = 0,   // SB
  SequenceAfter = 1,    // SA
  Conflict = 2,         // C
  ConflictFree = 3      // CF
};

/// Helper to convert enum to string for debugging
llvm::StringRef conflictRelationToString(ConflictRelation rel) {
  switch (rel) {
    case ConflictRelation::SequenceBefore: return "SB";
    case ConflictRelation::SequenceAfter: return "SA";  
    case ConflictRelation::Conflict: return "C";
    case ConflictRelation::ConflictFree: return "CF";
  }
  return "Unknown";
}

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
  
  // Removed: redundant reachability conditions
  // These are already computed by ReachabilityAnalysis pass and attached to operations
  
  // Removed: reachAbortCache - incorrect to cache reachAbort values
  // since they depend on call context, not just the action
  
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
  
  // Handle index types used for module types - not directly converted
  if (isa<IndexType>(type)) {
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

/// Helper to lookup called action method from a CallOp
static Operation* lookupCalledActionMethod(CallOp callOp, ConversionContext &ctx) {
  auto callee = callOp.getCallee();
  
  // Only handle direct calls to action methods (not instance method calls)
  if (callee.getNestedReferences().empty()) {
    StringRef methodName = callee.getRootReference().getValue();
    
    // Look for action method in current module
    Operation* foundMethod = nullptr;
    ctx.currentTxnModule.walk([&](ActionMethodOp actionMethod) {
      if (actionMethod.getSymName() == methodName) {
        foundMethod = actionMethod.getOperation();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return foundMethod;
  }
  return nullptr;
}

/// Get reachability condition for a call operation
static Value getCallReachCondition(CallOp callOp, ConversionContext &ctx) {
  auto &builder = ctx.firrtlBuilder;
  auto loc = callOp.getLoc();
  auto boolType = IntType::get(builder.getContext(), false, 1);
  
  // Check if ReachabilityAnalysis provided a condition
  if (auto condAttr = callOp.getOperation()->getAttrOfType<FlatSymbolRefAttr>("condition")) {
    // TODO: Get the actual value from the symbol reference
    // For now, skip analysis conditions and use operand-based conditions
  }
  
  // Fall back to condition operand if present
  if (callOp.getCondition()) {
    Value callCond = ctx.txnToFirrtl.lookup(callOp.getCondition());
    if (callCond) return callCond;
  }
  
  // Default: always reachable
  return builder.create<ConstantOp>(loc, Type(boolType), APSInt(APInt(1, 1), true));
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

// Forward declaration for convertGuardRegion
static Value convertGuardRegion(Region &guardRegion, ConversionContext &ctx);

/// Calculate reach_abort for an action with recursive analysis
/// reach_abort[action] = OR(reach(abort_i, action) for every abort_i in action) 
///                     || OR(reach(call_i, action) && reach_abort(method[call_i]) for every call_i in action)
static Value calculateReachAbort(Operation *action, ConversionContext &ctx, 
                               const DenseMap<StringRef, Operation*> &actionMap) {
  auto &builder = ctx.firrtlBuilder;
  auto loc = action->getLoc();
  auto boolType = IntType::get(builder.getContext(), false, 1);
  
  // Start with false (no abort) - no caching since reachAbort depends on call context
  Value reachAbort = builder.create<ConstantOp>(loc, Type(boolType), 
                                                APSInt(APInt(1, 0), true));
  
  // Handle guard regions first - if guard returns false, the action aborts
  Value guardAbort = nullptr;
  if (auto rule = dyn_cast<RuleOp>(action)) {
    if (rule.hasGuard()) {
      guardAbort = convertGuardRegion(rule.getGuardRegion(), ctx);
    }
  } else if (auto method = dyn_cast<ActionMethodOp>(action)) {
    if (method.hasGuard()) {
      guardAbort = convertGuardRegion(method.getGuardRegion(), ctx);
    }
  }
  
  if (guardAbort) {
    // Guard abort condition: NOT(guard_result)
    Value notGuard = builder.create<NotPrimOp>(loc, guardAbort);
    reachAbort = builder.create<OrPrimOp>(loc, reachAbort, notGuard);
  }
  
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
        if (auto abortOp = dyn_cast<::sharp::txn::AbortOp>(nested)) {
          // Found an abort - use its path condition combined with analysis result
          Value abortReach = pathCond;
          
          // Check if ReachabilityAnalysis provided a condition for this abort
          if (auto condAttr = nested->getAttrOfType<FlatSymbolRefAttr>("condition")) {
            // TODO: Get the actual value from the symbol reference
            // For now, use the abort's direct condition operand
          } else if (abortOp.getCondition()) {
            // Use the condition operand provided by ReachabilityAnalysis pass
            Value abortCond = ctx.txnToFirrtl.lookup(abortOp.getCondition());
            if (abortCond) {
              if (abortReach) {
                abortReach = builder.create<AndPrimOp>(loc, abortReach, abortCond);
              } else {
                abortReach = abortCond;
              }
            }
          }
          
          // Only include abort if it has a reachability condition
          // If there's no condition, it means the abort is unreachable or was eliminated
          if (abortReach) {
            // OR with existing reach_abort
            reachAbort = builder.create<OrPrimOp>(loc, reachAbort, abortReach);
          }
        } else if (auto callOp = dyn_cast<CallOp>(nested)) {
          // For method calls, use path condition combined with the call's own condition
          Value callReach = pathCond;
          
          // Check if ReachabilityAnalysis provided a condition 
          if (auto condAttr = nested->getAttrOfType<FlatSymbolRefAttr>("condition")) {
            // TODO: Get the actual value from the symbol reference
            // For now, use the call's direct condition operand
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
            
            // For instance method calls, the abort condition is NOT RDY
            // This implements the requirement: when action a0 calls @i::@ax, 
            // the abort should be @i::@ax's RDY signal (NOT RDY = abort)
            StringRef instName = callOp.getCallee().getRootReference().getValue();
            
            // Construct the RDY signal name: instanceName::methodNameRDY
            std::string rdySignalName = (instName + "_" + methodName + "RDY").str();
            
            // Look for the RDY signal in the current FIRRTL module ports
            auto currentModule = builder.getInsertionBlock()->getParentOp();
            if (auto fmodule = dyn_cast<circt::firrtl::FModuleOp>(currentModule)) {
              auto portNames = fmodule.getPortNames();
              auto blockArgs = fmodule.getBodyBlock()->getArguments();
              
              Value rdySignal = nullptr;
              for (size_t i = 0; i < portNames.size(); ++i) {
                if (cast<StringAttr>(portNames[i]).getValue() == rdySignalName) {
                  rdySignal = blockArgs[i];
                  break;
                }
              }
              
              if (rdySignal) {
                // Method abort condition = callReach && NOT(RDY)
                Value notRdy = builder.create<NotPrimOp>(loc, rdySignal);
                Value methodAbortCond = builder.create<AndPrimOp>(loc, callReach, notRdy);
                reachAbort = builder.create<OrPrimOp>(loc, reachAbort, methodAbortCond);
              } else {
                // Fallback: some methods don't have RDY signals (always ready)
                // Register/Wire read/write methods typically don't abort
                bool methodCanAbort = false;
                if (methodName == "dequeue" || methodName == "enqueue") {
                  // FIFO operations can abort when not ready
                  methodCanAbort = true;
                }
                
                if (methodCanAbort) {
                  // For methods without explicit RDY signals, use a conservative approach
                  Value methodAbortCond = builder.create<ConstantOp>(loc, Type(boolType), 
                                                                    APSInt(APInt(1, 0), true)); // Assume always ready
                  Value callAborts = builder.create<AndPrimOp>(loc, callReach, methodAbortCond);
                  reachAbort = builder.create<OrPrimOp>(loc, reachAbort, callAborts);
                }
              }
            }
          } else {
            // For local action method calls, perform recursive analysis
            if (auto calledAction = lookupCalledActionMethod(callOp, ctx)) {
              // Recursively calculate reach_abort for the called action method
              Value calleeReachAbort = calculateReachAbort(calledAction, ctx, actionMap);
              if (calleeReachAbort) {
                // Propagate abort: callReach && calleeReachAbort
                Value propagatedAbort = builder.create<AndPrimOp>(loc, callReach, calleeReachAbort);
                reachAbort = builder.create<OrPrimOp>(loc, reachAbort, propagatedAbort);
              }
            }
            // Note: If not an action method (e.g., value method), it cannot abort
          }
        }
      });
    }
  }
  
  // No caching - reachAbort depends on call context
  return reachAbort;
}

/// Collect all reachable calls (direct and indirect) from an action
static void collectAllReachableCalls(Operation *action, 
                                   ConversionContext &ctx,
                                   SmallVector<std::pair<CallOp, Value>> &allCalls,
                                   DenseSet<Operation*> &visited) {
  if (visited.count(action)) return; // Avoid cycles
  visited.insert(action);
  
  walkWithPathConditions(action, nullptr, ctx, [&](Operation *op, Value pathCond) {
    if (auto callOp = dyn_cast<CallOp>(op)) {
      Value callReachCondition = getCallReachCondition(callOp, ctx);
      
      // Combine path condition with call condition
      Value effectiveCondition = pathCond;
      if (pathCond && callReachCondition) {
        effectiveCondition = ctx.firrtlBuilder.create<AndPrimOp>(
          callOp.getLoc(), pathCond, callReachCondition);
      } else if (callReachCondition) {
        effectiveCondition = callReachCondition;
      }
      
      allCalls.push_back({callOp, effectiveCondition});
      
      // If calling local action method, recurse to collect indirect calls
      if (auto calleeAction = lookupCalledActionMethod(callOp, ctx)) {
        collectAllReachableCalls(calleeAction, ctx, allCalls, visited);
      }
    }
  });
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
  
  // Cannot reach here.
  // signalPassFailure();
}

/// Generate static mode will-fire logic for an action
/// wf[action] = enabled[action] && !reach_abort[action] && !conflicts_with_earlier[action] && !conflict_inside[action]
static Value generateStaticWillFire(StringRef actionName, Value enabled,
                                  ArrayRef<StringRef> schedule,
                                  ConversionContext &ctx,
                                  const DenseMap<StringRef, Operation*> &actionMap) {
  auto &builder = ctx.firrtlBuilder;
  auto loc = builder.getUnknownLoc();
  
  // Initialize debugging
  ConflictDebugger debugger("TxnToFIRRTL", TimingMode::Static);
  std::vector<ConflictDebugInfo> conflicts;
  
  // Start with enabled signal
  Value wf = enabled;
  
  if (!wf) {
    CONFLICT_DEBUG(debugger, logWillFireDecision(actionName, false, "", conflicts, false)
                             .setExplanation("FAILED: Enabled signal is null"));
    llvm::errs() << "generateStaticWillFire: enabled signal is null for " << actionName << "\n";
    // signalPassFailure();
    return nullptr;
  }
  
  // Calculate reach_abort for this action
  auto actionIt = actionMap.find(actionName);
  Value reachAbort = nullptr;
  std::string abortCondition = "none";
  if (actionIt != actionMap.end()) {
    reachAbort = calculateReachAbort(actionIt->second, ctx, actionMap);
    if (reachAbort) {
      abortCondition = "computed";
      CONFLICT_DEBUG(debugger, logAbortCondition(actionName, abortCondition, "calculated from method calls and abort operations"));
      
      // wf = enabled && !reach_abort
      Value notAbort = builder.create<NotPrimOp>(loc, reachAbort);
      wf = builder.create<AndPrimOp>(loc, wf, notAbort);
    } else {
      abortCondition = "none";
    }
  }
  
  // Check conflicts with earlier actions (static mode)
  // conflicts_with_earlier[a] = OR(wf[a1] && conflict(a1, a) for all a1 scheduled before a)
  for (StringRef earlier : schedule) {
    if (earlier == actionName) break;
    
    auto rel = getConflictRelation(earlier, actionName, ctx, actionMap);
    
    // Log conflict detection
    ConflictDebugInfo conflictInfo;
    conflictInfo.method1 = earlier.str();
    conflictInfo.method2 = actionName.str();
    conflictInfo.conflictType = conflictRelationToString(rel).str();
    conflictInfo.reachCondition1 = "1"; // Actions are always reachable at top level
    conflictInfo.reachCondition2 = "1";
    conflictInfo.isConflicting = (rel == ConflictRelation::Conflict || rel == ConflictRelation::SequenceBefore);
    conflicts.push_back(conflictInfo);
    
    CONFLICT_DEBUG(debugger, logConflictDetection(earlier, actionName, conflictRelationToString(rel), "1", "1", conflictInfo.isConflicting));
    
    // Generate conflict check if needed
    // conflicts(a1, a2) = (CM[a1,a2] == C) || (CM[a1,a2] == SA && wf[a1])
    if (rel == ConflictRelation::Conflict) {
      auto wfIt = ctx.willFireSignals.find(earlier);
      if (wfIt == ctx.willFireSignals.end()) {
        // Earlier action hasn't been processed yet or is a value method
        llvm::errs() << "Error: no will-fire signal found for earlier action " << earlier << "\n";
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
  
  // Calculate conflict_inside for this action
  // conflict_inside[action] = OR(conflict(m1,m2) && reach(m1) && reach(m2) for every m1,m2 in action_calls[action] where m1 before m2)
  auto conflictActionIt = actionMap.find(actionName);
  if (conflictActionIt != actionMap.end()) {
    Operation *action = conflictActionIt->second;
    
    // Collect all method calls from this action
    SmallVector<CallOp> calls;
    action->walk([&](CallOp call) { calls.push_back(call); });
    
    // Check for conflicts between method calls within the same action
    auto boolType = IntType::get(builder.getContext(), false, 1);
    Value conflictInside = builder.create<ConstantOp>(loc, Type(boolType), 
                                                     APSInt(APInt(1, 0), true));
    
    for (size_t i = 0; i < calls.size(); ++i) {
      for (size_t j = i + 1; j < calls.size(); ++j) {
        CallOp call_i = calls[i];
        CallOp call_j = calls[j];
        
        // Check if these method calls conflict
        StringRef method_i = call_i.getCallee().getRootReference().getValue();
        StringRef method_j = call_j.getCallee().getRootReference().getValue();
        
        auto rel = getConflictRelationFromString(method_i.str(), method_j.str(), ctx);
        if (rel == ConflictRelation::Conflict) {
          // Get reachability conditions for both calls
          Value reach_i = nullptr, reach_j = nullptr;
          
          // Check reachability analysis results first
          // Fall back to condition operand if analysis didn't provide result
          if (!reach_i && call_i.getCondition()) {
            reach_i = ctx.txnToFirrtl.lookup(call_i.getCondition());
          }
          // Default to always reachable if no condition
          if (!reach_i) {
            reach_i = builder.create<ConstantOp>(loc, Type(boolType), 
                                                APSInt(APInt(1, 1), true));
          }
          
          if (!reach_j && call_j.getCondition()) {
            reach_j = ctx.txnToFirrtl.lookup(call_j.getCondition());
          }
          if (!reach_j) {
            reach_j = builder.create<ConstantOp>(loc, Type(boolType), 
                                                APSInt(APInt(1, 1), true));
          }
          
          // conflict_inside |= reach(call_i) && reach(call_j)
          Value bothReachable = builder.create<AndPrimOp>(loc, reach_i, reach_j);
          conflictInside = builder.create<OrPrimOp>(loc, conflictInside, bothReachable);
        }
      }
    }
    
    // Apply conflict_inside: wf = wf && !conflict_inside
    Value notConflictInside = builder.create<NotPrimOp>(loc, conflictInside);
    wf = builder.create<AndPrimOp>(loc, wf, notConflictInside);
  }
  
  // Log final will-fire decision
  bool finalResult = (wf != nullptr);
  std::string explanation = finalResult ? 
    "SUCCESS: Generated static will-fire with " + std::to_string(conflicts.size()) + " conflict checks" :
    "FAILED: Could not generate will-fire signal";
    
  CONFLICT_DEBUG(debugger, logWillFireDecision(actionName, true, abortCondition, conflicts, finalResult)
                           .setExplanation(explanation));
  
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
  
  // Initialize debugging
  ConflictDebugger debugger("TxnToFIRRTL", TimingMode::Dynamic);
  std::vector<ConflictDebugInfo> conflicts;
  
  // Start with enabled signal
  Value wf = enabled;
  
  if (!wf) {
    CONFLICT_DEBUG(debugger, logWillFireDecision(actionName, false, "", conflicts, false)
                             .setExplanation("FAILED: Enabled signal is null"));
    llvm::errs() << "generateDynamicWillFire: enabled signal is null for " << actionName << "\n";
    return nullptr;
  }
  
  // Calculate reach_abort for this action
  Value reachAbort = calculateReachAbort(action, ctx, actionMap);
  std::string abortCondition = "none";
  if (reachAbort) {
    abortCondition = "computed";
    // CONFLICT_DEBUG(debugger, logAbortCondition(actionName, abortCondition, "calculated from method calls and abort operations"));
    
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
      // Extract full method signature for proper conflict detection
      std::string methodKey;
      if (call.getCallee().getNestedReferences().size() > 0) {
        // Instance method call: @instance::@method
        StringRef instance = call.getCallee().getRootReference().getValue();
        StringRef method = call.getCallee().getLeafReference().getValue();
        methodKey = instance.str() + "::" + method.str();
      }
      
      // Get reachability condition for this call
      Value condition;
      if (call.getCondition()) {
        condition = ctx.txnToFirrtl.lookup(call.getCondition());
        if (!condition) {
          // give a error
          llvm::errs() << "Error: no reachability condition implemented for call " << methodKey << "\n";
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
    // Extract full method signature for proper conflict detection
    std::string methodKey;
    if (call.getCallee().getNestedReferences().size() > 0) {
      // Instance method call: @instance::@method
      StringRef instance = call.getCallee().getRootReference().getValue();
      StringRef method = call.getCallee().getLeafReference().getValue();
      methodKey = instance.str() + "::" + method.str();
    }
    
    // Get reachability condition for this call
    Value condition;
    if (call.getCondition()) {
      condition = ctx.txnToFirrtl.lookup(call.getCondition());
      if (!condition) {
        // give error and abort
        llvm::errs() << "Error: no reachability condition found for call " << methodKey << "\n";
      }
    } else {
      // default to be true
      condition = builder.create<ConstantOp>(loc, Type(intType), 
                                            APSInt(APInt(1, 1), true));
    }
    
    // Check for conflicts with any earlier method calls
    for (auto &[earlierMethodKey, earlierCalled] : methodCalled) {
      // Check if these methods conflict
      // In dynamic mode, methods conflict if:
      // 1. They are the same method call (exact match), OR
      // 2. They access the same resource with conflicting operations (write-write conflicts)
      bool methodsConflict = false;
      
      // Exact method match
      if (earlierMethodKey.str() == methodKey) {
        methodsConflict = true;
      } else {
        // Resource-level conflict: check if both methods write to the same instance
        // Extract instance and method names
        auto parseMethodKey = [](const std::string& key) -> std::pair<std::string, std::string> {
          size_t pos = key.find("::");
          if (pos != std::string::npos) {
            return {key.substr(0, pos), key.substr(pos + 2)};
          }
          return {"", key};
        };
        
        auto [earlierInstance, earlierMethod] = parseMethodKey(earlierMethodKey.str());
        auto [currentInstance, currentMethod] = parseMethodKey(methodKey);
        
        // If they access the same instance and both are write operations, they conflict
        if (!earlierInstance.empty() && !currentInstance.empty() && 
            earlierInstance == currentInstance) {
          // Check if both are write methods (methods that modify state)
          bool earlierIsWrite = (earlierMethod == "write" || earlierMethod.find("set") == 0 || 
                                earlierMethod.find("push") == 0 || earlierMethod.find("pop") == 0);
          bool currentIsWrite = (currentMethod == "write" || currentMethod.find("set") == 0 ||
                                currentMethod.find("push") == 0 || currentMethod.find("pop") == 0);
          
          if (earlierIsWrite && currentIsWrite) {
            methodsConflict = true;
          }
        }
      }
      
      if (methodsConflict) {
        // conflict_with_earlier(m) = method_called[M'] && conflict(M', M)
        Value conflictCondition = builder.create<AndPrimOp>(loc, earlierCalled, condition);
        
        // wf = wf && !(reach(m) && conflict_with_earlier(m))
        auto notConflict = builder.create<NotPrimOp>(loc, conflictCondition);
        wf = builder.create<AndPrimOp>(loc, wf, notConflict);
        
        // Debug log the conflict detection
        ConflictDebugInfo conflictInfo;
        conflictInfo.actionName = actionName.str();
        conflictInfo.method1 = methodKey;
        conflictInfo.method2 = earlierMethodKey.str();
        conflictInfo.conflictType = "resource-write";
        conflictInfo.isConflicting = true;
        conflictInfo.explanation = "Write methods conflict on same resource instance";
        conflicts.push_back(conflictInfo);
      }
    }
  });
  
  // Log final will-fire decision for dynamic mode
  bool finalResult = (wf != nullptr);
  std::string explanation = finalResult ? 
    "SUCCESS: Generated dynamic will-fire with method-level conflict tracking" :
    "FAILED: Could not generate will-fire signal";
    
  CONFLICT_DEBUG(debugger, logWillFireDecision(actionName, true, abortCondition, conflicts, finalResult)
                           .setExplanation(explanation));
  
  // Return the will-fire value - node creation is handled in the main loop
  return wf;
}

/// Generate will-fire logic for an action (dispatcher for static/dynamic modes)
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
  } else {
    return generateStaticWillFire(actionName, enabled, schedule, ctx, *actionMap);
  }
}

// Forward declaration
static LogicalResult convertBodyOps(Region &region, ConversionContext &ctx);
static LogicalResult convertOp(Operation *op, ConversionContext &ctx);
static Value convertGuardRegion(Region &guardRegion, ConversionContext &ctx);


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
            return callOp.emitError("[TxnToFIRRTL] Pass failed - invalid call")
                   << ": Action cannot call another action '" << methodName 
                   << "' in the same module at " << callOp.getLoc() << ". "
                   << "Reason: The Sharp execution model prohibits actions from calling other actions within the same module "
                   << "to prevent recursion and undefined behavior. "
                   << "Solution: Refactor the code to have the action call a value method or a method of a child instance instead.";
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
              auto enablePortName = (methodName + "EN").str();
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
        
        // Handle primitive method calls with proper port connections
        if (methodName == "write" && callOp.getArgs().size() > 0) {
          // Convert the argument
          Value arg = callOp.getArgs()[0];
          Value firrtlArg = ctx.txnToFirrtl.lookup(arg);
          if (firrtlArg) {
            // Connect to write_data port
            auto writeDataPortName = "write_data";
            auto writeDataPort = instPorts->second.find(writeDataPortName);
            if (writeDataPort != instPorts->second.end()) {
              ctx.firrtlBuilder.create<ConnectOp>(callOp.getLoc(),
                                                 writeDataPort->second, firrtlArg);
            }
            
            // Enable the write
            auto writeEnablePortName = "write_enable";
            auto writeEnablePort = instPorts->second.find(writeEnablePortName);
            if (writeEnablePort != instPorts->second.end()) {
              auto intTy = IntType::get(ctx.firrtlBuilder.getContext(), false, 1);
              auto trueVal = ctx.firrtlBuilder.create<ConstantOp>(
                  callOp.getLoc(), Type(intTy), APSInt(APInt(1, 1), true));
              ctx.firrtlBuilder.create<ConnectOp>(callOp.getLoc(),
                                                 writeEnablePort->second, trueVal);
            }
          }
        } else if (methodName == "read" && callOp.getNumResults() > 0) {
          // Connect read result
          auto readDataPortName = "read_data";
          auto readDataPort = instPorts->second.find(readDataPortName);
          if (readDataPort != instPorts->second.end()) {
            // Map the result to the read_data port
            ctx.txnToFirrtl[callOp.getResult(0)] = readDataPort->second;
            
            // Enable the read
            auto readEnablePortName = "read_enable";
            auto readEnablePort = instPorts->second.find(readEnablePortName);
            if (readEnablePort != instPorts->second.end()) {
              auto intTy = IntType::get(ctx.firrtlBuilder.getContext(), false, 1);
              auto trueVal = ctx.firrtlBuilder.create<ConstantOp>(
                  callOp.getLoc(), Type(intTy), APSInt(APInt(1, 1), true));
              ctx.firrtlBuilder.create<ConnectOp>(callOp.getLoc(),
                                                 readEnablePort->second, trueVal);
            }
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
      return futureOp.emitError("[TxnToFIRRTL] Pass failed - unsupported operation")
             << ": txn.future operations are not yet supported in FIRRTL conversion at "
             << futureOp.getLoc() << ". "
             << "Reason: Multi-cycle operations require a more complex state machine and are not yet implemented in the FIRRTL backend. "
             << "Solution: Please refactor the design to use single-cycle operations or implement the required state machine manually in a separate module.";
    } else if (auto launchOp = dyn_cast<LaunchOp>(&op)) {
      return launchOp.emitError("[TxnToFIRRTL] Pass failed - unsupported operation")
             << ": txn.launch operations are not yet supported in FIRRTL conversion at "
             << launchOp.getLoc() << ". "
             << "Reason: Multi-cycle operations require a more complex state machine and are not yet implemented in the FIRRTL backend. "
             << "Solution: Please refactor the design to use single-cycle operations or implement the required state machine manually in a separate module.";
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
          // Try different port naming patterns
          std::vector<std::string> portNameCandidates = {
            (methodName + "_result").str(),  // For action methods with return values
            (methodName + "_data").str(),    // For value methods
            methodName.str()                 // Fallback
          };
          
          Value outputPort;
          for (const auto& candidateName : portNameCandidates) {
            auto portIt = instPorts->second.find(candidateName);
            if (portIt != instPorts->second.end()) {
              outputPort = portIt->second;
              break;
            }
          }
          
          if (outputPort) {
            // Map the result to the output port
            ctx.txnToFirrtl[callOp.getResult(0)] = outputPort;
          } else {
            // Port not found - this is an error
            return callOp.emitError("Could not find output port for instance method: ") 
                   << instName << "::" << methodName;
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
  } else if (auto andOp = dyn_cast<arith::AndIOp>(op)) {
    // Convert integer bitwise AND
    Value lhs = ctx.txnToFirrtl.lookup(andOp.getLhs());
    Value rhs = ctx.txnToFirrtl.lookup(andOp.getRhs());
    if (!lhs || !rhs) return failure();
    
    Value result = ctx.firrtlBuilder.create<AndPrimOp>(andOp.getLoc(), lhs, rhs);
    ctx.txnToFirrtl[andOp.getResult()] = result;
  } else if (auto orOp = dyn_cast<arith::OrIOp>(op)) {
    // Convert integer bitwise OR
    Value lhs = ctx.txnToFirrtl.lookup(orOp.getLhs());
    Value rhs = ctx.txnToFirrtl.lookup(orOp.getRhs());
    if (!lhs || !rhs) return failure();
    
    Value result = ctx.firrtlBuilder.create<OrPrimOp>(orOp.getLoc(), lhs, rhs);
    ctx.txnToFirrtl[orOp.getResult()] = result;
  } else if (auto xorOp = dyn_cast<arith::XOrIOp>(op)) {
    // Convert integer bitwise XOR
    Value lhs = ctx.txnToFirrtl.lookup(xorOp.getLhs());
    Value rhs = ctx.txnToFirrtl.lookup(xorOp.getRhs());
    if (!lhs || !rhs) return failure();
    
    Value result = ctx.firrtlBuilder.create<XorPrimOp>(xorOp.getLoc(), lhs, rhs);
    ctx.txnToFirrtl[xorOp.getResult()] = result;
  } else if (auto addOp = dyn_cast<arith::AddIOp>(op)) {
    // Convert integer addition
    Value lhs = ctx.txnToFirrtl.lookup(addOp.getLhs());
    Value rhs = ctx.txnToFirrtl.lookup(addOp.getRhs());
    if (!lhs || !rhs) return failure();
    
    Value result = ctx.firrtlBuilder.create<AddPrimOp>(addOp.getLoc(), lhs, rhs);
    ctx.txnToFirrtl[addOp.getResult()] = result;
  }
  // Add more operation types as needed
  
  return success();
}

/// Convert a guard region to FIRRTL and return the guard condition
/// Guard regions should contain logic that computes a boolean result
/// The result represents the "NOT abort" condition - if false, the action aborts
static Value convertGuardRegion(Region &guardRegion, ConversionContext &ctx) {
  if (guardRegion.empty() || guardRegion.front().empty()) {
    return nullptr;
  }
  
  auto &builder = ctx.firrtlBuilder;
  auto loc = guardRegion.getLoc();
  auto boolType = IntType::get(builder.getContext(), false, 1);
  
  // Convert all operations in the guard region
  for (auto &block : guardRegion) {
    for (auto &op : block) {
      if (auto yieldOp = dyn_cast<YieldOp>(&op)) {
        // The yielded value is the guard condition
        if (yieldOp.getNumOperands() > 0) {
          Value guardCondition = yieldOp.getOperand(0);
          if (ctx.txnToFirrtl.count(guardCondition)) {
            return ctx.txnToFirrtl[guardCondition];
          } else {
            // Try to convert the operand
            if (failed(convertOp(guardCondition.getDefiningOp(), ctx))) {
              return nullptr;
            }
            return ctx.txnToFirrtl.lookup(guardCondition);
          }
        }
      } else {
        // Convert other operations in the guard region
        if (failed(convertOp(&op, ctx))) {
          // If conversion fails, assume guard always succeeds (conservative)
          return builder.create<ConstantOp>(loc, Type(boolType), 
                                           APSInt(APInt(1, 1), true));
        }
      }
    }
  }
  
  // If no yield found, assume guard always succeeds
  return builder.create<ConstantOp>(loc, Type(boolType), 
                                   APSInt(APInt(1, 1), true));
}

/// Convert a Txn module to FIRRTL
static LogicalResult convertModule(::sharp::txn::ModuleOp txnModule, 
                                 ConversionContext &ctx,
                                 ::circt::firrtl::CircuitOp circuit,
                                 const std::string &willFireMode = "static",
                                 bool isTopLevel = false) {
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
    return txnModule.emitError("[TxnToFIRRTL] Pass failed - missing schedule")
           << ": The txn.module '" << txnModule.getName() << "' does not contain a txn.schedule operation at "
           << txnModule.getLoc() << ". "
           << "Reason: A schedule is required to determine the execution order of actions for FIRRTL conversion. "
           << "Solution: Please run the sharp-action-scheduling pass to generate a schedule before converting to FIRRTL.";
  }
  
  populateConflictMatrix(schedule, ctx);
  
  // Debug output for schedule and conflict matrix (if debug enabled)
  LLVM_DEBUG({
    ScheduleDebugger schedDebugger("TxnToFIRRTL", willFireMode);
    
    // Prepare schedule debug info
    ScheduleDebugInfo scheduleInfo;
    scheduleInfo.moduleName = txnModule.getName().str();
    scheduleInfo.timingMode = willFireMode;
    
    auto scheduleArrayAttr = schedule.getActions();
    for (auto attr : scheduleArrayAttr) {
      auto symRef = cast<SymbolRefAttr>(attr);
      scheduleInfo.actions.push_back(symRef.getRootReference().getValue().str());
    }
    schedDebugger.setSchedule(scheduleInfo);
    
    // Prepare conflict matrix debug info
    ConflictMatrixDebugInfo conflictInfo;
    conflictInfo.moduleName = txnModule.getName().str();
    conflictInfo.allActions = scheduleInfo.actions;
    
    // Convert conflict matrix from ctx.conflictMatrix
    for (const auto& entry : ctx.conflictMatrix) {
      std::string key = entry.first().str();
      int conflictValue = entry.second;
      std::string relStr;
      switch (conflictValue) {
        case 0: relStr = "SB"; break;  // SequenceBefore
        case 1: relStr = "SA"; break;  // SequenceAfter
        case 2: relStr = "C"; break;   // Conflict
        case 3: relStr = "CF"; break;  // ConflictFree
        default: relStr = "?"; break;  // Unknown
      }
      conflictInfo.conflictEntries[key] = relStr;
    }
    schedDebugger.setConflictMatrix(conflictInfo);
    
    // Print the pretty debug output
    schedDebugger.printAll();
  });
  
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
      StringRef enablePostfix = "EN";
      ports.push_back({builder.getStringAttr((prefix + enablePostfix).str()),
                      UIntType::get(mlirCtx, 1), Direction::In});
      
    } else if (auto actionMethod = dyn_cast<ActionMethodOp>(op)) {
      // Get method attributes
      StringRef prefix = actionMethod.getPrefix().value_or(actionMethod.getSymName());
      StringRef enablePostfix = actionMethod.getEnable().value_or("EN");
      StringRef readyPostfix = actionMethod.getReady().value_or("RDY");
      
      // Add data input ports for arguments
      auto funcType = actionMethod.getFunctionType();
      for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
        if (auto firrtlType = convertType(funcType.getInput(i))) {
          // Get the actual argument name from the method
          std::string argName;
          if (auto argNameAttr = actionMethod.getArgAttrOfType<StringAttr>(i, "mlir.arg_name")) {
            argName = argNameAttr.getValue().str();
          } else {
            // Fallback to default name if no arg name attribute
            argName = "arg" + std::to_string(i);
          }
          std::string portName = (prefix + "Arg_" + argName).str();
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
      
      // Add output port for return value if action method has results
      if (funcType.getNumResults() > 0) {
        if (auto firrtlType = convertType(funcType.getResult(0))) {
          std::string resultPortName = (prefix + "_result").str();
          ports.push_back({builder.getStringAttr(resultPortName),
                          firrtlType, Direction::Out});
        }
      }
    }
  });
  
  // Create FIRRTL module
  builder.setInsertionPointToEnd(circuit.getBodyBlock());
  auto originalModuleName = txnModule.getSymName();
  if (originalModuleName.empty()) {
    return txnModule.emitError("Txn module missing symbol name");
  }
  
  // Use original module name to match circuit name for toolchain compatibility
  StringRef moduleName = originalModuleName;
  
  auto firrtlModule = builder.create<FModuleOp>(
      txnModule.getLoc(), 
      builder.getStringAttr(moduleName),
      ConventionAttr::get(mlirCtx, Convention::Internal),
      ports);
  
  // No longer need annotations since we're using original module name
  
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
  
  // Add default connections for primitive instance input ports to avoid "sink not fully initialized" errors
  for (auto& [instName, ports] : instancePorts) {
    // For Register primitives, we need to provide default values for write_data and write_enable
    if (ports.count("write_data") && ports.count("write_enable")) {
      // Default write_enable to false (disable writing by default)
      auto falseVal = ctx.firrtlBuilder.create<ConstantOp>(
          txnModule.getLoc(), 
          ports["write_enable"].getType(),
          APSInt(APInt(1, 0), true));
      ctx.firrtlBuilder.create<ConnectOp>(txnModule.getLoc(), 
                                         ports["write_enable"], falseVal);
      
      // Default write_data to zero (value doesn't matter when write_enable is false)
      if (auto writeDataType = dyn_cast<UIntType>(ports["write_data"].getType())) {
        auto zeroVal = ctx.firrtlBuilder.create<ConstantOp>(
            txnModule.getLoc(),
            ports["write_data"].getType(),
            APSInt(APInt(writeDataType.getWidth().value_or(32), 0), true));
        ctx.firrtlBuilder.create<ConnectOp>(txnModule.getLoc(),
                                           ports["write_data"], zeroVal);
      }
    }
    
    // For other primitives, add similar default connections as needed
    // FIFO enqueue ports
    if (ports.count("enqueueEN")) {
      auto falseVal = ctx.firrtlBuilder.create<ConstantOp>(
          txnModule.getLoc(),
          ports["enqueueEN"].getType(), 
          APSInt(APInt(1, 0), true));
      ctx.firrtlBuilder.create<ConnectOp>(txnModule.getLoc(),
                                         ports["enqueueEN"], falseVal);
    }
    if (ports.count("enqueueOUT")) {
      if (auto enqueueDataType = dyn_cast<UIntType>(ports["enqueueOUT"].getType())) {
        auto zeroVal = ctx.firrtlBuilder.create<ConstantOp>(
            txnModule.getLoc(),
            ports["enqueueOUT"].getType(),
            APSInt(APInt(enqueueDataType.getWidth().value_or(32), 0), true));
        ctx.firrtlBuilder.create<ConnectOp>(txnModule.getLoc(),
                                           ports["enqueueOUT"], zeroVal);
      }
    }
    
    // FIFO dequeue ports  
    if (ports.count("dequeueEN")) {
      auto falseVal = ctx.firrtlBuilder.create<ConstantOp>(
          txnModule.getLoc(),
          ports["dequeueEN"].getType(),
          APSInt(APInt(1, 0), true));
      ctx.firrtlBuilder.create<ConnectOp>(txnModule.getLoc(),
                                         ports["dequeueEN"], falseVal);
    }
    
    // Register read_enable port (missed in initial implementation)
    if (ports.count("read_enable")) {
      auto falseVal = ctx.firrtlBuilder.create<ConstantOp>(
          txnModule.getLoc(),
          ports["read_enable"].getType(),
          APSInt(APInt(1, 0), true));
      ctx.firrtlBuilder.create<ConnectOp>(txnModule.getLoc(),
                                         ports["read_enable"], falseVal);
    }
    
    // Initialize all enable ports to false by default
    for (auto& [portName, portValue] : ports) {
      StringRef portNameStr = portName;
      // Skip ports that are already initialized
      if (portNameStr == "write_enable" || portNameStr == "enqueueEN" || 
          portNameStr == "dequeueEN" || portNameStr == "read_enable") {
        continue;
      }
      
      // Initialize all EN/enable ports to false
      if (portNameStr.ends_with("EN") || portNameStr.ends_with("_enable")) {
        auto falseVal = ctx.firrtlBuilder.create<ConstantOp>(
            txnModule.getLoc(),
            portValue.getType(),
            APSInt(APInt(1, 0), true));
        ctx.firrtlBuilder.create<ConnectOp>(txnModule.getLoc(),
                                           portValue, falseVal);
      }
      
      // Initialize all argument ports to zero (for method call arguments)
      if (portNameStr.contains("Arg_")) {
        if (auto portType = dyn_cast<UIntType>(portValue.getType())) {
          auto zeroVal = ctx.firrtlBuilder.create<ConstantOp>(
              txnModule.getLoc(),
              portValue.getType(),
              APSInt(APInt(portType.getWidth().value_or(32), 0), true));
          ctx.firrtlBuilder.create<ConnectOp>(txnModule.getLoc(),
                                             portValue, zeroVal);
        } else if (auto portType = dyn_cast<SIntType>(portValue.getType())) {
          auto zeroVal = ctx.firrtlBuilder.create<ConstantOp>(
              txnModule.getLoc(),
              portValue.getType(),
              APSInt(APInt(portType.getWidth().value_or(32), 0), false));
          ctx.firrtlBuilder.create<ConnectOp>(txnModule.getLoc(),
                                             portValue, zeroVal);
        }
      }
    }
  }
  
  // First step: Map all block arguments to FIRRTL ports for all action methods
  txnModule.walk([&](ActionMethodOp actionMethod) {
    StringRef prefix = actionMethod.getPrefix().value_or(actionMethod.getSymName());
    auto methodPortNames = firrtlModule.getPortNames();
    auto methodBlockArgs = firrtlModule.getBodyBlock()->getArguments();
    
    // Map method arguments to FIRRTL ports
    for (unsigned i = 0; i < actionMethod.getNumArguments(); ++i) {
      Value methodArg = actionMethod.getArgument(i);
      // Find corresponding FIRRTL input port using new naming convention
      std::string argName;
      if (auto argNameAttr = actionMethod.getArgAttrOfType<StringAttr>(i, "mlir.arg_name")) {
        argName = argNameAttr.getValue().str();
      } else {
        // Fallback to default name if no arg name attribute
        argName = "arg" + std::to_string(i);
      }
      std::string argPortName = (prefix + "Arg_" + argName).str();
      
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
  // In dynamic mode, we need to collect primitive information first,
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
    StringRef enablePostfix = "EN";
    
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
          // For action methods with args, we created ports with pattern: methodArg_argName
          std::string argName;
          if (auto argNameAttr = actionMethod.getArgAttrOfType<StringAttr>(i, "mlir.arg_name")) {
            argName = argNameAttr.getValue().str();
          } else {
            // Fallback to default name if no arg name attribute
            argName = "arg" + std::to_string(i);
          }
          std::string argPortName = (prefix + "Arg_" + argName).str();
          
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
    
    // Handle return value if action method has results
    auto funcType = actionMethod.getFunctionType();
    if (funcType.getNumResults() > 0) {
      // Find the return value in the method body
      Value returnValue;
      actionMethod.walk([&](ReturnOp ret) {
        if (ret.getNumOperands() > 0) {
          returnValue = ctx.txnToFirrtl.lookup(ret.getOperand(0));
        }
      });
      
      if (returnValue) {
        // Find the result output port
        std::string resultPortName = (prefix + "_result").str();
        auto portNames = firrtlModule.getPortNames();
        auto blockArgs = firrtlModule.getBodyBlock()->getArguments();
        for (size_t i = 0; i < portNames.size(); ++i) {
          if (cast<StringAttr>(portNames[i]).getValue() == resultPortName) {
            // Create wire to hold the result
            auto resultWire = ctx.firrtlBuilder.create<WireOp>(
                actionMethod.getLoc(), 
                blockArgs[i].getType(),
                (prefix + "_result_wire").str());
                
            // Initialize wire with default value to ensure it's fully initialized
            auto wireType = blockArgs[i].getType();
            if (auto uintType = dyn_cast<UIntType>(wireType)) {
              auto defaultVal = ctx.firrtlBuilder.create<ConstantOp>(
                  actionMethod.getLoc(),
                  wireType,
                  APSInt(APInt(uintType.getWidth().value_or(32), 0), true));
              ctx.firrtlBuilder.create<ConnectOp>(actionMethod.getLoc(),
                                                 resultWire.getResult(), defaultVal);
            } else if (auto sintType = dyn_cast<SIntType>(wireType)) {
              auto defaultVal = ctx.firrtlBuilder.create<ConstantOp>(
                  actionMethod.getLoc(),
                  wireType,
                  APSInt(APInt(sintType.getWidth().value_or(32), 0), false));
              ctx.firrtlBuilder.create<ConnectOp>(actionMethod.getLoc(),
                                                 resultWire.getResult(), defaultVal);
            }
                
            // Connect return value to wire when method fires
            ctx.firrtlBuilder.create<WhenOp>(actionMethod.getLoc(), wf, false, [&]() {
              ctx.firrtlBuilder.create<ConnectOp>(actionMethod.getLoc(), 
                                                 resultWire.getResult(), returnValue);
            });
            
            // Always connect wire to output port
            ctx.firrtlBuilder.create<ConnectOp>(actionMethod.getLoc(), 
                                               blockArgs[i], resultWire.getResult());
            break;
          }
        }
      }
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
        // No longer collecting reachability conditions in context
        // ReachabilityAnalysis pass attaches conditions directly to operations
        // TODO: Process conditions attached by ReachabilityAnalysis pass
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
    
    // Use top module name as circuit name for FIRRTL toolchain compatibility
    auto firrtlCircuit = builder.create<::circt::firrtl::CircuitOp>(
        module.getLoc(), builder.getStringAttr(topModuleName));
    
    // Convert each module bottom-up
    for (auto txnModule : sortedModules) {
      // Only convert Txn modules, not the top-level builtin module
      if (!isa<::sharp::txn::ModuleOp>(txnModule)) continue;
      if (txnModule.getName() == "firrtl_generated") continue;
      
      // Check if this is the top-level module
      bool isTopLevel = (txnModule.getName() == topModuleName);
      
      // Use will-fire mode from command-line option (not module attribute)
      std::string currentWillFireMode = this->willFireMode;
      
      
      if (failed(convertModule(txnModule, convCtx, firrtlCircuit, currentWillFireMode, isTopLevel))) {
        signalPassFailure();
        return;
      }
      ++numModulesConverted;
    }
    
    // Remove txn.primitive operations from entire module (firtool cannot handle them)
    SmallVector<::sharp::txn::PrimitiveOp> primitivesToErase;
    module.walk([&](::sharp::txn::PrimitiveOp primitiveOp) {
      primitivesToErase.push_back(primitiveOp);
    });
    for (auto primitiveOp : primitivesToErase) {
      primitiveOp.erase();
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