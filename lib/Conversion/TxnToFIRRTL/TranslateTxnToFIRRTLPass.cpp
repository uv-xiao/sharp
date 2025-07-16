//===- TranslateTxnToFIRRTLPass.cpp - Translate Txn to FIRRTL Modules ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the second pass of the two-phase Txn-to-FIRRTL conversion.
// It converts the Txn module structure to FIRRTL modules and circuits, generating
// the necessary will-fire logic and merging the converted method and rule bodies
// into a complete FIRRTL module.
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
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "sharp/Conversion/Passes.h"
#include "sharp/Dialect/Txn/TxnDialect.h"
#include "sharp/Dialect/Txn/TxnOps.h"
#include "sharp/Dialect/Txn/TxnTypes.h"
#include "sharp/Analysis/AnalysisError.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "translate-txn-to-firrtl"

namespace mlir {
namespace sharp {

#define GEN_PASS_DEF_TRANSLATETXNTOFIRRTLPASS
#include "sharp/Conversion/Passes.h.inc"

namespace {

using namespace ::sharp::txn;
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

//===----------------------------------------------------------------------===//
// Conversion State
//===----------------------------------------------------------------------===//

struct ConversionState {
  /// Value mapping from Txn to FIRRTL
  IRMapping valueMapping;
  
  /// Will-fire signals for all actions
  llvm::StringMap<Value> willFireSignals;
  
  /// Method ports for current module
  DenseMap<StringRef, SmallVector<Value>> methodPorts;
  
  /// Current FIRRTL module and builder
  FModuleOp currentFIRRTLModule;
  OpBuilder firrtlBuilder;
  
  ConversionState(MLIRContext *ctx) : firrtlBuilder(ctx) {}
};

//===----------------------------------------------------------------------===//
// Port Generation Utilities
//===----------------------------------------------------------------------===//

/// Generate FIRRTL module ports from Txn module
static void generateModulePorts(::sharp::txn::ModuleOp txnModule, 
                               SmallVectorImpl<PortInfo> &ports) {
  MLIRContext *ctx = txnModule.getContext();
  
  // Add clock and reset ports
  ports.push_back({StringAttr::get(ctx, "clk"), 
                   ::circt::firrtl::ClockType::get(ctx), 
                   Direction::In});
  ports.push_back({StringAttr::get(ctx, "rst"), 
                   ::circt::firrtl::ResetType::get(ctx), 
                   Direction::In});
  
  // Add EN ports for all action methods and rules
  for (auto actionOp : txnModule.getOps<ActionMethodOp>()) {
    StringRef actionName = actionOp.getName();
    ports.push_back({StringAttr::get(ctx, "EN_" + actionName), 
                     ::circt::firrtl::UIntType::get(ctx, 1), 
                     Direction::In});
  }
  
  for (auto ruleOp : txnModule.getOps<RuleOp>()) {
    StringRef ruleName = ruleOp.getName();
    ports.push_back({StringAttr::get(ctx, "EN_" + ruleName), 
                     ::circt::firrtl::UIntType::get(ctx, 1), 
                     Direction::In});
  }
  
  // Add method interface ports
  for (auto valueOp : txnModule.getOps<ValueMethodOp>()) {
    StringRef methodName = valueOp.getName();
    
    // Add input ports for method arguments
    for (auto arg : valueOp.getArguments()) {
      std::string portName = methodName.str() + "_" + std::to_string(arg.getArgNumber());
      // Types should already be converted to FIRRTL types by the first pass
      auto firrtlType = dyn_cast<FIRRTLType>(arg.getType());
      if (!firrtlType) {
        firrtlType = convertType(arg.getType()); // Fallback conversion
      }
      ports.push_back({StringAttr::get(ctx, portName), 
                       firrtlType, 
                       Direction::In});
    }
    
    // Add output port for method result
    if (valueOp.getNumResults() > 0) {
      Type resultType = valueOp.getResultTypes()[0];
      auto firrtlType = dyn_cast<FIRRTLType>(resultType);
      if (!firrtlType) {
        firrtlType = convertType(resultType); // Fallback conversion
      }
      ports.push_back({StringAttr::get(ctx, methodName.str() + "_result"), 
                       firrtlType, 
                       Direction::Out});
    }
  }
  
  for (auto actionOp : txnModule.getOps<ActionMethodOp>()) {
    StringRef methodName = actionOp.getName();
    
    // Add input ports for method arguments
    for (auto arg : actionOp.getArguments()) {
      std::string portName = methodName.str() + "_" + std::to_string(arg.getArgNumber());
      // Types should already be converted to FIRRTL types by the first pass
      auto firrtlType = dyn_cast<FIRRTLType>(arg.getType());
      if (!firrtlType) {
        firrtlType = convertType(arg.getType()); // Fallback conversion
      }
      ports.push_back({StringAttr::get(ctx, portName), 
                       firrtlType, 
                       Direction::In});
    }
  }
}

//===----------------------------------------------------------------------===//
// Will-Fire Logic Generation
//===----------------------------------------------------------------------===//

/// Generate static will-fire logic
static LogicalResult generateStaticWillFire(::sharp::txn::ModuleOp txnModule, 
                                           ConversionState &state) {
  LLVM_DEBUG(llvm::dbgs() << "Generating static will-fire logic\n");
  
  // For static mode, will-fire is simply EN signal
  // (conflict resolution is handled by external scheduler)
  
  // Generate will-fire for action methods
  for (auto actionOp : txnModule.getOps<ActionMethodOp>()) {
    StringRef actionName = actionOp.getName();
    
    // Find EN port
    Value enPort = nullptr;
    for (auto [index, port] : llvm::enumerate(state.currentFIRRTLModule.getArguments())) {
      if (state.currentFIRRTLModule.getPortName(index).str() == "EN_" + actionName.str()) {
        enPort = port;
        break;
      }
    }
    
    if (!enPort) {
      return actionOp.emitError("Could not find EN port for action ") << actionName;
    }
    
    state.willFireSignals[actionName] = enPort;
    LLVM_DEBUG(llvm::dbgs() << "Static will-fire for action " << actionName << " = EN signal\n");
  }
  
  // Generate will-fire for rules
  for (auto ruleOp : txnModule.getOps<RuleOp>()) {
    StringRef ruleName = ruleOp.getName();
    
    // Find EN port
    Value enPort = nullptr;
    for (auto [index, port] : llvm::enumerate(state.currentFIRRTLModule.getArguments())) {
      if (state.currentFIRRTLModule.getPortName(index).str() == "EN_" + ruleName.str()) {
        enPort = port;
        break;
      }
    }
    
    if (!enPort) {
      return ruleOp.emitError("Could not find EN port for rule ") << ruleName;
    }
    
    state.willFireSignals[ruleName] = enPort;
    LLVM_DEBUG(llvm::dbgs() << "Static will-fire for rule " << ruleName << " = EN signal\n");
  }
  
  return success();
}

/// Generate dynamic will-fire logic
static LogicalResult generateDynamicWillFire(::sharp::txn::ModuleOp txnModule, 
                                            ConversionState &state) {
  LLVM_DEBUG(llvm::dbgs() << "Generating dynamic will-fire logic\n");
  
  // For dynamic mode, will-fire includes conflict detection
  // This is a simplified implementation - full implementation would use
  // conflict matrices and reachability analysis
  
  // Generate will-fire for action methods
  for (auto actionOp : txnModule.getOps<ActionMethodOp>()) {
    StringRef actionName = actionOp.getName();
    
    // Find EN port
    Value enPort = nullptr;
    for (auto [index, port] : llvm::enumerate(state.currentFIRRTLModule.getArguments())) {
      if (state.currentFIRRTLModule.getPortName(index).str() == "EN_" + actionName.str()) {
        enPort = port;
        break;
      }
    }
    
    if (!enPort) {
      return actionOp.emitError("Could not find EN port for action ") << actionName;
    }
    
    // For now, dynamic will-fire is same as static
    // TODO: Add proper conflict detection logic
    state.willFireSignals[actionName] = enPort;
    LLVM_DEBUG(llvm::dbgs() << "Dynamic will-fire for action " << actionName << " = EN signal (simplified)\n");
  }
  
  // Generate will-fire for rules
  for (auto ruleOp : txnModule.getOps<RuleOp>()) {
    StringRef ruleName = ruleOp.getName();
    
    // Find EN port
    Value enPort = nullptr;
    for (auto [index, port] : llvm::enumerate(state.currentFIRRTLModule.getArguments())) {
      if (state.currentFIRRTLModule.getPortName(index).str() == "EN_" + ruleName.str()) {
        enPort = port;
        break;
      }
    }
    
    if (!enPort) {
      return ruleOp.emitError("Could not find EN port for rule ") << ruleName;
    }
    
    // For now, dynamic will-fire is same as static
    // TODO: Add proper conflict detection logic
    state.willFireSignals[ruleName] = enPort;
    LLVM_DEBUG(llvm::dbgs() << "Dynamic will-fire for rule " << ruleName << " = EN signal (simplified)\n");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Module Conversion
//===----------------------------------------------------------------------===//

/// Convert a Txn module to FIRRTL module
static LogicalResult convertTxnModule(::sharp::txn::ModuleOp txnModule, 
                                     ConversionState &state,
                                     StringRef willFireMode) {
  LLVM_DEBUG(llvm::dbgs() << "Converting Txn module " << txnModule.getName() << "\n");
  
  // Generate module ports
  SmallVector<PortInfo> ports;
  generateModulePorts(txnModule, ports);
  
  // Create FIRRTL module
  auto firrtlModule = state.firrtlBuilder.create<FModuleOp>(
      txnModule.getLoc(), 
      StringAttr::get(txnModule.getContext(), txnModule.getName()),
      ConventionAttr::get(txnModule.getContext(), Convention::Internal),
      ports);
  
  state.currentFIRRTLModule = firrtlModule;
  state.firrtlBuilder.setInsertionPointToStart(firrtlModule.getBodyBlock());
  
  // Map module block arguments
  for (auto [txnArg, firArg] : llvm::zip(txnModule.getBody().getArguments(), 
                                          firrtlModule.getBodyBlock()->getArguments())) {
    state.valueMapping.map(txnArg, firArg);
  }
  
  // Map instance values - create FIRRTL instances for each Txn instance
  for (auto instanceOp : txnModule.getOps<::sharp::txn::InstanceOp>()) {
    // For now, just create a placeholder wire to represent the instance
    // The actual instance conversion will be handled later
    auto wireType = UIntType::get(instanceOp.getContext(), 1);
    auto wire = state.firrtlBuilder.create<WireOp>(
        instanceOp.getLoc(), wireType, 
        StringAttr::get(instanceOp.getContext(), instanceOp.getSymName()));
    state.valueMapping.map(instanceOp.getResult(), wire.getResult());
  }
  
  // Generate will-fire logic
  if (willFireMode == "static") {
    if (failed(generateStaticWillFire(txnModule, state))) {
      return failure();
    }
  } else if (willFireMode == "dynamic") {
    if (failed(generateDynamicWillFire(txnModule, state))) {
      return failure();
    }
  } else {
    return txnModule.emitError("Unknown will-fire mode: ") << willFireMode;
  }
  
  // Build a port lookup map for efficient access
  DenseMap<StringRef, Value> portLookup;
  for (auto [idx, port] : llvm::enumerate(firrtlModule.getArguments())) {
    portLookup[firrtlModule.getPortName(idx)] = port;
  }

  // Clone operations from methods and rules
  for (auto valueOp : txnModule.getOps<ValueMethodOp>()) {
    // Map method arguments to module ports
    for (auto [idx, arg] : llvm::enumerate(valueOp.getArguments())) {
      // Find the corresponding port in the FIRRTL module
      std::string portName = valueOp.getName().str() + "_" + std::to_string(idx);
      auto portIt = portLookup.find(portName);
      if (portIt != portLookup.end()) {
        state.valueMapping.map(arg, portIt->second);
      } else {
        return valueOp.emitError("Could not find port for argument ") << idx 
               << " of value method " << valueOp.getName() 
               << " (expected port name: " << portName << ")";
      }
    }
    
    // Find the output port for this value method
    Value outputPort = nullptr;
    if (valueOp.getNumResults() > 0) {
      std::string resultPortName = valueOp.getName().str() + "_result";
      auto portIt = portLookup.find(resultPortName);
      if (portIt != portLookup.end()) {
        outputPort = portIt->second;
      } else {
        return valueOp.emitError("Could not find output port for value method ") 
               << valueOp.getName() << " (expected port name: " << resultPortName << ")";
      }
    }
    
    // Clone the method body into the FIRRTL module
    // The operations inside should already be converted to FIRRTL by the first pass
    for (auto &op : valueOp.getBody().getOps()) {
      if (auto returnOp = dyn_cast<ReturnOp>(op)) {
        // Connect the return value to the output port
        if (outputPort && returnOp.getNumOperands() > 0) {
          Value returnValue = state.valueMapping.lookup(returnOp.getOperand(0));
          if (!returnValue) {
            return returnOp.emitError("Return value not found in value mapping");
          }
          state.firrtlBuilder.create<ConnectOp>(
              returnOp.getLoc(), outputPort, returnValue);
        }
      } else if (auto callOp = dyn_cast<CallOp>(op)) {
        // Convert txn.call to appropriate FIRRTL operations
        // For now, create a wire to represent the call result
        if (callOp.getNumResults() > 0) {
          auto resultType = callOp.getResult(0).getType();
          auto wire = state.firrtlBuilder.create<WireOp>(
              callOp.getLoc(), resultType,
              StringAttr::get(callOp.getContext(), "call_result"));
          state.valueMapping.map(callOp.getResult(0), wire.getResult());
        }
        // TODO: Properly connect instance ports
      } else {
        state.firrtlBuilder.clone(op, state.valueMapping);
      }
    }
  }
  
  for (auto actionOp : txnModule.getOps<ActionMethodOp>()) {
    StringRef actionName = actionOp.getName();
    Value willFire = state.willFireSignals[actionName];
    
    // Map method arguments to module ports
    for (auto [idx, arg] : llvm::enumerate(actionOp.getArguments())) {
      // Find the corresponding port in the FIRRTL module
      std::string portName = actionName.str() + "_" + std::to_string(idx);
      auto portIt = portLookup.find(portName);
      if (portIt != portLookup.end()) {
        state.valueMapping.map(arg, portIt->second);
      } else {
        return actionOp.emitError("Could not find port for argument ") << idx 
               << " of action method " << actionName 
               << " (expected port name: " << portName << ")";
      }
    }
    
    // Create conditional execution based on will-fire
    auto whenOp = state.firrtlBuilder.create<WhenOp>(
        actionOp.getLoc(), willFire, /*withElseRegion=*/false);
    
    state.firrtlBuilder.setInsertionPointToStart(&whenOp.getThenRegion().front());
    
    // Clone the action body
    for (auto &op : actionOp.getBody().getOps()) {
      if (isa<ReturnOp>(op)) {
        // Skip return operations
        continue;
      } else if (auto callOp = dyn_cast<CallOp>(op)) {
        // Convert txn.call to appropriate FIRRTL operations
        // For action methods calling other actions, we need to handle this differently
        // For now, skip the call since action effects will be handled by will-fire logic
        continue;
      } else {
        state.firrtlBuilder.clone(op, state.valueMapping);
      }
    }
    
    state.firrtlBuilder.setInsertionPointAfter(whenOp);
  }
  
  for (auto ruleOp : txnModule.getOps<RuleOp>()) {
    StringRef ruleName = ruleOp.getName();
    Value willFire = state.willFireSignals[ruleName];
    
    // Create conditional execution based on will-fire
    auto whenOp = state.firrtlBuilder.create<WhenOp>(
        ruleOp.getLoc(), willFire, /*withElseRegion=*/false);
    
    state.firrtlBuilder.setInsertionPointToStart(&whenOp.getThenRegion().front());
    
    // Clone the rule body
    for (auto &op : ruleOp.getBody().getOps()) {
      if (auto callOp = dyn_cast<CallOp>(op)) {
        // Skip txn.call operations in rules for now
        // Rules typically call action methods which will be handled by will-fire
        continue;
      } else {
        state.firrtlBuilder.clone(op, state.valueMapping);
      }
    }
    
    state.firrtlBuilder.setInsertionPointAfter(whenOp);
  }
  
  // Reset insertion point to end of circuit after this module
  state.firrtlBuilder.setInsertionPointToEnd(&state.currentFIRRTLModule->getParentOp()->getRegion(0).front());
  
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

class TranslateTxnToFIRRTLPass : public impl::TranslateTxnToFIRRTLPassBase<TranslateTxnToFIRRTLPass> {
public:
  using impl::TranslateTxnToFIRRTLPassBase<TranslateTxnToFIRRTLPass>::TranslateTxnToFIRRTLPassBase;
  
  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "Running TranslateTxnToFIRRTLPass\n");
    
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    ConversionState state(ctx);
    
    // Collect all Txn operations first
    SmallVector<::sharp::txn::ModuleOp> txnModules;
    SmallVector<::sharp::txn::PrimitiveOp> txnPrimitives;
    
    module.walk([&](::sharp::txn::ModuleOp txnModule) {
      txnModules.push_back(txnModule);
    });
    
    module.walk([&](::sharp::txn::PrimitiveOp txnPrimitive) {
      txnPrimitives.push_back(txnPrimitive);
    });
    
    // Find the top module to name the circuit
    StringRef circuitName = "MainCircuit";
    for (auto txnModule : txnModules) {
      if (txnModule->hasAttr("top")) {
        circuitName = txnModule.getName();
        break;
      }
    }
    
    // Create circuit at the beginning of the module
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(module.getBody());
    auto circuit = builder.create<CircuitOp>(
        module.getLoc(), StringAttr::get(ctx, circuitName));
    
    state.firrtlBuilder.setInsertionPointToStart(circuit.getBodyBlock());
    
    // Convert all modules
    for (auto txnModule : txnModules) {
      if (failed(convertTxnModule(txnModule, state, willFireMode))) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to convert Txn module " << txnModule.getName() << "\n");
        signalPassFailure();
        return;
      }
    }
    
    // Reset insertion point to circuit level for primitive modules
    state.firrtlBuilder.setInsertionPointToEnd(circuit.getBodyBlock());
    
    // For now, just create placeholder FIRRTL modules for primitives
    for (auto txnPrimitive : txnPrimitives) {
      LLVM_DEBUG(llvm::dbgs() << "Creating Register primitive: " << txnPrimitive.getName() 
                              << " with dataType: " << txnPrimitive.getTypeAttr() << "\n");
      
      // Create a simple FIRRTL module for the primitive
      SmallVector<PortInfo> primPorts;
      primPorts.push_back({StringAttr::get(ctx, "clk"), 
                           ::circt::firrtl::ClockType::get(ctx), 
                           Direction::In});
      primPorts.push_back({StringAttr::get(ctx, "rst"), 
                           ::circt::firrtl::ResetType::get(ctx), 
                           Direction::In});
      
      // Add ports based on primitive methods
      for (auto &op : txnPrimitive.getBody().front()) {
        if (auto methodOp = dyn_cast<::sharp::txn::FirValueMethodOp>(&op)) {
          // Add output port for value method
          auto funcType = cast<FunctionType>(methodOp.getFunctionTypeAttr().getValue());
          if (funcType.getNumResults() > 0) {
            auto resultType = funcType.getResult(0);
            auto firrtlType = dyn_cast<FIRRTLType>(resultType);
            if (firrtlType) {
              primPorts.push_back({methodOp.getNameAttr(), 
                                   firrtlType, 
                                   Direction::Out});
            }
          }
        } else if (auto methodOp = dyn_cast<::sharp::txn::FirActionMethodOp>(&op)) {
          // Add input ports for action method
          auto funcType = cast<FunctionType>(methodOp.getFunctionTypeAttr().getValue());
          for (size_t i = 0; i < funcType.getNumInputs(); i++) {
            auto inputType = funcType.getInput(i);
            auto firrtlType = dyn_cast<FIRRTLType>(inputType);
            if (firrtlType) {
              primPorts.push_back({StringAttr::get(ctx, methodOp.getNameAttr().getValue().str() + "_" + std::to_string(i)), 
                                   firrtlType, 
                                   Direction::In});
            }
          }
          // Add enable port for action method
          primPorts.push_back({StringAttr::get(ctx, methodOp.getNameAttr().getValue().str() + "_en"), 
                               ::circt::firrtl::UIntType::get(ctx, 1), 
                               Direction::In});
        }
      }
      
      state.firrtlBuilder.create<FModuleOp>(
          txnPrimitive.getLoc(), 
          StringAttr::get(ctx, txnPrimitive.getName()),
          ConventionAttr::get(ctx, Convention::Internal),
          primPorts);
      
      LLVM_DEBUG(llvm::dbgs() << "Created Register primitive successfully\n");
    }
    
    // Don't erase operations - let the pass manager handle cleanup
    // The FIRRTL circuit is now created alongside the Txn operations
    
    LLVM_DEBUG(llvm::dbgs() << "TranslateTxnToFIRRTLPass completed successfully\n");
  }
};

} // end anonymous namespace

// Pass creation function is generated by tablegen

} // end namespace sharp
} // end namespace mlir