//===- PrimitiveGen.cpp - Primitive Generation Pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the primitive generation pass for Sharp Txn dialect.
//
//===----------------------------------------------------------------------===//

#include "sharp/Analysis/Passes.h"
#include "sharp/Dialect/Txn/TxnDialect.h"
#include "sharp/Dialect/Txn/TxnOps.h"
#include "sharp/Dialect/Txn/TxnPrimitives.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include <set>

#define DEBUG_TYPE "sharp-primitive-gen"

namespace mlir {
namespace sharp {

#define GEN_PASS_DEF_PRIMITIVEGEN
#include "sharp/Analysis/Passes.h.inc"

using namespace ::sharp::txn;

namespace {

class PrimitiveGenPass : public impl::PrimitiveGenBase<PrimitiveGenPass> {
public:
  void runOnOperation() override;

private:
  /// Generate missing primitives for a module
  void generateMissingPrimitives(ModuleOp module);
  
  /// Check if a primitive already exists
  bool primitiveExists(ModuleOp module, StringRef primitiveName);
  
  /// Generate a primitive based on its name and type
  void generatePrimitive(ModuleOp module, StringRef primitiveName, Type dataType);
  
  /// Parse primitive name to extract base name and type
  std::pair<std::string, Type> parsePrimitiveName(StringRef primitiveName, MLIRContext *ctx);
};

void PrimitiveGenPass::runOnOperation() {
  ModuleOp module = getOperation();
  generateMissingPrimitives(module);
}

void PrimitiveGenPass::generateMissingPrimitives(ModuleOp module) {
  std::set<std::string> neededPrimitives;
  
  // Collect all primitive instances
  module.walk([&](::sharp::txn::InstanceOp instanceOp) {
    auto moduleNameAttr = instanceOp.getModuleNameAttr();
    auto moduleName = moduleNameAttr.getValue();
    auto typeArgs = instanceOp.getTypeArguments();
    
    LLVM_DEBUG(llvm::dbgs() << "Found instance: " << moduleName);
    if (typeArgs) {
      LLVM_DEBUG(llvm::dbgs() << " with type args");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
    
    // Check if this is a parametric primitive (has type arguments)
    if (typeArgs && !typeArgs->empty()) {
      // Reconstruct the full primitive name like "Register<i1>"
      std::string fullName = moduleName.str() + "<";
      for (size_t i = 0; i < typeArgs->size(); ++i) {
        if (i > 0) fullName += ",";
        if (auto typeAttr = dyn_cast<TypeAttr>((*typeArgs)[i])) {
          // Simple type name extraction - could be improved
          auto type = typeAttr.getValue();
          if (auto intType = dyn_cast<IntegerType>(type)) {
            fullName += "i" + std::to_string(intType.getWidth());
          } else {
            fullName += "unknown";
          }
        }
      }
      fullName += ">";
      
      LLVM_DEBUG(llvm::dbgs() << "Adding parametric primitive: " << fullName << "\n");
      neededPrimitives.insert(fullName);
    }
  });
  
  // Generate missing primitives
  for (const auto &primitiveName : neededPrimitives) {
    if (!primitiveExists(module, primitiveName)) {
      LLVM_DEBUG(llvm::dbgs() << "Generating primitive: " << primitiveName << "\n");
      
      auto [baseName, dataType] = parsePrimitiveName(primitiveName, module.getContext());
      if (dataType) {
        generatePrimitive(module, primitiveName, dataType);
      }
    }
  }
}

bool PrimitiveGenPass::primitiveExists(ModuleOp module, StringRef primitiveName) {
  bool exists = false;
  module.walk([&](::sharp::txn::ModuleOp txnModule) {
    if (txnModule.getName() == primitiveName) {
      exists = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  // Also check for primitive operations
  module.walk([&](::sharp::txn::PrimitiveOp primitiveOp) {
    if (primitiveOp.getName() == primitiveName) {
      exists = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  return exists;
}

void PrimitiveGenPass::generatePrimitive(ModuleOp module, StringRef primitiveName, Type dataType) {
  auto [baseName, _] = parsePrimitiveName(primitiveName, module.getContext());
  
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  
  auto loc = module.getLoc();
  
  if (baseName == "Register") {
    txn::createRegisterPrimitive(builder, loc, primitiveName, dataType);
  } else if (baseName == "Wire") {
    txn::createWirePrimitive(builder, loc, primitiveName, dataType);
  } else if (baseName == "FIFO") {
    txn::createFIFOPrimitive(builder, loc, primitiveName, dataType);
  } else if (baseName == "Memory") {
    txn::createMemoryPrimitive(builder, loc, primitiveName, dataType);
  } else if (baseName == "SpecFIFO") {
    txn::createSpecFIFOPrimitive(builder, loc, primitiveName, dataType);
  } else if (baseName == "SpecMemory") {
    txn::createSpecMemoryPrimitive(builder, loc, primitiveName, dataType);
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Unknown primitive type: " << baseName << "\n");
  }
}

std::pair<std::string, Type> PrimitiveGenPass::parsePrimitiveName(StringRef primitiveName, MLIRContext *ctx) {
  // Parse names like "Register<i32>", "FIFO<i1>", etc.
  size_t anglePos = primitiveName.find('<');
  if (anglePos == StringRef::npos) {
    return {"", nullptr};
  }
  
  std::string baseName = primitiveName.substr(0, anglePos).str();
  
  // Extract type string
  size_t endPos = primitiveName.find('>', anglePos);
  if (endPos == StringRef::npos) {
    return {"", nullptr};
  }
  
  StringRef typeStr = primitiveName.substr(anglePos + 1, endPos - anglePos - 1);
  
  // Parse common types
  Type dataType = nullptr;
  if (typeStr == "i1") {
    dataType = IntegerType::get(ctx, 1);
  } else if (typeStr == "i8") {
    dataType = IntegerType::get(ctx, 8);
  } else if (typeStr == "i16") {
    dataType = IntegerType::get(ctx, 16);
  } else if (typeStr == "i32") {
    dataType = IntegerType::get(ctx, 32);
  } else if (typeStr == "i64") {
    dataType = IntegerType::get(ctx, 64);
  } else if (typeStr.starts_with("i")) {
    // Try to parse as integer width
    unsigned width;
    if (!typeStr.substr(1).getAsInteger(10, width)) {
      dataType = IntegerType::get(ctx, width);
    }
  }
  
  return {baseName, dataType};
}

} // namespace

std::unique_ptr<mlir::Pass> createPrimitiveGenPass() {
  return std::make_unique<PrimitiveGenPass>();
}

} // namespace sharp
} // namespace mlir