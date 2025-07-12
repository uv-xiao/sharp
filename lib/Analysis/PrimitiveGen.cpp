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

#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include <set>
#include <map>

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
  bool primitiveExists(ModuleOp module, StringRef fullName);
  
  /// Generate a primitive based on its base name and type arguments
  void generatePrimitive(ModuleOp module, StringRef baseName, ArrayAttr typeArgs);
  
  /// Parse primitive name to extract base name and type
  std::pair<std::string, Type> parsePrimitiveName(StringRef primitiveName, MLIRContext *ctx);
};

void PrimitiveGenPass::runOnOperation() {
  ModuleOp module = getOperation();
  
  // Report pass execution
  LLVM_DEBUG(llvm::dbgs() << "[PrimitiveGen] Starting primitive generation pass\n");
  
  generateMissingPrimitives(module);
  
  // Mark module as having completed primitive generation
  module->setAttr("sharp.primitive_gen_complete", 
                  UnitAttr::get(module.getContext()));
  
  LLVM_DEBUG(llvm::dbgs() << "[PrimitiveGen] Primitive generation completed successfully\n");
}

void PrimitiveGenPass::generateMissingPrimitives(ModuleOp module) {
  struct PrimitiveRequest {
    std::string baseName;
    ArrayAttr typeArgs;
    std::string fullName;
  };
  
  std::map<std::string, PrimitiveRequest> neededPrimitives; // Use map to deduplicate
  
  // Collect all primitive instances
  module.walk([&](::sharp::txn::InstanceOp instanceOp) {
    auto moduleNameAttr = instanceOp.getModuleNameAttr();
    auto baseName = moduleNameAttr.getValue().str();
    auto typeArgs = instanceOp.getTypeArguments();
    
    LLVM_DEBUG(llvm::dbgs() << "Found instance: " << baseName);
    if (typeArgs) {
      LLVM_DEBUG(llvm::dbgs() << " with type args");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
    
    // Check if this is a parametric primitive (has type arguments)
    if (typeArgs && !typeArgs->empty()) {
      // Generate full name for checking existence
      std::string fullName = module_name_with_type_args(baseName, typeArgs.value());
      
      LLVM_DEBUG(llvm::dbgs() << "Adding parametric primitive: " << fullName << "\n");
      neededPrimitives[fullName] = {baseName, typeArgs.value(), fullName};
    }
  });
  
  // Generate missing primitives
  for (const auto &[fullName, request] : neededPrimitives) {
    LLVM_DEBUG(llvm::dbgs() << "Checking if primitive exists: " << request.fullName << "\n");
    if (!primitiveExists(module, request.fullName)) {
      LLVM_DEBUG(llvm::dbgs() << "Generating primitive: " << request.fullName << "\n");
      generatePrimitive(module, request.baseName, request.typeArgs);
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Primitive already exists: " << request.fullName << "\n");
    }
  }
}

bool PrimitiveGenPass::primitiveExists(ModuleOp module, StringRef fullName) {
  bool exists = false;
  module.walk([&](::sharp::txn::ModuleOp txnModule) {
    if (txnModule.getFullName() == fullName) {
      exists = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  // Also check for primitive operations
  module.walk([&](::sharp::txn::PrimitiveOp primitiveOp) {
    if (primitiveOp.getFullName() == fullName) {
      exists = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  return exists;
}

void PrimitiveGenPass::generatePrimitive(ModuleOp module, StringRef baseName, ArrayAttr typeArgs) {
  LLVM_DEBUG(llvm::dbgs() << "Generating primitive: base=" << baseName << "\n");
  
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  
  auto loc = module.getLoc();
  
  // Extract the primary type argument (first type for single-type primitives)
  Type dataType = nullptr;
  if (typeArgs && !typeArgs.empty()) {
    if (auto typeAttr = dyn_cast<TypeAttr>(typeArgs[0])) {
      dataType = typeAttr.getValue();
    }
  }
  
  // Generate full name for the primitive
  std::string fullName = module_name_with_type_args(baseName, typeArgs);
  
  if (baseName == "Register") {
    LLVM_DEBUG(llvm::dbgs() << "Creating Register primitive with type: " << (dataType ? "valid" : "null") << "\n");
    txn::createRegisterPrimitive(builder, loc, fullName, dataType);
  } else if (baseName == "Wire") {
    txn::createWirePrimitive(builder, loc, fullName, dataType);
  } else if (baseName == "FIFO") {
    txn::createFIFOPrimitive(builder, loc, fullName, dataType);
  } else if (baseName == "Memory") {
    txn::createMemoryPrimitive(builder, loc, fullName, dataType);
  } else if (baseName == "SpecFIFO") {
    txn::createSpecFIFOPrimitive(builder, loc, fullName, dataType);
  } else if (baseName == "SpecMemory") {
    txn::createSpecMemoryPrimitive(builder, loc, fullName, dataType);
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
  } else if (typeStr.starts_with("!firrtl.uint<") && typeStr.ends_with(">")) {
    // Parse FIRRTL uint types like "!firrtl.uint<32>"
    StringRef widthStr = typeStr.substr(13, typeStr.size() - 14); // Remove "!firrtl.uint<" and ">"
    unsigned width;
    if (!widthStr.getAsInteger(10, width)) {
      dataType = circt::firrtl::UIntType::get(ctx, width);
    }
  } else if (typeStr.starts_with("!firrtl.sint<") && typeStr.ends_with(">")) {
    // Parse FIRRTL sint types like "!firrtl.sint<32>"
    StringRef widthStr = typeStr.substr(13, typeStr.size() - 14); // Remove "!firrtl.sint<" and ">"
    unsigned width;
    if (!widthStr.getAsInteger(10, width)) {
      dataType = circt::firrtl::SIntType::get(ctx, width);
    }
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