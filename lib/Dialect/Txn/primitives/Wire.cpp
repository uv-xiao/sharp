//===- Wire.cpp - Sharp Txn Wire primitive implementation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Wire primitive for the Sharp Txn dialect.
// Wires are combinational connections between components.
//
//===----------------------------------------------------------------------===//

#include "sharp/Dialect/Txn/TxnPrimitives.h"
#include "sharp/Dialect/Txn/TxnOps.h"
#include "sharp/Dialect/Txn/TxnTypes.h"

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace sharp {
namespace txn {

::sharp::txn::PrimitiveOp createWirePrimitive(OpBuilder &builder, Location loc,
                                             StringRef name, Type dataType) {
  // Create interface type for the primitive
  auto moduleType = ::sharp::txn::ModuleType::get(builder.getContext(),
                                                  StringAttr::get(builder.getContext(), name));
  
  // Create the primitive operation
  auto primitive = builder.create<::sharp::txn::PrimitiveOp>(loc,
                                                            StringAttr::get(builder.getContext(), name),
                                                            builder.getStringAttr("hw"),
                                                            TypeAttr::get(moduleType));
  
  // Create a new builder for the primitive body
  OpBuilder::InsertionGuard guard(builder);
  Block *body = &primitive.getBody().emplaceBlock();
  builder.setInsertionPointToStart(body);
  
  // Define the FIRRTL interface methods
  auto readType = builder.getFunctionType({}, {dataType});
  auto writeType = builder.getFunctionType({dataType}, {});
  
  // Create read method declaration (value method)
  // The implementation would connect to the FIRRTL module's read_data port
  auto readMethod = builder.create<::sharp::txn::FirValueMethodOp>(loc, "read", readType);
  readMethod->setAttr("firrtl.port", builder.getStringAttr("read_data"));
  
  // Create write method declaration (action method)
  // The implementation would connect to write_data and assert write_enable
  auto writeMethod = builder.create<::sharp::txn::FirActionMethodOp>(loc, "write", writeType);
  writeMethod->setAttr("firrtl.data_port", builder.getStringAttr("write_data"));
  writeMethod->setAttr("firrtl.enable_port", builder.getStringAttr("write_enable"));
  
  // Add default clock and reset (wires still need these for FIRRTL)
  builder.create<::sharp::txn::ClockByOp>(loc, SymbolRefAttr::get(builder.getContext(), "clk"));
  builder.create<::sharp::txn::ResetByOp>(loc, SymbolRefAttr::get(builder.getContext(), "rst"));
  
  // Add reference to the FIRRTL module implementation
  primitive->setAttr("firrtl.impl", builder.getStringAttr(name.str() + "_impl"));
  
  // Create schedule with conflict matrix
  // For Wire: read SB write (read must happen before write)
  auto conflictMatrix = builder.getDictionaryAttr({
    builder.getNamedAttr("read,read", builder.getI32IntegerAttr(3)),    // CF
    builder.getNamedAttr("read,write", builder.getI32IntegerAttr(0)),   // SB
    builder.getNamedAttr("write,read", builder.getI32IntegerAttr(1)),   // SA
    builder.getNamedAttr("write,write", builder.getI32IntegerAttr(2))   // C
  });
  
  auto actions = builder.getArrayAttr({
    SymbolRefAttr::get(builder.getContext(), "read"),
    SymbolRefAttr::get(builder.getContext(), "write")
  });
  
  // Schedule must be the last operation in the block
  builder.create<::sharp::txn::ScheduleOp>(loc, actions, conflictMatrix);
  
  return primitive;
}

circt::firrtl::FModuleOp createWireFIRRTLModule(OpBuilder &builder, Location loc,
                                                StringRef name, Type dataType) {
  // Convert dataType to FIRRTL type (assuming i32 -> uint<32>)
  circt::firrtl::IntType firrtlType;
  if (auto intType = dyn_cast<IntegerType>(dataType)) {
    firrtlType = circt::firrtl::UIntType::get(builder.getContext(), intType.getWidth());
  } else {
    // Default to uint<32> if type conversion fails
    firrtlType = circt::firrtl::UIntType::get(builder.getContext(), 32);
  }
  
  // Create port info for the FIRRTL module
  SmallVector<circt::firrtl::PortInfo> ports;
  ports.push_back({builder.getStringAttr("clock"), 
                   circt::firrtl::ClockType::get(builder.getContext()),
                   circt::firrtl::Direction::In, {}, loc});
  ports.push_back({builder.getStringAttr("reset"), 
                   circt::firrtl::UIntType::get(builder.getContext(), 1),
                   circt::firrtl::Direction::In, {}, loc});
  ports.push_back({builder.getStringAttr("read_data"), 
                   firrtlType,
                   circt::firrtl::Direction::Out, {}, loc});
  ports.push_back({builder.getStringAttr("write_data"), 
                   firrtlType,
                   circt::firrtl::Direction::In, {}, loc});
  ports.push_back({builder.getStringAttr("write_enable"), 
                   circt::firrtl::UIntType::get(builder.getContext(), 1),
                   circt::firrtl::Direction::In, {}, loc});
  
  // Create the FIRRTL module
  auto firrtlModule = builder.create<circt::firrtl::FModuleOp>(
      loc, builder.getStringAttr(name.str() + "_impl"),
      circt::firrtl::ConventionAttr::get(builder.getContext(), 
                                          circt::firrtl::Convention::Internal),
      ports);
  
  // Create FIRRTL module body
  {
    OpBuilder::InsertionGuard firrtlGuard(builder);
    Block *firrtlBody = firrtlModule.getBodyBlock();
    
    builder.setInsertionPointToStart(firrtlBody);
    
    // Get port arguments
    [[maybe_unused]] auto clock = firrtlBody->getArgument(0);
    [[maybe_unused]] auto reset = firrtlBody->getArgument(1);
    auto readData = firrtlBody->getArgument(2);
    auto writeData = firrtlBody->getArgument(3);
    auto writeEnable = firrtlBody->getArgument(4);
    
    // Create wire
    auto wireOp = builder.create<circt::firrtl::WireOp>(
        loc, firrtlType, builder.getStringAttr("wire"));
    auto wire = wireOp.getResult();
    
    // Connect write logic - wire takes new value when write_enable is high
    builder.create<circt::firrtl::WhenOp>(
        loc, writeEnable, /*withElseRegion=*/false,
        [&]() {
          builder.create<circt::firrtl::ConnectOp>(loc, wire, writeData);
        });
    
    // Connect read output - always reads current wire value
    builder.create<circt::firrtl::ConnectOp>(loc, readData, wire);
  }
  
  return firrtlModule;
}

} // namespace txn
} // namespace sharp
} // namespace mlir