//===- FIFO.cpp - Sharp Txn FIFO primitive implementation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the FIFO primitive for the Sharp Txn dialect.
// FIFOs provide first-in-first-out queue functionality with configurable depth.
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

::sharp::txn::PrimitiveOp createFIFOPrimitive(OpBuilder &builder, Location loc,
                                              StringRef name, Type dataType, 
                                              unsigned depth) {
  // Create interface type for the primitive
  auto moduleType = ::sharp::txn::ModuleType::get(builder.getContext(), 
                                                  StringAttr::get(builder.getContext(), name));
  
  // Create the primitive operation
  auto primitive = builder.create<::sharp::txn::PrimitiveOp>(loc, 
                                                            StringAttr::get(builder.getContext(), name),
                                                            builder.getStringAttr("hw"), 
                                                            TypeAttr::get(moduleType),
                                                            /*type_parameters=*/ArrayAttr());
  
  // Store depth as an attribute
  primitive->setAttr("depth", builder.getI32IntegerAttr(depth));
  
  // Create a new builder for the primitive body
  OpBuilder::InsertionGuard guard(builder);
  Block *body = &primitive.getBody().emplaceBlock();
  builder.setInsertionPointToStart(body);
  
  // Define the interface methods
  auto boolType = builder.getI1Type();
  auto enqType = builder.getFunctionType({dataType}, {});
  auto deqType = builder.getFunctionType({}, {dataType});
  auto statusType = builder.getFunctionType({}, {boolType});
  
  // Create enqueue method (action method)
  auto enqMethod = builder.create<::sharp::txn::FirActionMethodOp>(
      loc, builder.getStringAttr("enqueue"), TypeAttr::get(enqType),
      /*ready=*/StringAttr(), /*enable=*/StringAttr(), /*result=*/StringAttr(),
      /*prefix=*/StringAttr(), /*always_ready=*/UnitAttr(), /*always_enable=*/UnitAttr());
  enqMethod->setAttr("firrtl.data_port", builder.getStringAttr("enq_data"));
  enqMethod->setAttr("firrtl.enable_port", builder.getStringAttr("enq_valid"));
  enqMethod->setAttr("firrtl.ready_port", builder.getStringAttr("enq_ready"));
  
  // Create dequeue method (action method)
  auto deqMethod = builder.create<::sharp::txn::FirActionMethodOp>(
      loc, builder.getStringAttr("dequeue"), TypeAttr::get(deqType),
      /*ready=*/StringAttr(), /*enable=*/StringAttr(), /*result=*/StringAttr(),
      /*prefix=*/StringAttr(), /*always_ready=*/UnitAttr(), /*always_enable=*/UnitAttr());
  deqMethod->setAttr("firrtl.data_port", builder.getStringAttr("deq_data"));
  deqMethod->setAttr("firrtl.enable_port", builder.getStringAttr("deq_ready"));
  deqMethod->setAttr("firrtl.ready_port", builder.getStringAttr("deq_valid"));
  
  // Create isEmpty method (value method)
  auto isEmptyMethod = builder.create<::sharp::txn::FirValueMethodOp>(
      loc, builder.getStringAttr("isEmpty"), TypeAttr::get(statusType),
      /*result=*/StringAttr(), /*prefix=*/StringAttr());
  isEmptyMethod->setAttr("firrtl.port", builder.getStringAttr("empty"));
  
  // Create isFull method (value method)
  auto isFullMethod = builder.create<::sharp::txn::FirValueMethodOp>(
      loc, builder.getStringAttr("isFull"), TypeAttr::get(statusType),
      /*result=*/StringAttr(), /*prefix=*/StringAttr());
  isFullMethod->setAttr("firrtl.port", builder.getStringAttr("full"));
  
  // Add default clock and reset
  builder.create<::sharp::txn::ClockByOp>(loc, SymbolRefAttr::get(builder.getContext(), "clk"));
  builder.create<::sharp::txn::ResetByOp>(loc, SymbolRefAttr::get(builder.getContext(), "rst"));
  
  // Add reference to the FIRRTL module implementation
  primitive->setAttr("firrtl.impl", builder.getStringAttr(name.str() + "_impl"));
  
  // Create schedule with conflict matrix
  // For FIFO: enqueue conflicts with isFull, dequeue conflicts with isEmpty
  auto conflictMatrix = builder.getDictionaryAttr({
    // Status methods don't conflict with each other
    builder.getNamedAttr("isEmpty,isEmpty", builder.getI32IntegerAttr(3)),   // CF
    builder.getNamedAttr("isEmpty,isFull", builder.getI32IntegerAttr(3)),    // CF
    builder.getNamedAttr("isFull,isEmpty", builder.getI32IntegerAttr(3)),    // CF
    builder.getNamedAttr("isFull,isFull", builder.getI32IntegerAttr(3)),     // CF
    
    // Enqueue conflicts with isFull (must check before enqueue)
    builder.getNamedAttr("isFull,enqueue", builder.getI32IntegerAttr(0)),    // SB
    builder.getNamedAttr("enqueue,isFull", builder.getI32IntegerAttr(1)),    // SA
    
    // Dequeue conflicts with isEmpty (must check before dequeue)
    builder.getNamedAttr("isEmpty,dequeue", builder.getI32IntegerAttr(0)),   // SB
    builder.getNamedAttr("dequeue,isEmpty", builder.getI32IntegerAttr(1)),   // SA
    
    // Enqueue and dequeue conflict with each other
    builder.getNamedAttr("enqueue,dequeue", builder.getI32IntegerAttr(2)),   // C
    builder.getNamedAttr("dequeue,enqueue", builder.getI32IntegerAttr(2)),   // C
    
    // Other combinations are conflict-free
    builder.getNamedAttr("isEmpty,enqueue", builder.getI32IntegerAttr(3)),   // CF
    builder.getNamedAttr("enqueue,isEmpty", builder.getI32IntegerAttr(3)),   // CF
    builder.getNamedAttr("isFull,dequeue", builder.getI32IntegerAttr(3)),    // CF
    builder.getNamedAttr("dequeue,isFull", builder.getI32IntegerAttr(3))     // CF
  });
  
  auto actions = builder.getArrayAttr({
    SymbolRefAttr::get(builder.getContext(), "enqueue"),
    SymbolRefAttr::get(builder.getContext(), "dequeue"),
    SymbolRefAttr::get(builder.getContext(), "isEmpty"),
    SymbolRefAttr::get(builder.getContext(), "isFull")
  });
  
  // Schedule must be the last operation in the block
  builder.create<::sharp::txn::ScheduleOp>(loc, actions, conflictMatrix);
  
  return primitive;
}

// For now, we'll leave the FIRRTL implementation as a TODO
// In a real implementation, this would create a FIRRTL queue module
circt::firrtl::FModuleOp createFIFOFIRRTLModule(OpBuilder &builder, Location loc,
                                                StringRef name, Type dataType,
                                                unsigned depth) {
  // TODO: Implement FIRRTL FIFO module generation
  // This would create a module with:
  // - Internal circular buffer storage
  // - Read/write pointers
  // - Full/empty logic
  // - Proper handshaking signals
  
  return nullptr;
}

} // namespace txn
} // namespace sharp
} // namespace mlir