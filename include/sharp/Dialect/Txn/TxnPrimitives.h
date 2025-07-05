//===- TxnPrimitives.h - Sharp Txn primitive constructors ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares constructor functions for Sharp Txn primitives.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_DIALECT_TXN_TXNPRIMITIVES_H
#define SHARP_DIALECT_TXN_TXNPRIMITIVES_H

#include "mlir/IR/OpDefinition.h"
#include "sharp/Dialect/Txn/TxnOps.h"

namespace circt {
namespace firrtl {
class FModuleOp;
} // namespace firrtl
} // namespace circt

namespace mlir {
class OpBuilder;
class Location;
class Type;

namespace sharp {
namespace txn {

/// Create a Register primitive.
/// The register holds a value of the given type and supports read/write operations.
::sharp::txn::PrimitiveOp createRegisterPrimitive(OpBuilder &builder, Location loc,
                                                 StringRef name, Type dataType);

/// Create a Wire primitive.  
/// The wire provides a combinational connection with read/write operations.
/// Read must sequence before (SB) write.
::sharp::txn::PrimitiveOp createWirePrimitive(OpBuilder &builder, Location loc,
                                             StringRef name, Type dataType);

/// Create a FIRRTL module implementation for Register primitive.
/// This creates the actual hardware implementation with clock, reset, and data ports.
/// Note: This must be created within a firrtl.circuit context.
circt::firrtl::FModuleOp createRegisterFIRRTLModule(OpBuilder &builder, Location loc,
                                                    StringRef name, Type dataType);

/// Create a FIRRTL module implementation for Wire primitive.
/// This creates the actual hardware implementation with data ports.
/// Note: This must be created within a firrtl.circuit context.
circt::firrtl::FModuleOp createWireFIRRTLModule(OpBuilder &builder, Location loc,
                                                StringRef name, Type dataType);

/// Create a FIFO primitive.
/// The FIFO provides first-in-first-out queue functionality with enqueue/dequeue operations.
::sharp::txn::PrimitiveOp createFIFOPrimitive(OpBuilder &builder, Location loc,
                                              StringRef name, Type dataType,
                                              unsigned depth = 16);

/// Create a FIRRTL module implementation for FIFO primitive.
/// This creates the actual hardware implementation with handshaking signals.
/// Note: This must be created within a firrtl.circuit context.
circt::firrtl::FModuleOp createFIFOFIRRTLModule(OpBuilder &builder, Location loc,
                                                StringRef name, Type dataType,
                                                unsigned depth);

/// Create a Memory primitive.
/// The memory provides address-based storage with read/write/clear operations.
/// This is a spec primitive for verification.
::sharp::txn::PrimitiveOp createMemoryPrimitive(OpBuilder &builder, Location loc,
                                                StringRef name, Type dataType,
                                                unsigned addressWidth = 10);

/// Create a SpecFIFO primitive.
/// SpecFIFO is an unbounded FIFO for specification and verification.
/// It provides enqueue/dequeue/isEmpty/size/peek operations.
::sharp::txn::PrimitiveOp createSpecFIFOPrimitive(OpBuilder &builder, Location loc,
                                                  StringRef name, Type dataType);

/// Create a SpecMemory primitive.
/// SpecMemory is a memory with configurable read latency for verification.
/// It provides read/write/clear operations with latency configuration.
::sharp::txn::PrimitiveOp createSpecMemoryPrimitive(OpBuilder &builder, Location loc,
                                                    StringRef name, Type dataType,
                                                    unsigned addressWidth = 16,
                                                    unsigned defaultLatency = 1);

} // namespace txn
} // namespace sharp
} // namespace mlir

#endif // SHARP_DIALECT_TXN_TXNPRIMITIVES_H