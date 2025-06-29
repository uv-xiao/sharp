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

} // namespace txn
} // namespace sharp
} // namespace mlir

#endif // SHARP_DIALECT_TXN_TXNPRIMITIVES_H