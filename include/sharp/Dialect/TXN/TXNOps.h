//===- TXNOps.h - TXN dialect operations ----------------------------------===//
//
// Part of the Sharp Project.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_DIALECT_TXN_TXNOPS_H
#define SHARP_DIALECT_TXN_TXNOPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "sharp/Dialect/TXN/TXNDialect.h"
#include "sharp/Dialect/TXN/TXNTypes.h"

namespace sharp {
namespace txn {
using namespace mlir;
} // namespace txn
} // namespace sharp

#define GET_OP_CLASSES
#include "sharp/Dialect/TXN/TXN.h.inc"

#endif // SHARP_DIALECT_TXN_TXNOPS_H