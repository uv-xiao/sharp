//===- TxnTypes.h - Txn dialect types -------------------------------------===//
//
// Part of the Sharp Project.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_DIALECT_TXN_TXNTYPES_H
#define SHARP_DIALECT_TXN_TXNTYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "sharp/Dialect/Txn/TxnTypes.h.inc"

#endif // SHARP_DIALECT_TXN_TXNTYPES_H