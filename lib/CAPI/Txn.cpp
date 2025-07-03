//===- Txn.cpp - C API for Sharp Txn Dialect -----------------------------===//
//
// This file implements the C-API for the Sharp Txn dialect.
//
//===----------------------------------------------------------------------===//

#include "sharp-c/Dialects.h"
#include "sharp/Dialect/Txn/TxnDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SharpTxn, txn, 
                                      sharp::txn::TxnDialect)