//===- TxnAttrs.cpp - Txn dialect attributes ------------------------------===//
//
// Part of the Sharp Project.
//
//===----------------------------------------------------------------------===//

#include "sharp/Dialect/Txn/TxnAttrs.h"
#include "sharp/Dialect/Txn/TxnDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace sharp::txn;

#include "sharp/Dialect/Txn/TxnEnums.cpp.inc"

// Attribute implementations will be added here when needed