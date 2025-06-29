//===- TxnDialect.cpp - Txn dialect implementation ------------------------===//
//
// Part of the Sharp Project.
//
//===----------------------------------------------------------------------===//

#include "sharp/Dialect/Txn/TxnDialect.h"
#include "sharp/Dialect/Txn/TxnOps.h"
#include "sharp/Dialect/Txn/TxnTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

// Include CIRCT dialects we depend on
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace sharp;
using namespace sharp::txn;

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

void TxnDialect::initialize() {
  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "sharp/Dialect/Txn/TxnTypes.cpp.inc"
      >();

  // Register attributes.
  // TODO: Register custom attributes when we have proper AttrDef support

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "sharp/Dialect/Txn/Txn.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "sharp/Dialect/Txn/TxnTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect definitions
//===----------------------------------------------------------------------===//

#include "sharp/Dialect/Txn/TxnDialect.cpp.inc"