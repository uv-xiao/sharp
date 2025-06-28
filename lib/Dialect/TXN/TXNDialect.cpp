//===- TXNDialect.cpp - TXN dialect implementation ------------------------===//
//
// Part of the Sharp Project.
//
//===----------------------------------------------------------------------===//

#include "sharp/Dialect/TXN/TXNDialect.h"
#include "sharp/Dialect/TXN/TXNOps.h"
#include "sharp/Dialect/TXN/TXNTypes.h"

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

void TXNDialect::initialize() {
  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "sharp/Dialect/TXN/TXNTypes.cpp.inc"
      >();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "sharp/Dialect/TXN/TXN.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "sharp/Dialect/TXN/TXNTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect definitions
//===----------------------------------------------------------------------===//

#include "sharp/Dialect/TXN/TXNDialect.cpp.inc"