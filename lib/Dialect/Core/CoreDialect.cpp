//===- CoreDialect.cpp - Sharp Core Dialect Implementation -----*- C++ -*-===//
//
// Part of the Sharp Project
//
//===----------------------------------------------------------------------===//
//
// This file implements the Sharp Core dialect.
//
//===----------------------------------------------------------------------===//

#include "sharp/Dialect/Core/CoreDialect.h"
#include "sharp/Dialect/Core/CoreOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace sharp::core;

#include "sharp/Dialect/Core/CoreDialect.cpp.inc"

void CoreDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "sharp/Dialect/Core/CoreOps.cpp.inc"
      >();
}

Operation *CoreDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  return builder.create<ConstantOp>(loc, type, value);
}