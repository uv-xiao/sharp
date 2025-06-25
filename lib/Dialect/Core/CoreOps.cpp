//===- CoreOps.cpp - Sharp Core Operations Implementation -------*- C++ -*-===//
//
// Part of the Sharp Project
//
//===----------------------------------------------------------------------===//
//
// This file implements the Sharp Core dialect operations.
//
//===----------------------------------------------------------------------===//

#include "sharp/Dialect/Core/CoreOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace sharp::core;

#define GET_OP_CLASSES
#include "sharp/Dialect/Core/CoreOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return getValue();
}