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

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttribute(getValue());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  IntegerAttr valueAttr;

  if (parser.parseAttribute(valueAttr, getValueAttrName(result.name),
                            result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  result.addTypes(valueAttr.getType());
  return success();
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}