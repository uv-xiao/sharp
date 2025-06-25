//===- CoreOps.h - Sharp Core Operations ------------------------*- C++ -*-===//
//
// Part of the Sharp Project
//
//===----------------------------------------------------------------------===//
//
// This file declares the Sharp Core dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_DIALECT_CORE_COREOPS_H
#define SHARP_DIALECT_CORE_COREOPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "sharp/Dialect/Core/CoreDialect.h"

#define GET_OP_CLASSES
#include "sharp/Dialect/Core/CoreOps.h.inc"

#endif // SHARP_DIALECT_CORE_COREOPS_H