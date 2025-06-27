//===- Core.cpp - C API for Sharp Core Dialect ---------------------------===//
//
// This file implements the C-API for the Sharp Core dialect.
//
//===----------------------------------------------------------------------===//

#include "sharp-c/Dialects.h"
#include "sharp/Dialect/Core/CoreDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SharpCore, sharp, 
                                      sharp::core::CoreDialect)