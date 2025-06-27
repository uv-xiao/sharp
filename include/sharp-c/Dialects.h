//===- Dialects.h - C API for Sharp Dialects -----------------------------===//
//
// This file declares the C-API for Sharp dialects.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_C_DIALECTS_H
#define SHARP_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(SharpCore, sharp);

#ifdef __cplusplus
}
#endif

#endif // SHARP_C_DIALECTS_H