//===- Passes.h - Conversion Pass Entrypoints ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for all conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_CONVERSION_PASSES_H
#define SHARP_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
class Pass;
class RewritePatternSet;
namespace sharp {

//===----------------------------------------------------------------------===//
// TxnToFIRRTL
//===----------------------------------------------------------------------===//

// Pass creation functions are generated by tablegen

// Pass creation functions are generated by tablegen

/// Populate patterns for converting Txn dialect to FIRRTL dialect.
void populateTxnToFIRRTLConversionPatterns(RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// TxnToFunc
//===----------------------------------------------------------------------===//

// Forward declarations
class TypeConverter;

/// Populate patterns for converting Txn dialect to Func dialect.
void populateTxnToFuncConversionPatterns(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering conversion passes.
#define GEN_PASS_DECL
#include "sharp/Conversion/Passes.h.inc"

/// Register all conversion passes.
void registerConversionPasses();

#define GEN_PASS_REGISTRATION
#include "sharp/Conversion/Passes.h.inc"

} // namespace sharp
} // namespace mlir

#endif // SHARP_CONVERSION_PASSES_H