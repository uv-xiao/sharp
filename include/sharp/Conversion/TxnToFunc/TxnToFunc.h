//===- TxnToFunc.h - Txn to Func conversion --------------------*- C++ -*-===//
//
// This file declares the conversion from Txn dialect to Func dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_CONVERSION_TXNTOFUNC_TXNTOFUNC_H
#define SHARP_CONVERSION_TXNTOFUNC_TXNTOFUNC_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ConversionTarget;
class MLIRContext;
class RewritePatternSet;
class TypeConverter;
} // namespace mlir

namespace mlir {
namespace sharp {

/// Populate the given list with patterns that convert from Txn to Func.
void populateTxnToFuncConversionPatterns(mlir::TypeConverter &typeConverter,
                                        mlir::RewritePatternSet &patterns);

/// Create a pass to convert Txn dialect to Func dialect.
std::unique_ptr<mlir::Pass> createConvertTxnToFuncPass();

} // namespace sharp
} // namespace mlir

#endif // SHARP_CONVERSION_TXNTOFUNC_TXNTOFUNC_H