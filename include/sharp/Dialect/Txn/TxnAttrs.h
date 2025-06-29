//===- TxnAttrs.h - Txn dialect attributes --------------------------------===//
//
// Part of the Sharp Project.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_DIALECT_TXN_TXNATTRS_H
#define SHARP_DIALECT_TXN_TXNATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "sharp/Dialect/Txn/TxnEnums.h.inc"

// Timing attribute will be implemented as a simple string attribute for now
// Format: "combinational" | "static(<n>)" | "dynamic"
namespace sharp {
namespace txn {

inline ::mlir::StringAttr getTimingAttr(::mlir::MLIRContext *context, TimingKind kind, 
                                       std::optional<int32_t> cycles = std::nullopt) {
  std::string timingStr;
  switch (kind) {
    case TimingKind::Combinational:
      timingStr = "combinational";
      break;
    case TimingKind::Static:
      if (cycles)
        timingStr = "static(" + std::to_string(*cycles) + ")";
      else
        timingStr = "static";
      break;
    case TimingKind::Dynamic:
      timingStr = "dynamic";
      break;
  }
  return ::mlir::StringAttr::get(context, timingStr);
}

} // namespace txn
} // namespace sharp

#endif // SHARP_DIALECT_TXN_TXNATTRS_H