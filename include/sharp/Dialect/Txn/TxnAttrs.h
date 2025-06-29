//===- TxnAttrs.h - Txn dialect attributes --------------------------------===//
//
// Part of the Sharp Project.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_DIALECT_TXN_TXNATTRS_H
#define SHARP_DIALECT_TXN_TXNATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

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

// Helper functions for conflict matrix manipulation
ConflictRelation getConflictRelation(::mlir::DictionaryAttr matrix,
                                     ::mlir::StringRef action1,
                                     ::mlir::StringRef action2);

::mlir::DictionaryAttr setConflictRelation(::mlir::MLIRContext *ctx,
                                          ::mlir::DictionaryAttr matrix,
                                          ::mlir::StringRef action1,
                                          ::mlir::StringRef action2,
                                          ConflictRelation relation);

// Create a conflict matrix attribute with proper formatting
inline ::mlir::DictionaryAttr createConflictMatrixAttr(
    ::mlir::MLIRContext *ctx,
    std::initializer_list<std::tuple<std::string, std::string, ConflictRelation>> relations) {
  ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
  
  for (const auto& [action1, action2, relation] : relations) {
    std::string key = action1 < action2 ? 
        action1 + "," + action2 : action2 + "," + action1;
    attrs.push_back(::mlir::NamedAttribute(
        ::mlir::StringAttr::get(ctx, key),
        ::mlir::IntegerAttr::get(::mlir::IntegerType::get(ctx, 32), 
                                 static_cast<int32_t>(relation))));
  }
  
  return ::mlir::DictionaryAttr::get(ctx, attrs);
}

} // namespace txn
} // namespace sharp

#endif // SHARP_DIALECT_TXN_TXNATTRS_H