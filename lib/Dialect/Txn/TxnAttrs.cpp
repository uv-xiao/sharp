//===- TxnAttrs.cpp - Txn dialect attributes ------------------------------===//
//
// Part of the Sharp Project.
//
//===----------------------------------------------------------------------===//

#include "sharp/Dialect/Txn/TxnAttrs.h"
#include "sharp/Dialect/Txn/TxnDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace sharp::txn;

#include "sharp/Dialect/Txn/TxnEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// Conflict Matrix Helpers
//===----------------------------------------------------------------------===//

namespace sharp {
namespace txn {

static std::string makeActionPairKey(StringRef action1, StringRef action2) {
  // Normalize the order to ensure consistent lookup
  if (action1 < action2)
    return (action1 + "," + action2).str();
  return (action2 + "," + action1).str();
}

ConflictRelation getConflictRelation(DictionaryAttr matrix,
                                     StringRef action1,
                                     StringRef action2) {
  auto key = makeActionPairKey(action1, action2);
  
  if (auto attr = matrix.get(key)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      return static_cast<ConflictRelation>(intAttr.getInt());
    }
  }
  
  // Default to conflict-free if not specified
  return ConflictRelation::ConflictFree;
}

DictionaryAttr setConflictRelation(MLIRContext *ctx,
                                   DictionaryAttr matrix,
                                   StringRef action1,
                                   StringRef action2,
                                   ConflictRelation relation) {
  auto key = makeActionPairKey(action1, action2);
  
  SmallVector<NamedAttribute> attrs;
  for (auto attr : matrix) {
    if (attr.getName() != key)
      attrs.push_back(attr);
  }
  
  attrs.push_back(NamedAttribute(
      StringAttr::get(ctx, key),
      IntegerAttr::get(IntegerType::get(ctx, 32), static_cast<int32_t>(relation))));
  
  return DictionaryAttr::get(ctx, attrs);
}

} // namespace txn
} // namespace sharp