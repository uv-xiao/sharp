//===- Txn.td - Definition of Txn dialect ---------------------------------===//
//
// Part of the Sharp Project.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_DIALECT_TXN_TD
#define SHARP_DIALECT_TXN_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "sharp/Dialect/Txn/TxnBase.td"

include "sharp/Dialect/Txn/TxnTypes.td"
include "sharp/Dialect/Txn/TxnAttrs.td"
include "sharp/Dialect/Txn/TxnOps.td"

#endif // SHARP_DIALECT_TXN_TD