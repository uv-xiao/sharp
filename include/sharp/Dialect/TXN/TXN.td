//===- TXN.td - Definition of TXN dialect ---------------------------------===//
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

def TXNDialect : Dialect {
  let name = "txn";
  let cppNamespace = "::sharp::txn";

  let summary = "Transaction dialect for Sharp";
  let description = [{
    The Transaction (txn) dialect provides a transactional hardware description
    layer inspired by Fjfj. It enables sequential reasoning about concurrent
    hardware while maintaining the ability to synthesize efficient concurrent
    implementations.

    Key features:
    - Primitive-based design with synthesizable and non-synthesizable variants
    - Transaction semantics with abort propagation
    - Sequential scheduling within transactions
    - Clear separation between value (pure) and action (stateful) methods
  }];

  let useDefaultTypePrinterParser = 1;

  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::comb::CombDialect",
    "circt::seq::SeqDialect",
    "mlir::arith::ArithDialect"
  ];
}

// Base class for the types in this dialect.
class TXNType<string name> : TypeDef<TXNDialect, name> {}

// Base class for the operations in this dialect.
class TXNOp<string mnemonic, list<Trait> traits = []> :
    Op<TXNDialect, mnemonic, traits>;

include "sharp/Dialect/TXN/TXNTypes.td"
include "sharp/Dialect/TXN/TXNOps.td"

#endif // SHARP_DIALECT_TXN_TD