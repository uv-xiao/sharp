//===- TxnBase.td - Base definitions for Txn dialect ----------------------===//
//
// Part of the Sharp Project.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_DIALECT_TXN_BASE_TD
#define SHARP_DIALECT_TXN_BASE_TD

include "mlir/IR/OpBase.td"

def TxnDialect : Dialect {
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


  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::comb::CombDialect",
    "circt::seq::SeqDialect",
    "mlir::arith::ArithDialect"
  ];
}

// Base class for the types in this dialect.
class TxnType<string name> : TypeDef<TxnDialect, name> {}

// Base class for the operations in this dialect.
class TxnOp<string mnemonic, list<Trait> traits = []> :
    Op<TxnDialect, mnemonic, traits>;

#endif // SHARP_DIALECT_TXN_BASE_TD