//===- TxnTypes.td - Txn dialect type definitions -------------------------===//
//
// Part of the Sharp Project.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_DIALECT_TXN_TYPES_TD
#define SHARP_DIALECT_TXN_TYPES_TD

include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

def ActionType : TxnType<"Action"> {
  let mnemonic = "action";
  let summary = "Action type that may produce a value or abort";
  let description = [{
    Represents an action that either produces a value of the wrapped type
    or aborts the transaction. This is the fundamental type for stateful
    operations in the TXN dialect.
  }];

  let parameters = (ins "::mlir::Type":$resultType);

  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$resultType), [{
      return $_get(resultType.getContext(), resultType);
    }]>
  ];

  let assemblyFormat = "`<` $resultType `>`";
}

def ValueType : TxnType<"Value"> {
  let mnemonic = "value";
  let summary = "Pure computation type";
  let description = [{
    Represents a pure computation that produces a value without side effects.
    This type is used for value methods and pure computations.
  }];

  let parameters = (ins "::mlir::Type":$resultType);

  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$resultType), [{
      return $_get(resultType.getContext(), resultType);
    }]>
  ];

  let assemblyFormat = "`<` $resultType `>`";
}

def ModuleType : TxnType<"Module"> {
  let mnemonic = "module";
  let summary = "Module interface type";
  let description = [{
    References a module's interface, allowing for typed module instances
    and method calls.
  }];

  let parameters = (ins "::mlir::StringAttr":$moduleName);

  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::StringAttr":$moduleName), [{
      return $_get(moduleName.getContext(), moduleName);
    }]>
  ];

  let assemblyFormat = "`<` $moduleName `>`";
}

// Type constraints for operations
def Txn_ActionType : Type<CPred<"llvm::isa<::sharp::txn::ActionType>($_self)">,
                              "txn.action type">;

def Txn_ValueType : Type<CPred<"llvm::isa<::sharp::txn::ValueType>($_self)">,
                             "txn.value type">;

def Txn_ModuleType : Type<CPred<"llvm::isa<::sharp::txn::ModuleType>($_self)">,
                              "txn.module type">;

#endif // SHARP_DIALECT_TXN_TYPES_TD