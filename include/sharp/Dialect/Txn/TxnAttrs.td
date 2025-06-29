//===- TxnAttrs.td - Txn dialect attribute definitions --------------------===//
//
// Part of the Sharp Project.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_DIALECT_TXN_ATTRS_TD
#define SHARP_DIALECT_TXN_ATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "sharp/Dialect/Txn/TxnBase.td"

//===----------------------------------------------------------------------===//
// Conflict Matrix Enums
//===----------------------------------------------------------------------===//

def TxnConflictRelation : I32EnumAttr<"ConflictRelation",
    "Conflict relationship between actions",
    [
      I32EnumAttrCase<"SequenceBefore", 0, "SB">,
      I32EnumAttrCase<"SequenceAfter", 1, "SA">,
      I32EnumAttrCase<"Conflict", 2, "C">,
      I32EnumAttrCase<"ConflictFree", 3, "CF">
    ]> {
  let cppNamespace = "::sharp::txn";
}

//===----------------------------------------------------------------------===//
// Timing Attributes
//===----------------------------------------------------------------------===//

def TxnTimingKind : I32EnumAttr<"TimingKind",
    "Timing specification for rules and methods",
    [
      I32EnumAttrCase<"Combinational", 0, "combinational">,
      I32EnumAttrCase<"Static", 1, "static">,
      I32EnumAttrCase<"Dynamic", 2, "dynamic">
    ]> {
  let cppNamespace = "::sharp::txn";
}

// Timing attribute that includes the kind and optional cycle count
def TxnTimingAttr : AttrDef<TxnDialect, "Timing"> {
  let mnemonic = "timing";
  let summary = "Timing specification for a rule or method";
  let description = [{
    Specifies the timing behavior of a rule or method. Can be:
    - Combinational: Completes within the same cycle
    - Static(n): Takes exactly n cycles
    - Dynamic: Takes a variable number of cycles
  }];

  let parameters = (ins
    "TimingKind":$kind,
    OptionalParameter<"int32_t">:$cycles
  );

  let assemblyFormat = [{
    `<` $kind (`(` $cycles^ `)`)? `>`
  }];

  let builders = [
    AttrBuilder<(ins "TimingKind":$kind), [{
      return $_get($_ctxt, kind, std::nullopt);
    }]>,
    AttrBuilder<(ins "TimingKind":$kind, "int32_t":$cycles), [{
      return $_get($_ctxt, kind, cycles);
    }]>
  ];

  let extraClassDeclaration = [{
    bool isCombinational() const { return getKind() == TimingKind::Combinational; }
    bool isStatic() const { return getKind() == TimingKind::Static; }
    bool isDynamic() const { return getKind() == TimingKind::Dynamic; }
    bool isMultiCycle() const { return !isCombinational(); }
  }];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Conflict Matrix Attribute
//===----------------------------------------------------------------------===//

// Simplified conflict matrix attribute
def TxnConflictMatrixAttr : Attr<CPred<"isa<::mlir::DictionaryAttr>($_self)">,
                                  "conflict matrix attribute"> {
  let storageType = "::mlir::DictionaryAttr";
  let returnType = "::mlir::DictionaryAttr";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

#endif // SHARP_DIALECT_TXN_ATTRS_TD