//===- CoreOps.td - Core dialect operations ---------------*- tablegen -*-===//
//
// Part of the Sharp Project
//
//===----------------------------------------------------------------------===//
//
// This file defines the Core dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_DIALECT_CORE_COREOPS_TD
#define SHARP_DIALECT_CORE_COREOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Core dialect definition
//===----------------------------------------------------------------------===//

def Core_Dialect : Dialect {
  let name = "sharp.core";
  let summary = "Sharp Core dialect";
  let description = [{
    The Sharp Core dialect provides fundamental operations and types for the
    Sharp infrastructure.
  }];
  
  let cppNamespace = "::sharp::core";
  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Core dialect types
//===----------------------------------------------------------------------===//

class Core_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Core_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Base operation definition
//===----------------------------------------------------------------------===//

class Core_Op<string mnemonic, list<Trait> traits = []> :
    Op<Core_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Core operations
//===----------------------------------------------------------------------===//

def Core_ConstantOp : Core_Op<"constant", [Pure]> {
  let summary = "constant operation";
  let description = [{
    The `sharp.core.constant` operation produces a constant value.
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $value `:` type($result)";
  let hasFolder = 1;
}

#endif // SHARP_DIALECT_CORE_COREOPS_TD