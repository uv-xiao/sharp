//===- SharpModule.cpp - Sharp Python Bindings ---------------------------===//
//
// Main entry point for Sharp Python bindings.
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "sharp-c/Dialects.h"

// MLIR dialect headers
#include "mlir-c/Dialect/SCF.h"
#include "mlir-c/Dialect/SMT.h"
#include "mlir-c/Dialect/Index.h"
#include "mlir-c/Dialect/Arith.h"

// CIRCT dialect headers
#include "circt-c/Dialect/FIRRTL.h"
#include "circt-c/Dialect/Comb.h"
#include "circt-c/Dialect/HWArith.h"
#include "circt-c/Dialect/Seq.h"
#include "circt-c/Dialect/SV.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(_sharp, m) {
  m.doc() = "Sharp Python Native Extension";
  
  m.def(
      "register_dialects",
      [](MlirContext context) {
        // Register and load Sharp Txn dialect
        MlirDialectHandle sharpTxn = mlirGetDialectHandle__txn__();
        mlirDialectHandleRegisterDialect(sharpTxn, context);
        mlirDialectHandleLoadDialect(sharpTxn, context);
        
        // Register and load MLIR dialects
        MlirDialectHandle scf = mlirGetDialectHandle__scf__();
        mlirDialectHandleRegisterDialect(scf, context);
        mlirDialectHandleLoadDialect(scf, context);
        
        MlirDialectHandle smt = mlirGetDialectHandle__smt__();
        mlirDialectHandleRegisterDialect(smt, context);
        mlirDialectHandleLoadDialect(smt, context);
        
        MlirDialectHandle index = mlirGetDialectHandle__index__();
        mlirDialectHandleRegisterDialect(index, context);
        mlirDialectHandleLoadDialect(index, context);
        
        MlirDialectHandle arith = mlirGetDialectHandle__arith__();
        mlirDialectHandleRegisterDialect(arith, context);
        mlirDialectHandleLoadDialect(arith, context);
        
        // Register and load CIRCT dialects
        MlirDialectHandle firrtl = mlirGetDialectHandle__firrtl__();
        mlirDialectHandleRegisterDialect(firrtl, context);
        mlirDialectHandleLoadDialect(firrtl, context);
        
        MlirDialectHandle comb = mlirGetDialectHandle__comb__();
        mlirDialectHandleRegisterDialect(comb, context);
        mlirDialectHandleLoadDialect(comb, context);
        
        
        MlirDialectHandle hwarith = mlirGetDialectHandle__hwarith__();
        mlirDialectHandleRegisterDialect(hwarith, context);
        mlirDialectHandleLoadDialect(hwarith, context);
        
        MlirDialectHandle seq = mlirGetDialectHandle__seq__();
        mlirDialectHandleRegisterDialect(seq, context);
        mlirDialectHandleLoadDialect(seq, context);
        
        MlirDialectHandle sv = mlirGetDialectHandle__sv__();
        mlirDialectHandleRegisterDialect(sv, context);
        mlirDialectHandleLoadDialect(sv, context);
      },
      nb::arg("context"),
      "Register Sharp, MLIR and CIRCT dialects on a Context.");
}