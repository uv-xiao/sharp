//===- SharpModule.cpp - Sharp Python Bindings ---------------------------===//
//
// Main entry point for Sharp Python bindings.
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Bindings/Python/Interop.h"
#include "sharp-c/Dialects.h"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(_sharp, m) {
  m.doc() = "Sharp Python Native Extension";
  
  m.def(
      "register_dialects",
      [](nb::object capsule) {
        // Get the MlirContext capsule from PyMlirContext capsule.
        auto wrappedCapsule = capsule.attr(MLIR_PYTHON_CAPI_PTR_ATTR);
        MlirContext context = mlirPythonCapsuleToContext(wrappedCapsule.ptr());

        // Register and load Sharp dialect
        MlirDialectHandle sharp = mlirGetDialectHandle__sharp__();
        mlirDialectHandleRegisterDialect(sharp, context);
        mlirDialectHandleLoadDialect(sharp, context);
      },
      "Register Sharp dialects on a PyMlirContext.");
}