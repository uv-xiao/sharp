//===- CoreModule.cpp - Sharp Core Python Bindings -----------------------===//
//
// Python bindings for the Sharp Core dialect.
//
//===----------------------------------------------------------------------===//

#include "SharpModules.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace sharp {
namespace python {

void populateCoreModule(nb::module_ &m) {
  m.doc() = "Sharp Core dialect Python bindings";
  
  // Add any specific Core dialect bindings here
  // For now, the dialect registration is handled in the main module
}

} // namespace python
} // namespace sharp