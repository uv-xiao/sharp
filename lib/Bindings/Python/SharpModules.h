//===- SharpModules.h - Sharp Python Bindings ----------------------------===//
//
// Header for Sharp Python bindings.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_BINDINGS_PYTHON_SHARPMODULES_H
#define SHARP_BINDINGS_PYTHON_SHARPMODULES_H

#include <nanobind/nanobind.h>

namespace sharp {
namespace python {

void populateCoreModule(nanobind::module_ &m);

} // namespace python
} // namespace sharp

#endif // SHARP_BINDINGS_PYTHON_SHARPMODULES_H