# MLIR/CIRCT Python Binding Infrastructure

## Overview

MLIR and CIRCT Python bindings use nanobind (MLIR 17+) or pybind11 to expose C++ functionality to Python through a stable C API.

## Core Architecture

### Technology Stack
1. **nanobind/pybind11**: C++ to Python binding
2. **MLIR C API**: Stable interface avoiding ABI issues
3. **CMake Infrastructure**: Automated binding generation
4. **TableGen**: Generates Python classes from dialect definitions

### Package Structure
```
package/
├── __init__.py          # Public API
├── ir.py                # IR manipulation
├── dialects/            # Dialect bindings
│   └── my_dialect.py    # Generated from TableGen
└── _mlir_libs/          # Native extensions
    └── _mlir.so         # Compiled C++ module
```

## How Bindings Work

### 1. TableGen Generation
```cmake
mlir_python_dialect_op_binding(MyDialectBindings
  DIALECT_NAME my_dialect
  TD_FILE MyDialect.td
)
```
Generates Python operation classes from TableGen definitions.

### 2. C++ Extension Modules
```cpp
PYBIND11_MODULE(_myDialect, m) {
  m.doc() = "My dialect bindings";
  populateMyDialectBindings(m);
}
```
Compiled to shared libraries (.so/.pyd).

### 3. Package Prefix Mechanism
- MLIR: No prefix → `import mlir`
- CIRCT: `circt.` prefix → `from circt import ir`
- Sharp: `sharp.` prefix → `from sharp import ir`

## Sharp/PySharp Architecture

### Recommended: Sibling Packages
```
python_packages/
├── sharp/               # MLIR bindings
│   ├── ir.py
│   └── _mlir_libs/
└── pysharp/            # Frontend
    └── __init__.py     # Imports from sharp
```

### PySharp Usage
```python
# pysharp/__init__.py
import sharp
from sharp import ir

# Use Sharp's MLIR bindings
with ir.Context() as ctx:
    sharp.register_dialects(ctx)
```

## CMake Integration

### Key Macros
```cmake
# Declare Python sources
declare_mlir_python_sources(SharpPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python"
  SOURCES sharp/__init__.py
)

# Generate dialect bindings
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SharpPythonSources.Dialects
  TD_FILE TxnOps.td
  DIALECT_NAME txn
)

# Build C++ extension
add_mlir_python_extension(SharpExtension
  MODULE_NAME _sharp
  SOURCES SharpModule.cpp
  LINK_LIBS MLIRC
)

# Package everything
add_mlir_python_modules(Sharp
  ROOT_PREFIX "${PYTHON_PACKAGES_DIR}/"
  DECLARED_SOURCES SharpPythonSources
)
```

## Common Issues

### ImportError: cannot import name '_mlir'
- **Cause**: Package structure mismatch
- **Fix**: Ensure prefix matches CMake settings

### Dialect not registered
- **Cause**: Missing registration call
- **Fix**: Call `register_dialects()` on context

### Nested package problems
- **Cause**: Incorrect nesting of MLIR bindings
- **Fix**: Use sibling packages or proper prefix

## Key Principles

1. **Never import `_mlir` directly** - use public API
2. **Match package structure to CMake prefix**
3. **Register dialects before use**
4. **Follow established patterns** (CIRCT/PyCDE)
5. **Keep bindings minimal** - just expose dialects