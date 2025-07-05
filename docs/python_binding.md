# MLIR/CIRCT Python Binding Infrastructure

This document provides a comprehensive technical overview of how MLIR and CIRCT Python bindings work, including the underlying principles, libraries used, build system integration, and what gets generated. It also explains how to properly integrate these bindings into frontend projects like PySharp.

## Core Principles and Technologies

### Binding Technology Stack

1. **nanobind** (MLIR 17+) / **pybind11** (older versions)
   - C++17 header-only library for Python bindings
   - Chosen for performance and modern C++ support
   - Provides automatic type conversions and exception handling
   - Supports numpy arrays and buffer protocol

2. **MLIR C API (MLIR-C)**
   - Stable C interface to MLIR functionality
   - Located in `mlir/include/mlir-c/`
   - Provides opaque handles to MLIR objects
   - Enables language bindings without C++ ABI issues

3. **CMake Infrastructure**
   - `AddMLIRPython.cmake` provides binding generation macros
   - `MLIRPythonExtensions.cmake` handles extension module building
   - Automatic dependency tracking and module registration

### How Bindings Are Generated

1. **TableGen Generation** (for dialects):
   ```cmake
   mlir_python_dialect_op_binding(MyDialectBindings
     DIALECT_NAME my_dialect
     TD_FILE MyDialect.td
   )
   ```
   - Generates Python wrapper classes from TableGen definitions
   - Creates operation constructors and attribute accessors
   - Produces `_MyDialect_ops_gen.py` files

2. **C++ Extension Modules**:
   ```cpp
   PYBIND11_MODULE(_myDialect, m) {
     m.doc() = "My dialect bindings";
     populateMyDialectBindings(m);
   }
   ```
   - Compiled to `.so`/`.pyd` files
   - Linked against MLIR C API libraries
   - Registered with Python's import system

3. **Python Wrapper Layers**:
   - Pure Python files that provide Pythonic APIs
   - Handle imports and re-exports
   - Add convenience methods and properties

## Understanding the Architecture

### 1. MLIR Python Bindings Structure

MLIR provides Python bindings through a layered architecture:

```
MLIRPythonSources.Core/
├── ir.py                 # IR construction and manipulation
├── passmanager.py        # Pass management
├── extras/               # Additional utilities
└── _mlir_libs/          # Native extension modules
    ├── __init__.py      # Site initialization
    └── _mlir.so         # Compiled extension module
```

The `_mlir.so` extension module is compiled with submodules:
- `_mlir.ir` - IR manipulation classes
- `_mlir.passmanager` - Pass manager functionality
- etc.

### 2. Package Prefix Mechanism

MLIR uses `MLIR_PYTHON_PACKAGE_PREFIX` to control where bindings are installed:

- **CIRCT**: Sets `MLIR_PYTHON_PACKAGE_PREFIX=circt.`
  - Results in: `circt/ir.py`, `circt/_mlir_libs/_mlir.so`
  - Import as: `from circt import ir`

- **PyCDE**: Sets `MLIR_PYTHON_PACKAGE_PREFIX=pycde.circt.`
  - Results in: `pycde/circt/ir.py`, `pycde/circt/_mlir_libs/_mlir.so`
  - Import as: `from pycde.circt import ir`

### 3. The _mlir Import Problem

The `_mlir_libs/__init__.py` file contains site initialization code that does:
```python
from ._mlir import ir  # This expects _mlir.ir to exist as a submodule
```

This works when:
- The package structure matches what `_mlir.so` was compiled with
- The import paths align with the module's internal structure

This fails when:
- The module is nested in unexpected ways
- The package prefix doesn't match compilation settings

### 4. How PyCDE Avoids _mlir Imports

PyCDE succeeds because:

1. **It reuses CIRCT's bindings**: PyCDE doesn't create its own `_mlir.so`
2. **Proper nesting**: Places CIRCT bindings at `pycde.circt.*`
3. **No direct _mlir access**: Always imports through the public API (`from .circt import ir`)

Example PyCDE structure:
```
pycde/
├── __init__.py          # Imports: from .circt import ir
├── circt/               # CIRCT bindings (from build)
│   ├── ir.py
│   ├── passmanager.py
│   └── _mlir_libs/
│       └── _mlir.so
└── dialects/            # PyCDE's dialect wrappers
    └── hw.py            # Imports: from ..circt.dialects import hw
```

## Sharp/PySharp Binding Architecture Issues

### Current Problems

1. **Nested Module Structure**: PySharp tries to nest everything under `pysharp.sharp.*`
2. **Duplicate Bindings**: Both Sharp and PySharp try to create their own MLIR bindings
3. **Import Path Mismatch**: The `_mlir.so` module can't find its submodules due to nesting

### Why It Fails

When PySharp creates this structure:
```
pysharp/
└── sharp/
    ├── ir.py
    └── _mlir_libs/
        └── _mlir.so
```

The `_mlir.so` was compiled expecting to be at the top level or with a specific prefix, not nested under `pysharp.sharp`.

## Recommended Solution

### Option 1: Sibling Package Structure (Recommended)

Structure PySharp and Sharp as sibling packages:

```
python_packages/
├── sharp/                 # Sharp's MLIR bindings
│   ├── ir.py
│   ├── dialects/
│   └── _mlir_libs/
└── pysharp/              # PySharp frontend
    ├── __init__.py       # Imports: from sharp import ir
    └── dialects/
```

Benefits:
- No nesting issues
- Clear separation of concerns
- Works with standard MLIR package prefix

### Option 2: Follow PyCDE Pattern Exactly

Set `MLIR_PYTHON_PACKAGE_PREFIX=pysharp.sharp.` and ensure:
1. Only one set of MLIR bindings is built
2. All imports go through the public API
3. No direct `_mlir` references

### Option 3: Custom Import Hooks

Create import hooks to redirect imports, but this is complex and fragile.

## Implementation Guidelines

### For Sharp Bindings

1. Keep them minimal - just expose the dialects
2. Don't duplicate MLIR core functionality
3. Use standard MLIR CMake macros

### For PySharp Frontend

1. Import from Sharp (or bundled MLIR) bindings
2. Never import `_mlir` directly
3. Use the public API (`ir`, `passmanager`, etc.)
4. Follow PyCDE's pattern for dialect wrappers

## Example Fix

Instead of:
```python
# pysharp/__init__.py
from .sharp import ir  # This eventually tries to import _mlir
```

Use:
```python
# pysharp/__init__.py
import sharp  # Import sibling package
from sharp import ir
```

Or bundle everything properly:
```python
# pysharp/__init__.py
from . import mlir  # Properly bundled MLIR with correct prefix
from .mlir import ir
```

## Build System Details

### CMake Macros for Python Bindings

1. **`declare_mlir_python_sources`**
   ```cmake
   declare_mlir_python_sources(MyProjectPythonSources
     ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python"
     SOURCES
       my_module/__init__.py
       my_module/types.py
   )
   ```
   - Declares Python source files
   - Sets up installation rules
   - Handles package structure

2. **`declare_mlir_dialect_python_bindings`**
   ```cmake
   declare_mlir_dialect_python_bindings(
     ADD_TO_PARENT MyProjectPythonSources.Dialects
     TD_FILE dialects/MyDialect.td
     DIALECT_NAME my_dialect
   )
   ```
   - Generates dialect bindings from TableGen
   - Creates both `.py` and C++ extension files
   - Integrates with parent source set

3. **`add_mlir_python_extension`**
   ```cmake
   add_mlir_python_extension(MyExtensionModule
     MODULE_NAME _my_extension
     SOURCES
       MyExtension.cpp
     LINK_LIBS
       MLIRC  # Links against MLIR C API
   )
   ```
   - Builds C++ extension modules
   - Links against MLIR libraries
   - Handles platform-specific extensions

4. **`add_mlir_python_modules`**
   ```cmake
   add_mlir_python_modules(MyProject
     ROOT_PREFIX "${PYTHON_PACKAGES_DIR}/"
     INSTALL_PREFIX "python_packages/"
     DECLARED_SOURCES
       MyProjectPythonSources
   )
   ```
   - Final packaging step
   - Creates importable Python package
   - Sets up proper directory structure

### What Gets Generated

1. **Directory Structure**:
   ```
   build/python_packages/
   └── my_project/
       ├── __init__.py
       ├── ir.py                    # Re-exported from MLIR
       ├── dialects/
       │   ├── __init__.py
       │   ├── my_dialect.py        # Generated from TD
       │   └── _my_dialect_ops_gen.py
       └── _mlir_libs/
           ├── __init__.py          # Site initialization
           ├── _mlir.so             # Core MLIR extension
           └── _my_extension.so     # Custom extensions
   ```

2. **Generated Files**:
   - **`*_ops_gen.py`**: Operation definitions from TableGen
   - **`*.inc`**: C++ headers for dialect registration
   - **`*.so`/`.pyd`**: Compiled extension modules
   - **`__pycache__`**: Python bytecode cache

3. **Import Hooks and Site Initialization**:
   ```python
   # _mlir_libs/__init__.py
   import sys
   import importlib
   _site_packages = [...]
   for sp in _site_packages:
       if sp not in sys.path:
           sys.path.insert(0, sp)
   ```

### Compilation Process

1. **TableGen Phase**:
   - Parse `.td` files
   - Generate Python operation classes
   - Create C++ registration code

2. **C++ Compilation**:
   - Compile extension modules with nanobind/pybind11
   - Link against MLIR C API libraries
   - Include RTTI for cross-module type information

3. **Python Packaging**:
   - Copy pure Python files
   - Place extension modules in `_mlir_libs`
   - Generate `__init__.py` files

## Integration with CIRCT

CIRCT extends MLIR's binding infrastructure:

1. **Additional Dialects**:
   - HW, Comb, Seq, SV, FSM, etc.
   - Each gets its own Python bindings
   - Follows same TableGen → Python pattern

2. **Custom Extensions**:
   - `_circt` module for CIRCT-specific functionality
   - Additional passes and transformations
   - Integration with external tools

3. **PyCDE's Approach**:
   - Builds on top of CIRCT bindings
   - Adds hardware construction DSL
   - Provides higher-level abstractions

## Sharp's Implementation

Sharp follows the established pattern with some modifications:

1. **Package Prefix**:
   ```cpp
   // In CMakeLists.txt
   add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=sharp.")
   ```

2. **Dialect Registration**:
   ```cpp
   // In SharpModule.cpp
   void populateSharpPythonModule(py::module &m) {
     m.def("register_dialects", [](MlirContext context) {
       MlirDialectHandle txn = mlirGetDialectHandle__txn__();
       mlirDialectHandleRegisterDialect(txn, context);
     });
   }
   ```

3. **PySharp Integration**:
   - Uses sibling package approach
   - Imports from `sharp` package
   - Adds high-level construction API

## Common Issues and Solutions

### Issue: ImportError: cannot import name '_mlir'
**Cause**: Package structure mismatch
**Solution**: Ensure package prefix matches CMake configuration

### Issue: Symbol not found in .so file
**Cause**: Missing RTTI or ABI mismatch
**Solution**: Enable RTTI, ensure consistent compiler flags

### Issue: Dialect not registered
**Cause**: Missing dialect registration call
**Solution**: Add registration in module initialization

## Key Takeaways

1. **MLIR bindings use nanobind/pybind11 for C++ → Python**
2. **TableGen generates Python operation definitions**
3. **CMake macros automate the build process**
4. **Package structure must match compilation settings**
5. **The _mlir module is internal infrastructure**
6. **CIRCT and PyCDE build on MLIR's foundation**
7. **Sharp follows the established patterns with custom dialects**