# GEMINI.md

This file provides guidance to Gemini when working with code in this repository.

## Using Gemini CLI for Large Codebase Analysis

When analyzing large codebases or multiple files that might exceed context limits, use the Gemini CLI with its massive
context window. Use `gemini -p` to leverage Google Gemini's large context capacity.

### File and Directory Inclusion Syntax

Use the `@` syntax to include files and directories in your Gemini prompts. The paths should be relative to WHERE you run the
  gemini command:

#### Examples:

**Single file analysis:**
gemini -p "@src/main.py Explain this file's purpose and structure"

Multiple files:
gemini -p "@package.json @src/index.js Analyze the dependencies used in the code"

Entire directory:
gemini -p "@src/ Summarize the architecture of this codebase"

Multiple directories:
gemini -p "@src/ @tests/ Analyze test coverage for the source code"

Current directory and subdirectories:
gemini -p "@./ Give me an overview of this entire project"

### Or use --all_files flag:
gemini --all_files -p "Analyze the project structure and dependencies"

### Implementation Verification Examples

Check if a feature is implemented:
gemini -p "@src/ @lib/ Has dark mode been implemented in this codebase? Show me the relevant files and functions"

Verify authentication implementation:
gemini -p "@src/ @middleware/ Is JWT authentication implemented? List all auth-related endpoints and middleware"

Check for specific patterns:
gemini -p "@src/ Are there any React hooks that handle WebSocket connections? List them with file paths"

Verify error handling:
gemini -p "@src/ @api/ Is proper error handling implemented for all API endpoints? Show examples of try-catch blocks"

Check for rate limiting:
gemini -p "@backend/ @middleware/ Is rate limiting implemented for the API? Show the implementation details"

Verify caching strategy:
gemini -p "@src/ @lib/ @services/ Is Redis caching implemented? List all cache-related functions and their usage"

Check for specific security measures:
gemini -p "@src/ @api/ Are SQL injection protections implemented? Show how user inputs are sanitized"

Verify test coverage for features:
gemini -p "@src/payment/ @tests/ Is the payment processing module fully tested? List all test cases"

When to Use Gemini CLI

Use gemini -p when:
- Analyzing entire codebases or large directories
- Comparing multiple large files
- Need to understand project-wide patterns or architecture
- Current context window is insufficient for the task
- Working with files totaling more than 100KB
- Verifying if specific features, patterns, or security measures are implemented
- Checking for the presence of certain coding patterns across the entire codebase

Important Notes

- Paths in @ syntax are relative to your current working directory when invoking gemini
- The CLI will include file contents directly in the context
- No need for --yolo flag for read-only analysis
- Gemini's context window can handle entire codebases that would overflow Claude's context
- When checking implementations, be specific about what you're looking for to get accurate results

## Essential Commands

### Root Directory

```bash
pwd # should end with /sharp
```
When cannot find the required file or directory, you can use `cd` to change the current working directory to the root directory of the project.

You can also use `pixi info`, extract `Manifest file` and get the base directory of the pixi file as the root directory.


### Build Commands
```bash
pixi run build              # Standard quiet build
pixi run build-unified-verbose  # Verbose build with full output
pixi run build-release      # Release mode build
pixi run clean             # Clean build artifacts
```

### Testing Commands

#### Standard Test Commands
```bash
# Run all tests (lit + unit tests if available)
pixi run test              # Runs: cmake --build build --target check-sharp

# Run lit tests only (most commonly used)
pixi run test-lit          # Runs: cmake --build build --target check-sharp-lit

# Run unit tests only (requires gtest support in LLVM build)
pixi run test-unit         # Runs check-sharp-unit target if available
```

#### Running Individual Tests

**From the Sharp root directory:**
```bash
# Using lit.py directly (most reliable method)
cd build && ../circt/llvm/llvm/utils/lit/lit.py test/Dialect/Txn/basic.mlir -v

# Using pixi environment lit
cd build && /home/uvxiao/sharp/.pixi/envs/default/bin/lit test/Dialect/Txn/basic.mlir -v

# Run all tests in a directory
cd build && ../circt/llvm/llvm/utils/lit/lit.py test/Dialect/Txn/ -v

# Run tests matching a pattern
cd build && ../circt/llvm/llvm/utils/lit/lit.py test/ -v --filter "conflict"
```

**Debug a test by running sharp-opt directly:**
```bash
# From Sharp root directory
./build/bin/sharp-opt test/Dialect/Txn/basic.mlir

# With specific passes
./build/bin/sharp-opt test/Dialect/Txn/basic.mlir --sharp-infer-conflict-matrix

# Allow unregistered dialects (for tests with mock operations)
./build/bin/sharp-opt test/Dialect/Txn/basic.mlir -allow-unregistered-dialect
```

#### Test Infrastructure Details

- **Test Runner**: Uses LLVM's lit (LLVM Integrated Tester)
- **Test Location**: All tests are in `test/` directory
- **Test Format**: `.mlir` files with RUN lines at the top
- **FileCheck**: Located at `.install/unified/bin/FileCheck`
- **Working Directory**: Tests should be run from `build/` directory; move back to the root directory of the project after running a test.

#### Common Test Patterns

```mlir
// Basic test
// RUN: sharp-opt %s | FileCheck %s

// Test with specific pass
// RUN: sharp-opt %s --sharp-infer-conflict-matrix | FileCheck %s

// Test expecting failure
// RUN: not sharp-opt %s --verify-diagnostics 2>&1 | FileCheck %s

// Test with multiple check prefixes
// RUN: sharp-opt %s | FileCheck %s --check-prefix=CHECK
// RUN: sharp-opt %s --convert-txn-to-firrtl | FileCheck %s --check-prefix=FIRRTL
```

#### Python Test Execution

```bash
# Python tests are integrated into lit test suite
# Run from build directory:
cd build && ../circt/llvm/llvm/utils/lit/lit.py test/python/ -v

# Or run Python test directly (from Sharp root)
PYTHONPATH=build/python_packages:build/python_packages/pysharp python test/python/pysharp/test_pysharp.py
```

#### Troubleshooting Tests

1. **"unregistered dialect" error**: Add `-allow-unregistered-dialect` flag
2. **FileCheck not found**: Use full path `.install/unified/bin/FileCheck`
3. **lit not found**: Use `../circt/llvm/llvm/utils/lit/lit.py` from build directory
4. **Tests not discovered**: Ensure you're in the `build/` directory
5. **Python import errors**: Set PYTHONPATH to include build/python_packages

### Development Commands
```bash
pixi run format            # Format C++ code with clang-format
pixi run tidy             # Run clang-tidy static analysis
pixi run shell            # Enter development shell with proper environment

# Run sharp-opt (the main compiler tool)
./build/bin/sharp-opt <input.mlir>
pixi run run-opt

# VSCode development
./dev-vscode.sh    # Opens configured VSCode workspace
```

## Architecture Overview

### Dialect Structure
Sharp implements custom MLIR dialects following CIRCT patterns:

1. **Sharp Core Dialect** (`include/sharp/Dialect/Core/`)
   - Basic infrastructure and common operations
   - Currently implements constant operations

2. **Sharp Txn Dialect** (`include/sharp/Dialect/Txn/`)
   - Transaction-based hardware description
   - Key concepts: Modules, Methods (value/action), Rules, Scheduling
   - Inspired by Fjfj language for concurrent hardware verification
   - Conflict matrix support: SB=0, SA=1, C=2, CF=3
   - Timing attributes: "combinational", "static(n)", "dynamic"
   - See `docs/txn.md` for detailed semantics

### Key Implementation Patterns

**Adding New Operations:**
1. Define the operation in `include/sharp/Dialect/*/IR/*Ops.td`
2. Implement verifiers/builders in `lib/Dialect/*/IR/*Ops.cpp`
3. Add tests in `test/Dialect/*/`
4. Update Python bindings if needed in `lib/Bindings/Python/`

**Conversion Passes:**
- Conversions go in `lib/Conversion/`
- Follow the pattern: source dialect → target dialect
- Register in `include/sharp/Conversion/Passes.td`
- Main pass implementation in `lib/Conversion/SourceToTarget/SourceToTargetPass.cpp`

**Analysis Passes:**
- Implement in `lib/Analysis/`
- Register in `include/sharp/Analysis/Passes.td`

### Build System Details

- Uses CMake with Ninja generator
- Managed via Pixi package manager
- All dependencies (LLVM/MLIR/CIRCT) built from source using Clang-20
- Build artifacts in `.install/unified/`
- Sharp-specific builds in `build/`

### Testing Infrastructure

- Uses LLVM lit with FileCheck
- Integration tests in `test/` mirror the source structure
- Test files use `.mlir` extension with RUN lines
- Common test patterns:
  ```mlir
  // RUN: sharp-opt %s | FileCheck %s
  // RUN: sharp-opt --convert-txn-to-firrtl %s | FileCheck %s
  // RUN: sharp-opt %s -sharp-infer-conflict-matrix | FileCheck %s
  ```
- FileCheck directives:
  - `CHECK-LABEL:` - Anchors to specific operations
  - `CHECK:` - Exact matching
  - `CHECK-DAG:` - Order-independent matching
  - `CHECK-NEXT:` - Must appear on next line

### Analysis Passes Available

- `--sharp-infer-conflict-matrix` - Infers conflict relationships between actions
- `--sharp-validate-method-attributes` - Validates signal names and method constraints
- `--sharp-reachability-analysis` - Computes reachability conditions for method calls (adds condition operands to txn.call)
- `--sharp-pre-synthesis-check` - Checks for non-synthesizable constructs
- `--convert-txn-to-firrtl` - Converts Txn modules to FIRRTL (includes conflict_inside calculation)

### Current Development Status

Per `STATUS.md`:
- **Completed**: 
  - Full Txn dialect with all operations and analysis passes
  - Txn-to-FIRRTL conversion with conflict resolution
  - Verilog export through CIRCT pipeline
  - Comprehensive simulation infrastructure (TL, RTL, JIT, Hybrid)
  - Complete 8-chapter tutorial with examples
  - All hardware primitives (Register, Wire, FIFO, Memory, SpecFIFO, SpecMemory)
  - Python frontend following PyCDE pattern
- **In Progress**: Nothing currently
- **Future Work**: JIT lowering fixes, enhanced tooling

### Development Diary

Claude Code should note down the development progress in `DIARY.md`. It should be updated every claude code session (how the user interacts with the claude code and what the claude code does).


## Key Documentation Files

- `STATUS.md` - Current implementation status and completed features
- `docs/txn.md` - Transaction dialect semantics and operation definitions
- `docs/txn_to_firrtl.md` - Conversion algorithm documentation
- `docs/txn_primitive.md` - Primitive infrastructure documentation
- `docs/firrtl_operations_guide.md` - FIRRTL operations reference
- `docs/simulation.md` - Comprehensive simulation methodology
- `docs/test.md` - Testing infrastructure documentation
- `examples/sharp-tutorial/` - 8-chapter progressive tutorial

## Critical Implementation Insights

### MLIR Development Guide - Hard-Earned Lessons

#### 1. **TableGen Operation Definitions**

**Common Pitfalls and Solutions:**

1. **Namespace Resolution Issues**
   - **Problem**: TableGen-generated code in `sharp::txn` namespace can't find MLIR types
   - **Solution**: Use fully-qualified types in TableGen:
   ```tablegen
   let extraClassDeclaration = [{
     ::mlir::Value getCondition() { return getOperation()->getOperand(0); }
     ::mlir::Block& getBody() { return getRegion().front(); }
     static ::mlir::OpBuilder::InsertPoint createBuilder(::mlir::OpBuilder &builder);
   }];
   ```

2. **Operation Constructor Issues**
   - **Problem**: Missing optional attributes cause build errors
   - **Solution**: Always provide all attributes, use empty constructors for optional:
   ```cpp
   // Wrong
   auto op = builder.create<FutureOp>(loc);
   
   // Correct
   auto op = builder.create<FutureOp>(loc, 
     /*attributes=*/builder.getDictionaryAttr({}),
     /*properties=*/nullptr);
   ```

3. **Region and Block Handling**
   - Use `SizedRegion<1>` instead of `AnyRegion` for better code generation
   - Add proper traits: `SingleBlock`, `NoTerminator`, or `SingleBlockImplicitTerminator`
   - For operations with regions, define builders in C++:
   ```cpp
   void FutureOp::build(OpBuilder &builder, OperationState &result) {
     Region *region = result.addRegion();
     Block *block = builder.createBlock(region);
     builder.setInsertionPointToStart(block);
   }
   ```

#### 2. **Pass Implementation Patterns**

**Structure:**
```cpp
// In Passes.td
def MyPass : Pass<"sharp-my-pass", "mlir::ModuleOp"> {
  let summary = "...";
  let constructor = "mlir::sharp::createMyPass()";
  let dependentDialects = ["::sharp::txn::TxnDialect"];
}

// In MyPass.cpp
#define GEN_PASS_DEF_MYPASS
#include "sharp/Analysis/Passes.h.inc"

namespace {
class MyPass : public impl::MyPassBase<MyPass> {
  void runOnOperation() override {
    // Implementation
  }
};
}

std::unique_ptr<mlir::Pass> mlir::sharp::createMyPass() {
  return std::make_unique<MyPass>();
}
```

**Common Issues:**
- Forgetting `#define GEN_PASS_DEF_*` before include
- Not implementing the constructor function
- Missing dependent dialects causing undefined references

#### 3. **CMakeLists.txt Patterns**

**Correct dependency management:**
```cmake
add_mlir_dialect_library(SharpTxn
  TxnDialect.cpp
  TxnOps.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${SHARP_MAIN_INCLUDE_DIR}/sharp/Dialect/Txn
  
  DEPENDS
  MLIRTxnIncGen  # TableGen targets
  
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  # List ALL dependencies explicitly
)
```

**Common Build Errors:**
- **Undefined references**: Missing LINK_LIBS entries
- **Include errors**: Wrong ADDITIONAL_HEADER_DIRS
- **TableGen errors**: Missing DEPENDS on *IncGen targets

#### 4. **Symbol References and Method Calls**

**MLIR Symbol Reference Syntax:**
```mlir
// Nested symbol reference (instance method call)
%result = txn.call @instance::@method(%arg) : (i32) -> i32

// Direct symbol reference (module-level method)
%result = txn.call @method(%arg) : (i32) -> i32
```

**In C++ handling:**
```cpp
auto callee = callOp.getCallee();
StringRef rootRef = callee.getRootReference().getValue();  // "instance"
StringRef leafRef = callee.getLeafReference().getValue();  // "method"
```

#### 5. **Python Bindings**

**Key Learning**: Extend existing bindings, don't create parallel structures
```cmake
# Correct approach
mlir_configure_python_dev_packages()
declare_mlir_python_extension(SharpPythonExtension
  MODULE_NAME _sharp
  ADD_TO_PARENT circt.dialects  # Extend CIRCT
  SOURCES
    SharpModule.cpp
)
```

#### 6. **Testing Patterns**

**FileCheck best practices:**
```mlir
// RUN: sharp-opt %s --my-pass | FileCheck %s
// RUN: not sharp-opt %s --verify-diagnostics 2>&1 | FileCheck %s --check-prefix=ERROR

// CHECK-LABEL: txn.module @MyModule
// CHECK-NEXT: %[[REG:.*]] = txn.instance
// CHECK: txn.call @[[REG]]::@read()

// ERROR: error: expected failure message
```

#### 7. **Common MLIR Patterns to Follow**

1. **Include Order**: Project headers → MLIR headers → LLVM headers → Standard headers
2. **Namespace Usage**: 
   ```cpp
   using namespace mlir;
   using namespace sharp;
   using namespace sharp::txn;  // After includes
   ```
3. **Pass Registration**: Use anonymous namespace for pass class
4. **Error Handling**: Use `emitError()` on operations, not `llvm::errs()`
5. **Type Handling**: Prefer `cast<>` over `dyn_cast<>` when type is known

#### 8. **Debugging Build Issues**

**When you get cryptic build errors:**

1. **"unknown type name" in generated code**
   - Check if you need `::mlir::` prefix in TableGen
   - Verify all necessary includes in the .h file
   - Example fix: `Region&` → `::mlir::Region&`

2. **"undefined reference to vtable"**
   - Missing method implementation
   - Forgot to include generated .cpp.inc file
   - Check if operation needs custom parser/printer

3. **"no member named 'getODSOperands'"**
   - TableGen generation issue
   - Check operation traits and base class
   - Verify `arguments` and `results` definitions

4. **Iterator/accessor errors**
   - Wrong accessor pattern for regions/blocks
   - Use `getBody().front()` not `getBody()[0]`
   - Check iterator vs direct access

#### 9. **Operation Definition Checklist**

When adding a new operation:
- [ ] Define in Ops.td with proper traits
- [ ] Add to CMakeLists.txt DEPENDS
- [ ] Implement verifier if `hasVerifier = 1`
- [ ] Implement builders if has regions
- [ ] Add parser/printer if `hasCustomAssemblyFormat = 1`
- [ ] Create test file in test/Dialect/
- [ ] Update Python bindings if user-facing
- [ ] Document in relevant .md files

#### 10. **Real Examples from Sharp Development**

**LaunchOp Implementation Journey:**
1. Started with simple TableGen definition
2. Hit namespace issues → added `::mlir::` prefixes
3. Region accessor problems → changed to `SizedRegion<1>`
4. Builder issues → implemented custom build() method
5. Assembly format conflicts → simplified syntax
6. Optional attribute handling → used `OptionalAttr<I32Attr>`

**Key lesson**: Start simple, build incrementally, test at each step

### Primitive Implementation Pattern
When implementing new primitives:
1. Use correct operation constructors with all optional parameters
2. Mark spec primitives with `spec` attribute
3. Include `software_semantics` dictionary attribute for simulation
4. Follow the pattern in Register.cpp for hardware primitives
5. Follow the pattern in Memory.cpp for spec primitives

### Common Pitfalls and Solutions
1. **Operation Build Errors**: Always provide all optional attributes (use StringAttr(), ArrayAttr(), UnitAttr() for empty)
2. **Primitive Instance Types**: Use parametric syntax `@instance of @Primitive<Type>`
3. **Method Calls**: Use `::` syntax for instance method calls
4. **FileCheck Tests**: Use `not` command for tests expecting failure, redirect stderr with `2>&1`
5. **Optional Handling**: Use `.value_or()` or check `.has_value()` before `.value()`
6. **Builder Context**: Always set insertion point when creating blocks

### Testing Best Practices
- Run `pixi run test` frequently during development
- Use `-DSHARP_BINDINGS_PYTHON_ENABLED=OFF` if Python bindings cause issues
- Write both positive and negative tests
- Follow existing test patterns in `test/Dialect/Txn/`

### Code Generation Insights
When adding support for new operations in simulation:
1. Check both `.method` and plain `method` syntax for primitive calls
2. Handle arithmetic operations in the appropriate visitor method
3. Update state variable names to match generated code patterns
4. Test with workspace generation tool to verify end-to-end flow

### Collaboration Patterns
- User often requests "move forward STATUS.md" - check STATUS.md for next tasks
- Always update both STATUS.md and DIARY.md after completing work
- Create comprehensive tests for all new features
- Document everything in appropriate docs/ files
- When implementing features, also create tutorial examples
