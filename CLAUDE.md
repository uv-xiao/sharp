# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Or use --all_files flag:
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

### Build Commands
```bash
pixi run build              # Standard quiet build
pixi run build-unified-verbose  # Verbose build with full output
pixi run build-release      # Release mode build
pixi run clean             # Clean build artifacts
```

### Testing Commands
```bash
pixi run test              # Run all tests
pixi run test-lit          # Run lit tests only
pixi run test-unit         # Run unit tests only
pixi run test-python       # Test Python bindings

# Run a single test file
./build/bin/llvm-lit test/Dialect/Txn/basic.mlir -v

# Alternative: use the full path to llvm-lit from the unified build
/home/uvxiao/sharp/.install/unified-build/bin/llvm-lit test/Dialect/Txn/basic.mlir -v

# Debug a test by running sharp-opt directly
./build/bin/sharp-opt test/Dialect/Txn/basic.mlir
```

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

Claude Code should note down the development progress in `archieves/DIARY.md`. It should be updated every claude code session (how the user interacts with the claude code and what the claude code does). The user could provide some guidance files (like PLAN.md, etc.) to the claude code, and when claude code finished using the files and do not need them anymore, the files should be moved to `archieves/` directory with date marked (e.g., `archieves/2025-06-29-PLAN.md`). The `archieves/DIARY.md` entries should refer to the guidance files.


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