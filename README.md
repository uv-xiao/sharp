# SHARP = Stacking Hardware, Architecture, and Programming

## Description

Sharp is a transaction-based hardware description language built on MLIR that enables conflict-free hardware design through explicit conflict matrices and transaction-level modeling. Inspired by Bluespec and Koika, Sharp provides a high-level abstraction for hardware description while ensuring synthesizability to FIRRTL and Verilog.

The project implements custom MLIR dialects that integrate with CIRCT for RTL generation, offering a modern approach to hardware design with built-in conflict detection and resolution.

> [!NOTE]
> **Key Features:**
> - Transaction-level modeling with explicit conflict matrices
> - Automatic conflict detection and resolution
> - Parametric primitive types (Register<T>, Wire<T>)
> - FIRRTL generation with will-fire logic
> - Comprehensive analysis passes for synthesizability

## Quick Start

```bash
# Install pixi and build Sharp
curl -fsSL https://pixi.sh/install.sh | bash
git clone https://github.com/xuyang2/sharp.git
cd sharp
pixi install
pixi run build

# Run a simple example
./build/bin/sharp-opt test/Dialect/Txn/basic.mlir

# Convert Txn to FIRRTL
./build/bin/sharp-opt --convert-txn-to-firrtl test/Conversion/TxnToFIRRTL/counter.mlir
```

## Example: Simple Counter

```mlir
txn.module @Counter {
  // Instantiate a parametric register
  %count = txn.instance @count of @Register<i32> : !txn.module<"Register">
  
  txn.action_method @increment() {
    %current = txn.call @count::@read() : () -> i32
    %one = arith.constant 1 : i32
    %next = arith.addi %current, %one : i32
    txn.call @count::@write(%next) : (i32) -> ()
    txn.return
  }
  
  txn.value_method @getCount() -> i32 {
    %val = txn.call @count::@read() : () -> i32
    txn.return %val : i32
  }
  
  txn.schedule [@increment, @getCount] {
    conflict_matrix = {}
  }
}
```

## Development with VSCode

The project includes comprehensive VSCode configuration:

```bash
# Open as workspace (recommended)
code .vscode/sharp.code-workspace

# Or use the development script
./dev-vscode.sh
```

Features:
- C/C++ IntelliSense with clang-20
- MLIR/TableGen language support
- Debugging configurations for sharp-opt
- Integrated build tasks with pixi

## Documentation

- **[STATUS.md](./STATUS.md)** - Current implementation status and roadmap
- **[USAGE.md](./USAGE.md)** - Comprehensive setup and usage guide
- **[docs/txn_to_firrtl.md](./docs/txn_to_firrtl.md)** - Txn to FIRRTL conversion details
- **[docs/txn.md](./docs/txn.md)** - Transaction dialect specification

## Project Status

### ✅ Completed
- Txn dialect with modules, methods, rules, and scheduling
- Conflict matrix support with inference analysis
- Complete Txn-to-FIRRTL conversion pass
- Parametric primitive types (Register<T>, Wire<T>)
- Automatic primitive instantiation
- 45/45 tests passing

### 🚧 In Progress
- Additional hardware primitives (FIFO, Memory)
- Verilog export through CIRCT

### 📋 Planned
- Formal verification primitives
- Performance optimization passes
- IDE language server support

See [STATUS.md](./STATUS.md) for detailed implementation status.