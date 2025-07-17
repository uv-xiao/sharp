# Sharp - Transaction-based Hardware Description Language

## Overview

Sharp is a transaction-based hardware description language built on MLIR that enables conflict-free hardware design through explicit conflict matrices and transaction-level modeling. Inspired by Bluespec and Koika, Sharp provides high-level abstractions while ensuring synthesizability to FIRRTL and Verilog.

**Key Features:**
- Transaction-level modeling with automatic conflict resolution
- Multiple simulation modes (TL, RTL, JIT, concurrent)
- Python frontend for programmatic hardware generation
- Complete FIRRTL/Verilog generation pipeline
- Comprehensive analysis and validation passes

## Quick Start

```bash
# Build Sharp
curl -fsSL https://pixi.sh/install.sh | bash
git clone https://github.com/uv-xiao/sharp.git
cd sharp
pixi install
pixi run build

# Run tests
pixi run test  # 98/102 tests passing

# Try an example
./build/bin/sharp-opt examples/sharp-tutorial/chapter1/toggle.mlir
```

## Example

```mlir
txn.module @Counter {
  %reg = txn.instance @count of @Register<i32> : index
  
  txn.rule @increment {
    %val = txn.call @count::@read() : () -> i32
    %one = arith.constant 1 : i32
    %next = arith.addi %val, %one : i32
    txn.call @count::@write(%next) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@increment]
}
```

## Documentation

- **[Tutorial](examples/sharp-tutorial/)** - 8-chapter hands-on guide
- **[Txn Dialect](docs/txn.md)** - Language specification
- **[Execution Model](docs/execution_model.md)** - Three-phase execution semantics
- **[Python Frontend](docs/pythonic_frontend.md)** - PySharp API guide

## Project Structure

```
sharp/
├── lib/           # Core implementation
├── include/       # Headers and TableGen
├── test/          # Comprehensive test suite
├── docs/          # Technical documentation
├── examples/      # Tutorial and examples
└── frontends/     # Python frontend (PySharp)
```

## Development Status

See [STATUS.md](STATUS.md) for detailed feature status. Current highlights:
- ✅ Core dialect and primitives complete
- ✅ All major conversion passes implemented
- ✅ Multiple simulation modes working
- ✅ Python frontend following PyCDE pattern
- 🚧 Runtime Python binding fixes needed
- 📋 IDE support and tooling planned