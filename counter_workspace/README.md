# Counter Simulation

This is a generated transaction-level simulation of the `Counter` module.

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Running

```bash
./build/Counter_sim [options]
```

### Options

- `--cycles <n>`: Run simulation for n cycles (default: 100)
- `--verbose`: Enable verbose output
- `--stats`: Print performance statistics
- `--help`: Show help message

## Module Description

The module contains the following methods:

- **getValue** (value method): Returns 1 value(s)
- **increment** (action method): Modifies state
- **autoIncrement** (rule): Executes automatically when enabled

## Generated from Sharp

This simulation was generated using the Sharp framework's `--sharp-simulate` pass.
