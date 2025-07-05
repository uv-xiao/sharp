# Toggle Simulation

This is a generated transaction-level simulation of the `Toggle` module.

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Running

```bash
./build/Toggle_sim [options]
```

### Options

- `--cycles <n>`: Run simulation for n cycles (default: 100)
- `--verbose`: Enable verbose output
- `--stats`: Print performance statistics
- `--help`: Show help message

## Module Description

This simulation was generated from the MLIR input file using the Sharp framework's
`--sharp-simulate` pass in translation mode.

## Generated from Sharp

Generated on Sun Jul  6 01:09:05 AM CST 2025
Source: toggle.mlir
