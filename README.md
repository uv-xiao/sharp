# SHARP = Stacking Hardware, Architecture, and Programming

## Description

This framework aims to bridge description, optimization, and implementation of hardware, architecture, and software.

The short-term plan is to develop individual IRs for architecture and hardware. They are implemented as MLIR dialects and interact with CIRCT for RTL generation and other tasks.

> [!NOTE]
> Ultimate Goal: Unified IR for multi-purpose architecture design, synthesized for hardware and programming model generation. 

## Quick Start

```bash
# Install pixi and build Sharp
curl -fsSL https://pixi.sh/install.sh | bash
git clone <your-repo-url> sharp
cd sharp
pixi install
pixi run build

# Run sharp-opt
./build/bin/sharp-opt <input.mlir>
```

## Development with VSCode

The project includes comprehensive VSCode configuration for an optimal development experience:

1. **Open as Workspace** (Recommended):
   ```bash
   code .vscode/sharp.code-workspace
   ```
   This provides a multi-root workspace with Sharp, CIRCT, MLIR, and LLVM folders.

2. **Open as Folder**:
   ```bash
   code .
   ```
   Uses the settings in `.vscode/settings.json` for single-folder mode.

The VSCode configuration includes:
- C/C++ IntelliSense with clang-20
- CMake integration with Ninja
- MLIR/TableGen language support
- Debugging configurations for sharp-opt
- Code formatting with clang-format
- Python environment for MLIR/CIRCT bindings
- Build tasks integration with pixi

Install recommended extensions when prompted for the best experience.

## Documentation

- **[USAGE.md](./USAGE.md)** - Comprehensive guide for setup, building, and using Sharp
- **[docs/](./docs/)** - Additional technical documentation

## Project Status

- ✅ Build system with clang-20 and unified LLVM/MLIR/CIRCT build
- ✅ sharp-opt tool successfully built and working
- ✅ Initial Sharp Core dialect with basic operations
- ✅ Python bindings following CIRCT's pattern
- ⚠️ Sharp dialect parser needs minor adjustments
- ⚠️ Python bindings have a runtime issue under investigation
- 🚧 More dialect operations and transformations in development

See [USAGE.md](./USAGE.md#current-status) for detailed status information.