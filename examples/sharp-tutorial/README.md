# Sharp Tutorial

This tutorial walks through the Sharp framework step-by-step, demonstrating all major features through practical examples.

## Overview

Sharp is a transaction-based hardware description language built on MLIR. It provides:
- Transaction-level modeling with atomic semantics
- Conflict matrix-based scheduling
- Multiple simulation levels (TL, RTL, hybrid)
- Python frontend for hardware construction
- Analysis passes for verification and optimization

## Tutorial Structure

Each chapter builds on the previous ones:

1. **Chapter 1: Basic Concepts** - Introduction to Sharp's transaction model
2. **Chapter 2: Modules and Methods** - Creating hardware modules with value/action methods
3. **Chapter 3: Scheduling and Conflicts** - Understanding conflict matrices and scheduling
4. **Chapter 4: Primitives** - Using Register, Wire, and other hardware primitives
5. **Chapter 5: Analysis Passes** - Running verification and optimization passes
6. **Chapter 6: Translation** - Converting to FIRRTL and Verilog
7. **Chapter 7: Simulation** - Transaction-level and RTL simulation
8. **Chapter 8: Python Frontend** - Using PySharp for hardware construction

## Prerequisites

- Sharp framework built and installed
- Basic understanding of hardware description languages
- Familiarity with MLIR concepts (helpful but not required)

## Getting Started

Navigate to each chapter directory and follow the README instructions:

```bash
cd chapter1-basics
cat README.md
```

Each chapter includes:
- Conceptual overview
- Example code (MLIR and/or Python)
- Build and run scripts
- Exercises to reinforce learning