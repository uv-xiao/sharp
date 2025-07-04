//===- Passes.h - Sharp Simulation Passes -------------------------*- C++ -*-===//
//
// This header declares simulation-related passes for Sharp.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_SIMULATION_PASSES_H
#define SHARP_SIMULATION_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace sharp {

/// Options for the TxnSimulate pass
struct TxnSimulateOptions {
  enum class Mode {
    Translation,  // Generate C++ code
    JIT          // JIT compile and execute
  };
  
  Mode mode = Mode::JIT;
  std::string outputFile = "";  // For translation mode
  bool verbose = false;
  bool dumpStats = false;
  unsigned maxCycles = 1000000;  // Maximum simulation cycles
};

// Generate pass declarations
#define GEN_PASS_DECL
#include "sharp/Simulation/Passes.h.inc"

/// Registration
void registerSimulationPasses();

} // namespace sharp

#endif // SHARP_SIMULATION_PASSES_H