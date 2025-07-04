//===- SimConfig.h - Simulation Configuration -----------------------------===//
//
// This file defines the configuration structures for simulation.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_SIMULATION_SIMCONFIG_H
#define SHARP_SIMULATION_SIMCONFIG_H

#include <cstdint>
#include <string>

namespace sharp {
namespace sim {

/// Basic simulation configuration
struct SimConfig {
  /// Maximum number of cycles to simulate
  uint64_t maxCycles = 1000000;
  
  /// Enable verbose output
  bool verbose = false;
  
  /// Enable performance tracking
  bool trackPerformance = true;
  
  /// VCD trace file (empty = no trace)
  std::string vcdFile;
  
  /// Random seed for deterministic simulation
  uint64_t seed = 42;
};

} // namespace sim
} // namespace sharp

#endif // SHARP_SIMULATION_SIMCONFIG_H