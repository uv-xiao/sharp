// Generated Txn Module Simulation
// Module: Toggle

#include "SimulationBase.h"
#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <string>
#include <functional>
#include <cassert>
#include <chrono>
#include <queue>

class ToggleModule : public SimModule {
public:
  ToggleModule() : SimModule("Toggle") {
    // Register methods
    registerValueMethod("read", 
      [this](const std::vector<int64_t>& args) -> std::vector<int64_t> {
        return read();
      });
    registerActionMethod("toggle", 
      [this](const std::vector<int64_t>& args) {
        toggle();
      });

    // Set schedule
    setSchedule({"read", "toggle"});
  }

  // Value method: read
  std::vector<int64_t> read() {
    int64_t _0 = state_data;
    return {_0};
  }

  // Action method: toggle
  void toggle() {
    int64_t _0 = state_data;
    int64_t _1 = -1;
    int64_t _2 = _0 ^ _1;
    state_data = _2;
  }

private:
  // Primitive state
  int32_t state_data = 0;
};

// Main function
int main(int argc, char* argv[]) {
  // Parse command line arguments
  int maxCycles = 100;
  bool verbose = false;
  bool dumpStats = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--cycles" && i + 1 < argc) {
      maxCycles = std::stoi(argv[++i]);
    } else if (arg == "--verbose") {
      verbose = true;
    } else if (arg == "--stats") {
      dumpStats = true;
    } else if (arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n";
      std::cout << "Options:\n";
      std::cout << "  --cycles <n>  Run for n cycles (default: 100)\n";
      std::cout << "  --verbose     Enable verbose output\n";
      std::cout << "  --stats       Dump performance statistics\n";
      std::cout << "  --help        Show this help message\n";
      return 0;
    }
  }

  // Create simulator
  Simulator sim;
  sim.setVerbose(verbose);
  sim.setDumpStats(dumpStats);

  // Create and add module
  auto module = std::make_unique<ToggleModule>();
  sim.addModule(std::move(module));

  // Run simulation
  sim.run(maxCycles);

  return 0;
}
