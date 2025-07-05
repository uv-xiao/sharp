#include "SimulationBase.h"
#include <iostream>
#include <chrono>

void Simulator::run(int maxCycles) {
  if (verbose) {
    std::cout << "Starting simulation for " << maxCycles << " cycles\n";
  }

  auto startTime = std::chrono::high_resolution_clock::now();

  for (cycles = 0; cycles < maxCycles; ++cycles) {
    if (verbose && cycles % 100 == 0) {
      std::cout << "Cycle " << cycles << "\n";
    }

    // Execute rules for each module
    for (auto& module : modules) {
      // For now, just execute the autoIncrement rule if it can fire
      if (module->canFireRule("autoIncrement")) {
        module->callActionMethod("increment", {});
      }
    }
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

  if (dumpStats) {
    std::cout << "\nSimulation Statistics:\n";
    std::cout << "  Total cycles: " << cycles << "\n";
    std::cout << "  Execution time: " << duration.count() << " ms\n";
    std::cout << "  Cycles per second: " << (cycles * 1000.0 / duration.count()) << "\n";
  }
}
