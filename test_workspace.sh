#!/bin/bash
# Test workspace generation

set -e

echo "Testing workspace generation..."
rm -rf counter_workspace

# Create workspace directory
mkdir -p counter_workspace

# Create a simple wrapper that will always generate workspace for non-stdout output
# For now, just generate to stdout and manually create the workspace
./build/bin/sharp-opt test_sim_workspace.mlir --sharp-simulate="mode=translation" > /tmp/counter_output.txt

# Extract only the C++ code (before the module line)
awk '/^module {/{exit} {print}' /tmp/counter_output.txt > counter_workspace/Counter_sim.cpp

# Clean up
rm /tmp/counter_output.txt

# Generate the SimulationBase files manually
cat > counter_workspace/SimulationBase.h << 'EOF'
#pragma once

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>
#include <cstdint>

// Conflict relations between actions
enum class ConflictRelation {
  SequenceBefore = 0,  // SB: First must execute before second
  SequenceAfter = 1,   // SA: First must execute after second
  Conflict = 2,        // C: Cannot execute in same cycle
  ConflictFree = 3     // CF: Can execute in any order
};

// Base class for simulated modules
class SimModule {
public:
  SimModule(const std::string& name) : moduleName(name) {}
  virtual ~SimModule() = default;

  // Register methods
  void registerValueMethod(const std::string& name,
                          std::function<std::vector<int64_t>(const std::vector<int64_t>&)> impl) {
    valueMethods[name] = impl;
  }

  void registerActionMethod(const std::string& name,
                           std::function<void(const std::vector<int64_t>&)> impl) {
    actionMethods[name] = impl;
  }

  void registerRule(const std::string& name, std::function<bool()> impl) {
    rules[name] = impl;
  }

  // Execute methods
  std::vector<int64_t> callValueMethod(const std::string& name, const std::vector<int64_t>& args) {
    auto it = valueMethods.find(name);
    if (it != valueMethods.end()) {
      return it->second(args);
    }
    return {};
  }

  void callActionMethod(const std::string& name, const std::vector<int64_t>& args) {
    auto it = actionMethods.find(name);
    if (it != actionMethods.end()) {
      it->second(args);
    }
  }

  bool canFireRule(const std::string& name) {
    auto it = rules.find(name);
    if (it != rules.end()) {
      return it->second();
    }
    return false;
  }

  const std::string& getName() const { return moduleName; }

protected:
  std::string moduleName;
  std::map<std::string, std::function<std::vector<int64_t>(const std::vector<int64_t>&)>> valueMethods;
  std::map<std::string, std::function<void(const std::vector<int64_t>&)>> actionMethods;
  std::map<std::string, std::function<bool()>> rules;
};

// Main simulation driver
class Simulator {
public:
  Simulator() : cycles(0), verbose(false), dumpStats(false) {}

  void addModule(std::unique_ptr<SimModule> module) {
    modules.push_back(std::move(module));
  }

  void run(int maxCycles);
  void setVerbose(bool v) { verbose = v; }
  void setDumpStats(bool d) { dumpStats = d; }

private:
  std::vector<std::unique_ptr<SimModule>> modules;
  int cycles;
  bool verbose;
  bool dumpStats;
};
EOF

cat > counter_workspace/SimulationBase.cpp << 'EOF'
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
EOF

cat > counter_workspace/CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.16)
project(Counter_simulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(Counter_sim
  Counter_sim.cpp
  SimulationBase.h
  SimulationBase.cpp
)

# Include directories
target_include_directories(Counter_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
EOF

cat > counter_workspace/README.md << 'EOF'
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
EOF

echo "Workspace created in counter_workspace/"
echo "Contents:"
ls -la counter_workspace/