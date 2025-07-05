#!/bin/bash

# Generate simulation workspace from MLIR input
# Usage: ./generate-workspace.sh <input.mlir> <output_dir>

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input.mlir> <output_dir>"
    echo "Example: $0 counter.mlir counter_workspace"
    exit 1
fi

INPUT_FILE=$1
OUTPUT_DIR=$2
SHARP_ROOT=$(dirname $(dirname $(realpath $0)))

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

# Create temporary file for the pass output
TEMP_OUTPUT=$(mktemp)

echo "Generating simulation workspace for $INPUT_FILE..."

# Run the simulation pass in translation mode
# The pass currently outputs to stdout, so we capture it
"$SHARP_ROOT/build/bin/sharp-opt" "$INPUT_FILE" \
    --sharp-simulate="mode=translation" > "$TEMP_OUTPUT" 2>&1

# Extract just the C++ code (before the MLIR module output)
mkdir -p "$OUTPUT_DIR"
awk '/^module {/{exit} {print}' "$TEMP_OUTPUT" > "$OUTPUT_DIR/simulation.cpp"

# Get module name from the C++ code
MODULE_NAME=$(grep -o "class \w\+Module" "$OUTPUT_DIR/simulation.cpp" | head -1 | sed 's/class \(.*\)Module/\1/')

if [ -z "$MODULE_NAME" ]; then
    echo "Error: Could not determine module name"
    rm -rf "$OUTPUT_DIR"
    rm "$TEMP_OUTPUT"
    exit 1
fi

# Rename the C++ file
mv "$OUTPUT_DIR/simulation.cpp" "$OUTPUT_DIR/${MODULE_NAME}_sim.cpp"

# Generate CMakeLists.txt
cat > "$OUTPUT_DIR/CMakeLists.txt" << EOF
cmake_minimum_required(VERSION 3.16)
project(${MODULE_NAME}_simulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(${MODULE_NAME}_sim
  ${MODULE_NAME}_sim.cpp
  SimulationBase.h
  SimulationBase.cpp
)

# Include directories
target_include_directories(${MODULE_NAME}_sim PRIVATE \${CMAKE_CURRENT_SOURCE_DIR})
EOF

# Generate SimulationBase.h
cat > "$OUTPUT_DIR/SimulationBase.h" << 'EOF'
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

# Generate SimulationBase.cpp
cat > "$OUTPUT_DIR/SimulationBase.cpp" << 'EOF'
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
      // Execute all rules that can fire
      if (module->canFireRule("autoIncrement")) {
        module->callActionMethod("increment", {});
      }
      // TODO: Add more sophisticated rule scheduling
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

# Generate README.md
cat > "$OUTPUT_DIR/README.md" << EOF
# ${MODULE_NAME} Simulation

This is a generated transaction-level simulation of the \`${MODULE_NAME}\` module.

## Building

\`\`\`bash
mkdir build
cd build
cmake ..
make
\`\`\`

## Running

\`\`\`bash
./build/${MODULE_NAME}_sim [options]
\`\`\`

### Options

- \`--cycles <n>\`: Run simulation for n cycles (default: 100)
- \`--verbose\`: Enable verbose output
- \`--stats\`: Print performance statistics
- \`--help\`: Show help message

## Module Description

This simulation was generated from the MLIR input file using the Sharp framework's
\`--sharp-simulate\` pass in translation mode.

## Generated from Sharp

Generated on $(date)
Source: $INPUT_FILE
EOF

# Clean up
rm "$TEMP_OUTPUT"

echo "✅ Workspace generated successfully in $OUTPUT_DIR/"
echo ""
echo "📁 Generated files:"
echo "   - ${MODULE_NAME}_sim.cpp    (main simulation code)"
echo "   - SimulationBase.h/cpp      (simulation infrastructure)"
echo "   - CMakeLists.txt            (build configuration)"
echo "   - README.md                 (documentation)"
echo ""
echo "🔨 To build and run:"
echo "   cd $OUTPUT_DIR"
echo "   mkdir build && cd build"
echo "   cmake .. && make"
echo "   ./${MODULE_NAME}_sim --cycles 100 --verbose"