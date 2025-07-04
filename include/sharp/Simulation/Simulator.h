//===- Simulator.h - Sharp Transaction-Level Simulator --------------------===//
//
// Main simulator class for Sharp's transaction-level simulation.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_SIMULATION_SIMULATOR_H
#define SHARP_SIMULATION_SIMULATOR_H

#include "sharp/Simulation/Event.h"
#include "sharp/Simulation/SimModule.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <vector>

namespace sharp {
namespace sim {

using llvm::StringMap;

/// Configuration for simulator
struct SimConfig {
  /// Maximum number of cycles to simulate
  uint64_t maxCycles = 1000000;
  
  /// Enable debug output
  bool debug = false;
  
  /// Enable performance profiling
  bool profile = true;
  
  /// Random seed for non-deterministic operations
  uint64_t seed = 0;
};

/// Main transaction-level simulator
class Simulator {
public:
  Simulator(SimConfig config = {}) : config(config) {}
  
  /// Add a module to the simulation
  void addModule(StringRef name, std::unique_ptr<SimModule> module);
  
  /// Get a module by name
  SimModule* getModule(StringRef name);
  
  /// Schedule an event
  EventPtr schedule(SimTime time, StringRef module, StringRef method,
                   ArrayRef<Value> args = {}, Callback callback = nullptr);
  
  /// Schedule an event with dependencies
  EventPtr scheduleWithDeps(SimTime time, StringRef module, StringRef method,
                           ArrayRef<Value> args, ArrayRef<EventPtr> deps,
                           Callback callback = nullptr);
  
  /// Run simulation until completion or max cycles
  void run();
  
  /// Run for a specific number of cycles
  void runCycles(uint64_t cycles);
  
  /// Step one event
  bool step();
  
  /// Reset simulation
  void reset();
  
  /// Get current simulation time
  SimTime getCurrentTime() const { return eventQueue.getCurrentTime(); }
  
  /// Get performance statistics
  std::map<std::string, std::map<std::string, uint64_t>> getStatistics() const;
  
  /// Enable/disable debug output
  void setDebug(bool enable) { config.debug = enable; }
  
  /// Set breakpoint on method
  void setBreakpoint(StringRef module, StringRef method);
  
  /// Clear breakpoint
  void clearBreakpoint(StringRef module, StringRef method);

private:
  SimConfig config;
  EventQueue eventQueue;
  StringMap<std::unique_ptr<SimModule>> modules;
  std::vector<EventPtr> executingEvents;
  
  // Breakpoints: module -> set of methods
  StringMap<std::set<std::string>> breakpoints;
  
  /// Check if event conflicts with currently executing events
  bool hasConflicts(EventPtr event) const;
  
  /// Execute a single event
  void executeEvent(EventPtr event);
  
  /// Handle multi-cycle continuation
  void scheduleContinuation(EventPtr event, const ExecutionResult& result);
  
  /// Debug output
  void debugPrint(StringRef message);
  
  /// Check if we should break
  bool shouldBreak(StringRef module, StringRef method) const;
};

/// Builder for setting up simulations
class SimulationBuilder {
public:
  SimulationBuilder() : sim(std::make_unique<Simulator>()) {}
  
  /// Configure the simulator
  SimulationBuilder& withConfig(SimConfig config) {
    sim = std::make_unique<Simulator>(config);
    return *this;
  }
  
  /// Add a module
  template <typename ModuleType, typename... Args>
  SimulationBuilder& withModule(StringRef name, Args&&... args) {
    auto module = std::make_unique<ModuleType>(std::forward<Args>(args)...);
    sim->addModule(name, std::move(module));
    return *this;
  }
  
  /// Add initial events
  SimulationBuilder& withInitialEvent(SimTime time, StringRef module,
                                     StringRef method, ArrayRef<Value> args = {}) {
    initialEvents.push_back({time, module.str(), method.str(),
                           SmallVector<Value>(args.begin(), args.end())});
    return *this;
  }
  
  /// Build and return the simulator
  std::unique_ptr<Simulator> build() {
    // Schedule initial events
    for (auto& e : initialEvents) {
      sim->schedule(e.time, e.module, e.method, e.args);
    }
    return std::move(sim);
  }

private:
  std::unique_ptr<Simulator> sim;
  
  struct InitialEvent {
    SimTime time;
    std::string module;
    std::string method;
    SmallVector<Value> args;
  };
  std::vector<InitialEvent> initialEvents;
};

} // namespace sim
} // namespace sharp

#endif // SHARP_SIMULATION_SIMULATOR_H