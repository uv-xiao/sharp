//===- Context.h - DAM-style Simulation Context ---------------------------===//
//
// This file defines the Context class for DAM-based concurrent simulation.
// Each context represents an independent execution unit with its own local time.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_SIMULATION_CONCURRENT_CONTEXT_H
#define SHARP_SIMULATION_CONCURRENT_CONTEXT_H

#include "sharp/Simulation/SimModule.h"
#include "sharp/Simulation/Event.h"
#include <atomic>
#include <thread>
#include <memory>
#include <queue>

namespace sharp {
namespace sim {
namespace concurrent {

/// Represents a simulation context with local time following DAM methodology
class Context {
public:
  Context(std::string name, std::unique_ptr<SimModule> module)
    : name(std::move(name)), module(std::move(module)), 
      localTime(0), isRunning(false) {}

  /// Get the context name
  const std::string& getName() const { return name; }

  /// Get current local simulated time
  uint64_t getLocalTime() const { 
    return localTime.load(std::memory_order_acquire); 
  }

  /// Advance local time
  void advanceTime(uint64_t newTime) {
    uint64_t expected = localTime.load(std::memory_order_relaxed);
    while (newTime > expected && 
           !localTime.compare_exchange_weak(expected, newTime,
                                           std::memory_order_release,
                                           std::memory_order_relaxed)) {
      // Retry on failure
    }
  }

  /// Execute the context until blocked or max cycles reached
  void run(uint64_t maxCycles);

  /// Stop the context execution
  void stop() { isRunning.store(false, std::memory_order_release); }

  /// Check if context is running
  bool running() const { return isRunning.load(std::memory_order_acquire); }

  /// Get the underlying module
  SimModule* getModule() { return module.get(); }

private:
  std::string name;
  std::unique_ptr<SimModule> module;
  std::atomic<uint64_t> localTime;
  std::atomic<bool> isRunning;
  
  /// Local event queue for this context
  std::priority_queue<Event> localEvents;
};

} // namespace concurrent
} // namespace sim
} // namespace sharp

#endif // SHARP_SIMULATION_CONCURRENT_CONTEXT_H