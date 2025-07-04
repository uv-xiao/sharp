//===- ConcurrentSimulator.h - DAM-based Concurrent Simulator -------------===//
//
// This file defines the ConcurrentSimulator class that implements DAM-based
// parallel simulation with asynchronous distributed time.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_SIMULATION_CONCURRENT_CONCURRENTSIMULATOR_H
#define SHARP_SIMULATION_CONCURRENT_CONCURRENTSIMULATOR_H

#include "sharp/Simulation/Concurrent/Context.h"
#include "sharp/Simulation/Concurrent/Channel.h"
#include "sharp/Simulation/SimConfig.h"
#include <thread>
#include <vector>
#include <unordered_map>
#include <memory>

namespace sharp {
namespace sim {
namespace concurrent {

/// Configuration for concurrent simulation
struct ConcurrentSimConfig : public SimConfig {
  unsigned numThreads = 0;  // 0 = auto-detect
  enum Granularity {
    Fine,      // Each method/rule as separate context
    Coarse,    // Each module as separate context  
    Adaptive   // Dynamically adjust based on workload
  } granularity = Adaptive;
  
  bool useSchedulerFIFO = false; // Use SCHED_FIFO for oversaturated workloads
};

/// DAM-based concurrent simulator
class ConcurrentSimulator {
public:
  explicit ConcurrentSimulator(const ConcurrentSimConfig& config);
  ~ConcurrentSimulator();

  /// Add a context to the simulation
  void addContext(std::unique_ptr<Context> context);

  /// Create a channel between contexts
  template <typename T>
  std::shared_ptr<Channel<T>> createChannel(const std::string& name,
                                           size_t capacity = 0) {
    auto channel = std::make_shared<Channel<T>>(capacity);
    // Store channel reference for management
    channels[name] = channel;
    return channel;
  }

  /// Connect two contexts via a channel
  template <typename T>
  void connect(Context* sender, Context* receiver,
               std::shared_ptr<Channel<T>> channel) {
    // Record connection for dependency analysis
    connections.push_back({sender, receiver});
  }

  /// Run the concurrent simulation
  void run();

  /// Stop all contexts
  void stop();

  /// Get simulation statistics
  struct Stats {
    uint64_t totalEvents;
    uint64_t totalCycles;
    double speedup;
    std::unordered_map<std::string, uint64_t> contextCycles;
    std::unordered_map<std::string, uint64_t> contextEvents;
  };
  
  Stats getStats() const;

private:
  ConcurrentSimConfig config;
  std::vector<std::unique_ptr<Context>> contexts;
  std::vector<std::thread> threads;
  
  // Channel management
  std::unordered_map<std::string, std::shared_ptr<void>> channels;
  
  // Connection tracking for dependency analysis
  struct Connection {
    Context* sender;
    Context* receiver;
  };
  std::vector<Connection> connections;
  
  // Statistics
  mutable std::mutex statsMutex;
  Stats stats;
  
  /// Worker thread function
  void workerThread(Context* context);
  
  /// Analyze dependencies and determine parallelization strategy
  void analyzeDependencies();
  
  /// Setup thread scheduling policy
  void setupScheduling();
};

} // namespace concurrent
} // namespace sim
} // namespace sharp

#endif // SHARP_SIMULATION_CONCURRENT_CONCURRENTSIMULATOR_H