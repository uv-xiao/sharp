//===- ConcurrentSimulator.cpp - DAM-based Concurrent Simulator -----------===//
//
// This file implements the ConcurrentSimulator class following DAM methodology.
//
//===----------------------------------------------------------------------===//

#include "sharp/Simulation/Concurrent/ConcurrentSimulator.h"
#include <algorithm>
#include <sched.h>
#include <pthread.h>

namespace sharp {
namespace sim {
namespace concurrent {

ConcurrentSimulator::ConcurrentSimulator(const ConcurrentSimConfig& config)
  : config(config) {
  if (this->config.numThreads == 0) {
    // Auto-detect number of threads
    this->config.numThreads = std::thread::hardware_concurrency();
  }
}

ConcurrentSimulator::~ConcurrentSimulator() {
  stop();
}

void ConcurrentSimulator::addContext(std::unique_ptr<Context> context) {
  contexts.push_back(std::move(context));
}

void ConcurrentSimulator::run() {
  // Analyze dependencies to determine parallelization strategy
  analyzeDependencies();
  
  // Setup thread scheduling policy
  setupScheduling();
  
  // Create worker threads for each context
  // Following DAM, each context runs independently with its own thread
  for (auto& ctx : contexts) {
    threads.emplace_back(&ConcurrentSimulator::workerThread, this, ctx.get());
  }
  
  // Wait for all threads to complete
  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

void ConcurrentSimulator::stop() {
  // Signal all contexts to stop
  for (auto& ctx : contexts) {
    ctx->stop();
  }
  
  // Wait for threads to finish
  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  threads.clear();
}

ConcurrentSimulator::Stats ConcurrentSimulator::getStats() const {
  std::lock_guard<std::mutex> lock(statsMutex);
  
  Stats result = stats;
  
  // Calculate speedup based on parallel execution
  if (result.totalCycles > 0) {
    // Find max cycles among all contexts
    uint64_t maxContextCycles = 0;
    for (const auto& [name, cycles] : result.contextCycles) {
      maxContextCycles = std::max(maxContextCycles, cycles);
    }
    
    // Speedup = sequential execution time / parallel execution time
    // Sequential time would be sum of all context cycles
    uint64_t sequentialCycles = 0;
    for (const auto& [name, cycles] : result.contextCycles) {
      sequentialCycles += cycles;
    }
    
    if (maxContextCycles > 0) {
      result.speedup = static_cast<double>(sequentialCycles) / maxContextCycles;
    }
  }
  
  return result;
}

void ConcurrentSimulator::workerThread(Context* context) {
  // Set thread name for debugging
  std::string threadName = "sim_" + context->getName();
  pthread_setname_np(pthread_self(), threadName.c_str());
  
  // Run the context
  context->run(config.maxCycles);
  
  // Update statistics
  {
    std::lock_guard<std::mutex> lock(statsMutex);
    stats.contextCycles[context->getName()] = context->getLocalTime();
    stats.totalCycles = std::max(stats.totalCycles, context->getLocalTime());
    // TODO: Track events per context
  }
}

void ConcurrentSimulator::analyzeDependencies() {
  // Analyze connections between contexts to understand dependencies
  // This helps determine:
  // 1. Which contexts can run fully in parallel
  // 2. Which contexts need frequent synchronization
  // 3. Optimal channel depths for performance
  
  // In DAM, contexts with loose coupling (large channel depths or
  // infrequent communication) can run far ahead of each other
  
  // TODO: Implement dependency analysis based on connections
}

void ConcurrentSimulator::setupScheduling() {
  if (config.useSchedulerFIFO) {
    // Set SCHED_FIFO for better performance on oversaturated workloads
    // This reduces context switches significantly (30.3x in DAM paper)
    struct sched_param param;
    param.sched_priority = sched_get_priority_min(SCHED_FIFO);
    
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
      // Fall back to normal scheduling if FIFO not available
      // (requires elevated privileges)
    }
  }
}

} // namespace concurrent
} // namespace sim
} // namespace sharp