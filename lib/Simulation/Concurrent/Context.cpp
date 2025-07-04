//===- Context.cpp - DAM-style Simulation Context Implementation ----------===//
//
// This file implements the Context class for DAM-based concurrent simulation.
//
//===----------------------------------------------------------------------===//

#include "sharp/Simulation/Concurrent/Context.h"
#include <chrono>

namespace sharp {
namespace sim {
namespace concurrent {

void Context::run(uint64_t maxCycles) {
  isRunning.store(true, std::memory_order_release);
  
  while (running() && getLocalTime() < maxCycles) {
    // Process local events up to current time
    while (!localEvents.empty() && 
           localEvents.top().getTime() <= getLocalTime()) {
      Event event = localEvents.top();
      localEvents.pop();
      
      // Execute the event
      auto result = module->execute(event.getMethod(), event.getArgs());
      
      // Handle multi-cycle operations
      if (result.isContinuation) {
        // Schedule continuation event
        Event continuation(result.nextCycle, event.getModule(), 
                          event.getMethod(), event.getArgs());
        continuation.setContinuationState(result.continuationState);
        localEvents.push(continuation);
      }
    }
    
    // In DAM methodology, modules process events rather than having 
    // explicit cycles. The local time advances based on event timestamps.
    
    // Find next event time
    if (!localEvents.empty()) {
      uint64_t nextTime = localEvents.top().getTime();
      if (nextTime > getLocalTime()) {
        // Jump to next event time (time acceleration in DAM)
        advanceTime(nextTime);
      }
    } else {
      // No more events, advance by one cycle
      advanceTime(getLocalTime() + 1);
    }
    
    // Check for synchronization needs
    // In DAM, contexts can run far ahead until synchronization is needed
    // This is handled by channels when communication occurs
  }
  
  isRunning.store(false, std::memory_order_release);
}

} // namespace concurrent
} // namespace sim
} // namespace sharp