//===- Event.h - Sharp Simulation Event Types -----------------------------===//
//
// Defines the event types and structures for Sharp's event-driven simulation.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_SIMULATION_EVENT_H
#define SHARP_SIMULATION_EVENT_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sharp {
namespace sim {

using mlir::Value;
using llvm::ArrayRef;
using llvm::SmallVector;

// Forward declarations
class SimModule;
class Event;

using EventID = uint64_t;
using ModuleID = std::string;
using MethodID = std::string;
using SimTime = uint64_t;
using EventPtr = std::shared_ptr<Event>;
using Callback = std::function<void(ArrayRef<Value>)>;

/// Result of executing an event
struct ExecutionResult {
  /// Whether this is a multi-cycle operation
  bool isContinuation = false;
  
  /// Next cycle to resume execution
  SimTime nextCycle = 0;
  
  /// State for continuation
  void* continuationState = nullptr;
  
  /// Return values
  SmallVector<Value, 4> returns;
  
  /// Events to trigger
  SmallVector<EventPtr, 4> triggeredEvents;
  
  /// Whether the execution was aborted
  bool aborted = false;
  
  /// Source event that generated this result (for commit phase)
  EventPtr sourceEvent;
};

/// Represents a simulation event
class Event {
public:
  Event(SimTime time, ModuleID module, MethodID method,
        ArrayRef<Value> args = {})
      : id(nextID++), time(time), module(module), method(method),
        args(args.begin(), args.end()) {}

  /// Unique event identifier
  EventID getID() const { return id; }
  
  /// Simulation time when event should execute
  SimTime getTime() const { return time; }
  
  /// Target module
  const ModuleID& getModule() const { return module; }
  
  /// Method/rule to execute
  const MethodID& getMethod() const { return method; }
  
  /// Arguments for the method
  ArrayRef<Value> getArgs() const { return args; }
  
  /// Add a dependency - this event cannot execute until dep completes
  void addDependency(EventPtr dep) {
    dependencies.push_back(dep);
  }
  
  /// Get all dependencies
  ArrayRef<EventPtr> getDependencies() const { return dependencies; }
  
  /// Check if all dependencies are satisfied
  bool isReady() const;
  
  /// Mark a dependency as completed
  void markDependencyComplete(EventID depID);
  
  /// Set callback for when event completes
  void setCallback(Callback cb) { callback = cb; }
  
  /// Get callback
  const Callback& getCallback() const { return callback; }
  
  /// For continuation events
  void setContinuationState(void* state) { continuationState = state; }
  void* getContinuationState() const { return continuationState; }
  
  /// Comparison for priority queue (earlier time = higher priority)
  bool operator<(const Event& other) const {
    return time > other.time; // Reverse for min-heap
  }

private:
  EventID id;
  SimTime time;
  ModuleID module;
  MethodID method;
  SmallVector<Value, 4> args;
  SmallVector<EventPtr, 4> dependencies;
  SmallVector<EventID, 4> completedDeps;
  Callback callback;
  void* continuationState = nullptr;
  
  static EventID nextID;
};

/// Event queue with dependency tracking
class EventQueue {
public:
  /// Add an event to the queue
  void push(EventPtr event);
  
  /// Get the next ready event (nullptr if none ready)
  EventPtr popReady();
  
  /// Check if queue is empty
  bool empty() const { return events.empty(); }
  
  /// Get current simulation time
  SimTime getCurrentTime() const { return currentTime; }
  
  /// Mark an event as completed
  void markComplete(EventID id);
  
  /// Peek at the next ready event without removing it
  EventPtr peekReady() const;

private:
  std::vector<EventPtr> events; // Min-heap by time
  std::vector<EventPtr> deferred; // Events waiting on dependencies
  SimTime currentTime = 0;
  
  /// Move ready events from deferred to main queue
  void checkDeferred();
};

} // namespace sim
} // namespace sharp

#endif // SHARP_SIMULATION_EVENT_H