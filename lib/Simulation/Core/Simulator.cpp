//===- Simulator.cpp - Sharp Transaction-Level Simulator Implementation ---===//
//
// Implementation of the main simulation engine for Sharp.
//
//===----------------------------------------------------------------------===//

#include "sharp/Simulation/Simulator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Twine.h"
#include <iostream>

#define DEBUG_TYPE "sharp-simulator"

namespace sharp {
namespace sim {

void Simulator::addModule(StringRef name, std::unique_ptr<SimModule> module) {
  if (modules.find(name) != modules.end()) {
    llvm::report_fatal_error(llvm::Twine("Module '") + name + "' already exists");
  }
  modules[name] = std::move(module);
}

SimModule* Simulator::getModule(StringRef name) {
  auto it = modules.find(name);
  return it != modules.end() ? it->second.get() : nullptr;
}

EventPtr Simulator::schedule(SimTime time, StringRef module, StringRef method,
                            ArrayRef<Value> args, Callback callback) {
  // Create event with absolute time
  SimTime absTime = eventQueue.getCurrentTime() + time;
  auto event = std::make_shared<Event>(absTime, module.str(), method.str(), args);
  
  if (callback) {
    event->setCallback(callback);
  }
  
  eventQueue.push(event);
  return event;
}

EventPtr Simulator::scheduleWithDeps(SimTime time, StringRef module, StringRef method,
                                    ArrayRef<Value> args, ArrayRef<EventPtr> deps,
                                    Callback callback) {
  // Create event with dependencies
  SimTime absTime = eventQueue.getCurrentTime() + time;
  auto event = std::make_shared<Event>(absTime, module.str(), method.str(), args);
  
  // Add dependencies
  for (auto dep : deps) {
    event->addDependency(dep);
  }
  
  if (callback) {
    event->setCallback(callback);
  }
  
  eventQueue.push(event);
  return event;
}

void Simulator::run() {
  uint64_t cycles = 0;
  while (!eventQueue.empty() && cycles < config.maxCycles) {
    if (!step()) {
      break;
    }
    cycles++;
  }
  
  if (cycles >= config.maxCycles) {
    debugPrint("Simulation stopped: maximum cycles reached");
  }
}

void Simulator::runCycles(uint64_t cycles) {
  for (uint64_t i = 0; i < cycles && !eventQueue.empty(); i++) {
    if (!step()) {
      break;
    }
  }
}

bool Simulator::step() {
  // Get next ready event
  EventPtr event = eventQueue.popReady();
  if (!event) {
    return false;
  }
  
  // Check for breakpoint
  if (shouldBreak(event->getModule(), event->getMethod())) {
    std::cout << "Breakpoint hit: " << event->getModule() << "::" << event->getMethod() << "\n";
    // In a real implementation, we'd pause here for debugging
  }
  
  // Check for conflicts with currently executing events
  if (hasConflicts(event)) {
    // Defer this event
    eventQueue.push(event);
    return true;
  }
  
  // Execute the event
  executeEvent(event);
  
  return true;
}

void Simulator::reset() {
  // Clear event queue
  eventQueue = EventQueue();
  
  // Clear executing events
  executingEvents.clear();
  
  // Reset all modules
  for (auto& kv : modules) {
    kv.second->reset();
  }
}

std::map<std::string, std::map<std::string, uint64_t>> Simulator::getStatistics() const {
  std::map<std::string, std::map<std::string, uint64_t>> stats;
  
  for (const auto& kv : modules) {
    stats[std::string(kv.first())] = kv.second->getMetrics();
  }
  
  // Add global statistics
  stats["_global"]["simulation_time"] = eventQueue.getCurrentTime();
  stats["_global"]["events_executed"] = executingEvents.size();
  
  return stats;
}

void Simulator::setBreakpoint(StringRef module, StringRef method) {
  breakpoints[module].insert(method.str());
}

void Simulator::clearBreakpoint(StringRef module, StringRef method) {
  auto it = breakpoints.find(module);
  if (it != breakpoints.end()) {
    it->second.erase(method.str());
    if (it->second.empty()) {
      breakpoints.erase(it);
    }
  }
}

bool Simulator::hasConflicts(EventPtr event) const {
  auto* module = const_cast<Simulator*>(this)->getModule(event->getModule());
  if (!module) {
    return false;
  }
  
  // Check against all currently executing events
  for (const auto& exec : executingEvents) {
    if (exec->getModule() == event->getModule()) {
      // Same module - check conflict relation
      auto rel = module->getConflict(exec->getMethod(), event->getMethod());
      if (rel != ConflictRelation::CF) {
        return true;
      }
    }
  }
  
  return false;
}

void Simulator::executeEvent(EventPtr event) {
  debugPrint("Executing event: " + event->getModule() + "::" + event->getMethod() +
             " at time " + std::to_string(event->getTime()));
  
  // Get the module
  auto* module = getModule(event->getModule());
  if (!module) {
    llvm::report_fatal_error(llvm::Twine("Module '") + event->getModule() + "' not found");
  }
  
  // Add to executing list
  executingEvents.push_back(event);
  
  // Execute the method
  ExecutionResult result = module->execute(event->getMethod(), event->getArgs());
  
  // Handle continuation
  if (result.isContinuation) {
    scheduleContinuation(event, result);
  }
  
  // Schedule triggered events
  for (auto& triggered : result.triggeredEvents) {
    eventQueue.push(triggered);
  }
  
  // Call callback if present
  if (event->getCallback()) {
    event->getCallback()(result.returns);
  }
  
  // Mark event as complete
  eventQueue.markComplete(event->getID());
  
  // Remove from executing list
  executingEvents.erase(
    std::remove(executingEvents.begin(), executingEvents.end(), event),
    executingEvents.end()
  );
}

void Simulator::scheduleContinuation(EventPtr event, const ExecutionResult& result) {
  // Create continuation event
  auto cont = std::make_shared<Event>(
    eventQueue.getCurrentTime() + result.nextCycle,
    event->getModule(),
    event->getMethod(),
    event->getArgs()
  );
  
  // Copy state
  cont->setContinuationState(result.continuationState);
  
  // Maintain callback
  if (event->getCallback()) {
    cont->setCallback(event->getCallback());
  }
  
  // Schedule it
  eventQueue.push(cont);
}

void Simulator::debugPrint(StringRef message) {
  if (config.debug) {
    llvm::outs() << "[SIM @ " << eventQueue.getCurrentTime() << "] " << message << "\n";
  }
  LLVM_DEBUG(llvm::dbgs() << message << "\n");
}

bool Simulator::shouldBreak(StringRef module, StringRef method) const {
  auto it = breakpoints.find(module);
  if (it != breakpoints.end()) {
    return it->second.count(method.str()) > 0;
  }
  return false;
}

} // namespace sim
} // namespace sharp