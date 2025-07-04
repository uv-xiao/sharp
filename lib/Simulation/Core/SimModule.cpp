//===- SimModule.cpp - Sharp Simulation Module Implementation -------------===//
//
// Implementation of the base simulation module class and related utilities.
//
//===----------------------------------------------------------------------===//

#include "sharp/Simulation/SimModule.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/Twine.h"

#define DEBUG_TYPE "sharp-simulation"

namespace sharp {
namespace sim {

ExecutionResult SimModule::execute(StringRef method, ArrayRef<Value> args) {
  // Find the method implementation
  auto it = methods.find(method);
  if (it == methods.end()) {
    llvm::report_fatal_error(llvm::Twine("Method '") + method + "' not found in module '" + name + "'");
  }
  
  // Update performance counters
  callCount++;
  
  // Execute the method
  LLVM_DEBUG(llvm::dbgs() << "Executing " << name << "::" << method << "\n");
  ExecutionResult result = it->second(args);
  
  // Update cycle count if this was a multi-cycle operation
  if (result.isContinuation && result.nextCycle > 0) {
    cycleCount += result.nextCycle;
  } else {
    cycleCount++;
  }
  
  return result;
}

ConflictRelation SimModule::getConflict(StringRef method1, StringRef method2) const {
  // Create ordered key for consistent lookup
  ConflictKey key = method1 < method2 
    ? std::make_pair(method1.str(), method2.str())
    : std::make_pair(method2.str(), method1.str());
  
  auto it = conflicts.find(key);
  if (it != conflicts.end()) {
    // If we swapped the order, need to invert SB/SA
    if (method1 > method2) {
      switch (it->second) {
        case ConflictRelation::SB: return ConflictRelation::SA;
        case ConflictRelation::SA: return ConflictRelation::SB;
        default: return it->second;
      }
    }
    return it->second;
  }
  
  // Default to conflict if not specified
  return ConflictRelation::C;
}

void SimModule::setConflict(StringRef method1, StringRef method2, ConflictRelation rel) {
  // Store with ordered key
  ConflictKey key = method1 < method2
    ? std::make_pair(method1.str(), method2.str())
    : std::make_pair(method2.str(), method1.str());
  
  // Adjust relation if we swapped order
  if (method1 > method2) {
    switch (rel) {
      case ConflictRelation::SB: rel = ConflictRelation::SA; break;
      case ConflictRelation::SA: rel = ConflictRelation::SB; break;
      default: break;
    }
  }
  
  conflicts[key] = rel;
}

EventPtr SimModule::createContinuation(SimTime delay, StringRef method,
                                      ArrayRef<Value> args, void* state) {
  // Create a continuation event
  auto event = std::make_shared<Event>(delay, name, method.str(), args);
  event->setContinuationState(state);
  return event;
}

EventPtr SimModule::triggerMethod(StringRef targetModule, StringRef method,
                                 ArrayRef<Value> args, SimTime delay) {
  // Create an event to trigger another module's method
  auto event = std::make_shared<Event>(delay, targetModule.str(), method.str(), args);
  return event;
}

//===----------------------------------------------------------------------===//
// SimModuleFactory
//===----------------------------------------------------------------------===//

std::unique_ptr<SimModule> SimModuleFactory::create(StringRef typeName,
                                                   StringRef instanceName) {
  auto it = creators.find(typeName);
  if (it == creators.end()) {
    llvm::report_fatal_error(llvm::Twine("Module type '") + typeName + "' not registered");
  }
  
  auto module = it->second();
  // The module constructor sets the name, but we might want to override
  // with the instance name
  return module;
}

SimModuleFactory& SimModuleFactory::getInstance() {
  static SimModuleFactory instance;
  return instance;
}

} // namespace sim
} // namespace sharp