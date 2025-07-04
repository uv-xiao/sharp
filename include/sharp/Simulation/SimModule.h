//===- SimModule.h - Sharp Simulation Module Interface --------------------===//
//
// Defines the interface for simulated modules in Sharp's transaction-level
// simulation framework.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_SIMULATION_SIMMODULE_H
#define SHARP_SIMULATION_SIMMODULE_H

#include "sharp/Simulation/Event.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <map>
#include <memory>

namespace sharp {
namespace sim {

using mlir::Value;
using llvm::StringMap;
using llvm::StringRef;

/// Conflict relation between methods/rules
enum class ConflictRelation {
  SB = 0,  // Sequenced Before
  SA = 1,  // Sequenced After
  C = 2,   // Conflict
  CF = 3   // Conflict-Free
};

/// Method implementation function type
using MethodImpl = std::function<ExecutionResult(ArrayRef<Value>)>;

/// Base class for transaction-level simulated modules
class SimModule {
public:
  SimModule(StringRef name) : name(name.str()) {}
  virtual ~SimModule() = default;
  
  /// Get module name
  const std::string& getName() const { return name; }
  
  /// Execute a method with given arguments
  virtual ExecutionResult execute(StringRef method, ArrayRef<Value> args);
  
  /// Register a method implementation
  void registerMethod(StringRef name, MethodImpl impl) {
    methods[name] = impl;
  }
  
  /// Check if module has a method
  bool hasMethod(StringRef name) const {
    return methods.find(name) != methods.end();
  }
  
  /// Get conflict relation between two methods
  ConflictRelation getConflict(StringRef method1, StringRef method2) const;
  
  /// Set conflict relation between two methods
  void setConflict(StringRef method1, StringRef method2, ConflictRelation rel);
  
  /// Check if two methods can execute concurrently
  bool canExecuteConcurrently(StringRef method1, StringRef method2) const {
    auto rel = getConflict(method1, method2);
    return rel == ConflictRelation::CF;
  }
  
  /// Reset module state (for simulation restart)
  virtual void reset() {}
  
  /// Execute one simulation cycle (for cycle-based simulation)
  virtual void executeCycle() {}
  
  /// Get current simulation time
  virtual uint64_t getCurrentTime() const { return 0; }
  
  /// Commit pending state updates
  virtual void commitStateUpdates() {}
  
  /// Register a rule
  void registerRule(StringRef name, std::function<bool()> canFire) {
    rules[name] = canFire;
  }
  
  /// Handle execution result (for continuations)
  virtual void handleExecutionResult(const ExecutionResult& result) {
    (void)result; // Default: ignore
  }
  
  /// Get performance metrics
  virtual std::map<std::string, uint64_t> getMetrics() const {
    return {{"cycles", cycleCount}, {"calls", callCount}};
  }

protected:
  /// Module state variables - subclasses should add their state here
  
  /// Helper to create a continuation event
  EventPtr createContinuation(SimTime delay, StringRef method,
                             ArrayRef<Value> args, void* state);
  
  /// Helper to trigger another method
  EventPtr triggerMethod(StringRef targetModule, StringRef method,
                        ArrayRef<Value> args, SimTime delay = 0);

private:
  std::string name;
  StringMap<MethodImpl> methods;
  StringMap<std::function<bool()>> rules;
  
  // Conflict matrix: pair of method names -> relation
  using ConflictKey = std::pair<std::string, std::string>;
  std::map<ConflictKey, ConflictRelation> conflicts;
  
  // Performance counters
  uint64_t cycleCount = 0;
  uint64_t callCount = 0;
};

/// Module with explicit state management
template <typename StateType>
class StatefulModule : public SimModule {
public:
  StatefulModule(StringRef name) : SimModule(name) {}
  
  /// Get current state
  const StateType& getState() const { return state; }
  
  /// Set state (for initialization)
  void setState(const StateType& s) { state = s; }
  
  /// Reset to initial state
  void reset() override {
    state = initialState;
  }
  
  /// Set initial state
  void setInitialState(const StateType& s) {
    initialState = s;
    state = s;
  }

protected:
  StateType state;
  StateType initialState;
};

/// Factory for creating simulation modules
class SimModuleFactory {
public:
  using Creator = std::function<std::unique_ptr<SimModule>()>;
  
  /// Register a module type
  template <typename ModuleType>
  void registerModule(StringRef typeName) {
    creators[typeName] = []() {
      return std::make_unique<ModuleType>();
    };
  }
  
  /// Create a module instance
  std::unique_ptr<SimModule> create(StringRef typeName, StringRef instanceName);
  
  /// Get singleton instance
  static SimModuleFactory& getInstance();

private:
  StringMap<Creator> creators;
};

} // namespace sim
} // namespace sharp

#endif // SHARP_SIMULATION_SIMMODULE_H