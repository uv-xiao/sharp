//===- basic.cpp - Basic simulation tests ---------------------------------===//
//
// Tests for Sharp's simulation framework.
//
//===----------------------------------------------------------------------===//

#include "sharp/Simulation/Simulator.h"
#include "sharp/Simulation/SimModule.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"
#include <memory>

using namespace sharp::sim;
using namespace mlir;

namespace {

// Simple counter module for testing
class CounterModule : public SimModule {
public:
  CounterModule() : SimModule("Counter"), value(0) {
    registerMethod("getValue", [this](ArrayRef<Value>) {
      ExecutionResult result;
      // In real impl, would create MLIR Value
      return result;
    });
    
    registerMethod("increment", [this](ArrayRef<Value>) {
      value++;
      return ExecutionResult();
    });
    
    registerMethod("decrement", [this](ArrayRef<Value>) {
      value--;
      return ExecutionResult();
    });
    
    registerMethod("setValue", [this](ArrayRef<Value> args) {
      // In real impl, would extract value from args
      value = 42; // Placeholder
      return ExecutionResult();
    });
  }
  
  void reset() override {
    value = 0;
  }
  
  int getValue() const { return value; }

private:
  int value;
};

// Module with multi-cycle operations
class MultiCycleModule : public SimModule {
public:
  MultiCycleModule() : SimModule("MultiCycle") {
    registerMethod("longOp", [this](ArrayRef<Value>) {
      ExecutionResult result;
      
      if (!state) {
        // First call - initiate multi-cycle operation
        state = new int(1);
        result.isContinuation = true;
        result.nextCycle = 3; // Continue after 3 cycles
        result.continuationState = state;
      } else {
        // Continuation
        delete state;
        state = nullptr;
        result.isContinuation = false;
      }
      
      return result;
    });
  }
  
  ~MultiCycleModule() {
    delete state;
  }

private:
  int* state = nullptr;
};

TEST(SimulatorTest, BasicExecution) {
  MLIRContext context;
  
  auto sim = std::make_unique<Simulator>();
  auto counter = std::make_unique<CounterModule>();
  CounterModule* counterPtr = counter.get();
  
  sim->addModule("counter", std::move(counter));
  
  // Schedule some events
  sim->schedule(0, "counter", "increment");
  sim->schedule(1, "counter", "increment");
  sim->schedule(2, "counter", "increment");
  
  // Run simulation
  sim->runCycles(5);
  
  EXPECT_EQ(counterPtr->getValue(), 3);
  EXPECT_EQ(sim->getCurrentTime(), 2);
}

TEST(SimulatorTest, ConflictHandling) {
  MLIRContext context;
  
  auto sim = std::make_unique<Simulator>();
  auto counter = std::make_unique<CounterModule>();
  
  // Set up conflicts
  counter->setConflict("increment", "decrement", ConflictRelation::C);
  counter->setConflict("getValue", "setValue", ConflictRelation::SB);
  
  sim->addModule("counter", std::move(counter));
  
  // These should conflict
  sim->schedule(0, "counter", "increment");
  sim->schedule(0, "counter", "decrement");
  
  sim->runCycles(2);
  
  // Both should have executed (sequentially)
  auto stats = sim->getStatistics();
  EXPECT_GE(stats["counter"]["calls"], 2);
}

TEST(SimulatorTest, MultiCycleOperation) {
  MLIRContext context;
  
  auto sim = std::make_unique<Simulator>();
  sim->addModule("mc", std::make_unique<MultiCycleModule>());
  
  // Schedule multi-cycle operation
  bool completed = false;
  sim->schedule(0, "mc", "longOp", {}, [&completed](ArrayRef<Value>) {
    completed = true;
  });
  
  // Should not complete immediately
  sim->runCycles(2);
  EXPECT_FALSE(completed);
  
  // Should complete after continuation
  sim->runCycles(5);
  EXPECT_TRUE(completed);
}

TEST(SimulatorTest, Dependencies) {
  MLIRContext context;
  
  auto sim = std::make_unique<Simulator>();
  auto counter = std::make_unique<CounterModule>();
  CounterModule* counterPtr = counter.get();
  
  sim->addModule("counter", std::move(counter));
  
  // Create dependent events
  auto e1 = sim->schedule(0, "counter", "increment");
  auto e2 = sim->schedule(0, "counter", "increment");
  auto e3 = sim->scheduleWithDeps(0, "counter", "increment", {}, {e1, e2});
  
  sim->run();
  
  // All three increments should have executed
  EXPECT_EQ(counterPtr->getValue(), 3);
}

TEST(EventQueueTest, PriorityOrdering) {
  EventQueue queue;
  
  // Add events out of order
  queue.push(std::make_shared<Event>(5, "m1", "method1"));
  queue.push(std::make_shared<Event>(2, "m2", "method2"));
  queue.push(std::make_shared<Event>(8, "m3", "method3"));
  queue.push(std::make_shared<Event>(2, "m4", "method4"));
  
  // Should get events in time order
  auto e1 = queue.popReady();
  ASSERT_NE(e1, nullptr);
  EXPECT_EQ(e1->getTime(), 2);
  
  auto e2 = queue.popReady();
  ASSERT_NE(e2, nullptr);
  EXPECT_EQ(e2->getTime(), 2);
}

TEST(SimModuleTest, ConflictMatrix) {
  CounterModule module;
  
  // Set up conflict relations
  module.setConflict("m1", "m2", ConflictRelation::SB);
  module.setConflict("m2", "m3", ConflictRelation::CF);
  module.setConflict("m1", "m3", ConflictRelation::C);
  
  // Test queries
  EXPECT_EQ(module.getConflict("m1", "m2"), ConflictRelation::SB);
  EXPECT_EQ(module.getConflict("m2", "m1"), ConflictRelation::SA); // Inverted
  EXPECT_EQ(module.getConflict("m2", "m3"), ConflictRelation::CF);
  EXPECT_EQ(module.getConflict("m3", "m2"), ConflictRelation::CF); // Symmetric
  
  EXPECT_TRUE(module.canExecuteConcurrently("m2", "m3"));
  EXPECT_FALSE(module.canExecuteConcurrently("m1", "m3"));
}

} // namespace