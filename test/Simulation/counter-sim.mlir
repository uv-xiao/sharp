// RUN: sharp-opt %s -sharp-simulate=mode=translation | FileCheck %s

// Counter module demonstrating action methods and conflict handling
txn.module @Counter {
  // Value method to get current count (always returns 0 for now)
  txn.value_method @getValue() -> i32 {
    %zero = arith.constant 0 : i32
    txn.return %zero : i32
  }
  
  // Action methods (no-op for now since we don't have state)
  txn.action_method @increment() {
    txn.yield
  }
  
  txn.action_method @decrement() {
    txn.yield
  }
  
  // Conflict matrix - increment and decrement conflict
  txn.schedule [@getValue, @increment, @decrement] {
    conflict_matrix = {
      "increment,decrement" = 2 : i32,  // Conflict
      "getValue,increment" = 3 : i32,   // ConflictFree
      "getValue,decrement" = 3 : i32    // ConflictFree
    }
  }
}

// CHECK: // Generated Txn Module Simulation
// CHECK: class CounterModule : public SimModule {
// CHECK:   CounterModule() : SimModule("Counter") {
// CHECK:     // Register methods
// CHECK:     registerValueMethod("getValue",
// CHECK:     registerActionMethod("increment",
// CHECK:     registerActionMethod("decrement",
// CHECK:   }

// CHECK:   // Conflict matrix
// CHECK:   std::map<std::pair<std::string, std::string>, ConflictRelation> conflicts = {
// CHECK-DAG:     {{[{][{]"increment", "decrement"[}], ConflictRelation::Conflict[}],}}
// CHECK-DAG:     {{[{][{]"getValue", "increment"[}], ConflictRelation::ConflictFree[}],}}
// CHECK-DAG:     {{[{][{]"getValue", "decrement"[}], ConflictRelation::ConflictFree[}],}}
// CHECK:   };