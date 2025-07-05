// RUN: sharp-opt %s 2>&1 | FileCheck %s

// Test txn.state operations (placeholder for future implementation)

// This test documents the expected syntax for state operations when implemented
// Currently, just parse as far as we can

// CHECK: error: custom op 'txn.state' is unknown

txn.module @StatefulCounter {
  // Future syntax for state declarations
  %count = txn.state @count : i32 = 0
  
  txn.value_method @getValue() -> i32 {
    %val = txn.read %count : i32
    txn.return %val : i32
  }
  
  txn.action_method @increment() {
    %old = txn.read %count : i32
    %c1 = arith.constant 1 : i32
    %new = arith.addi %old, %c1 : i32
    txn.write %count, %new : i32
    txn.yield
  }
  
  txn.action_method @reset() {
    %zero = arith.constant 0 : i32
    txn.write %count, %zero : i32
    txn.yield
  }
  
  txn.schedule [@getValue, @increment, @reset] {
    conflict_matrix = {
      "increment,reset" = 2 : i32,     // Conflict between state writes
      "getValue,increment" = 3 : i32,  // ConflictFree - read doesn't conflict
      "getValue,reset" = 3 : i32       // ConflictFree - read doesn't conflict
    }
  }
}