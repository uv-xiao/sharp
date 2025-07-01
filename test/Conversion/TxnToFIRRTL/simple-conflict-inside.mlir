// RUN: sharp-opt --convert-txn-to-firrtl %s | FileCheck %s

// Simple test for conflict_inside calculation

// CHECK-LABEL: firrtl.circuit "SimpleConflictTest"
// CHECK: firrtl.module @SimpleConflictTest

txn.module @SimpleConflictTest {
  %reg = txn.instance @r of @Register : !txn.module<"Register">
  
  // Action method with guaranteed internal conflict
  txn.action_method @conflictingWrites() {
    %c5 = arith.constant 5 : i32
    %c7 = arith.constant 7 : i32
    
    // Two writes to same register - always conflicts
    txn.call @r::@write(%c5) : (i32) -> ()
    txn.call @r::@write(%c7) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@conflictingWrites] {
    conflict_matrix = {
      "r::write,r::write" = 2 : i32  // C - Conflict
    }
  }
}

// The generated FIRRTL should include conflict_inside detection
// CHECK-DAG: firrtl.and
// CHECK-DAG: firrtl.not
// The conflict detection creates AND for reachability and NOT for negation