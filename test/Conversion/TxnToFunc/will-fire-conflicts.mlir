// RUN: sharp-opt %s --convert-txn-to-func | FileCheck %s

// Test will-fire logic with conflicts
txn.module @ConflictTest {
  %reg = txn.instance @reg of @Register<i32> : index
  
  txn.action_method @write1() {
    %c1 = arith.constant 1 : i32
    txn.call @reg::@write(%c1) : (i32) -> ()
    txn.return
  }
  
  txn.action_method @write2() {
    %c2 = arith.constant 2 : i32
    txn.call @reg::@write(%c2) : (i32) -> ()
    txn.return
  }
  
  txn.rule @rule1 {
    txn.call @write1() : () -> ()
    txn.yield
  }
  
  txn.rule @rule2 {
    txn.call @write2() : () -> ()
    txn.yield
  }
  
  // Both rules conflict because they call write methods
  txn.schedule [@rule1, @rule2] {
    conflict_matrix = {
      "rule1,rule2" = 2 : i32,  // Conflict
      "rule2,rule1" = 2 : i32   // Conflict
    }
  }
}

// CHECK-LABEL: func.func @ConflictTest_scheduler()
// Check that conflict checking is generated
// CHECK: %[[RULE1_FIRED:.*]] = memref.alloc() : memref<i1>
// CHECK: %[[RULE2_FIRED:.*]] = memref.alloc() : memref<i1>

// Rule 1 executes
// CHECK: scf.if %{{.*}} {
// CHECK:   %[[ABORTED1:.*]] = func.call @ConflictTest_rule_rule1() : () -> i1
// CHECK:   %[[NOT_ABORTED1:.*]] = arith.xori %[[ABORTED1]], %{{.*}} : i1
// CHECK:   memref.store %[[NOT_ABORTED1]], %[[RULE1_FIRED]]
// CHECK: }

// Rule 2 checks conflicts before executing
// CHECK: %[[R1_FIRED:.*]] = memref.load %[[RULE1_FIRED]]
// CHECK: %[[NO_CONFLICT:.*]] = arith.xori %[[R1_FIRED]], %{{.*}} : i1
// CHECK: %{{.*}} = arith.andi %{{.*}}, %[[NO_CONFLICT]] : i1
// CHECK: %[[WILL_FIRE2:.*]] = arith.andi %{{.*}}, %{{.*}} : i1
// CHECK: scf.if %[[WILL_FIRE2]] {
// CHECK:   func.call @ConflictTest_rule_rule2() : () -> i1
// CHECK: }