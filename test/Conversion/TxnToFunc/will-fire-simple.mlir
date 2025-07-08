// RUN: sharp-opt %s --convert-txn-to-func | FileCheck %s

// Test basic will-fire logic without conflicts
txn.module @SimpleTest {
  // Simple state variable (to be implemented later)
  
  txn.action_method @doWork() {
    // Just a placeholder action
    txn.return
  }
  
  txn.rule @rule1 {
    txn.call @doWork() : () -> ()
    txn.yield
  }
  
  txn.rule @rule2 {
    txn.call @doWork() : () -> ()
    txn.yield
  }
  
  txn.schedule [@rule1, @rule2]
}

// CHECK-LABEL: func.func @SimpleTest_doWork() -> i1
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: return %[[FALSE]] : i1

// CHECK-LABEL: func.func @SimpleTest_rule_rule1() -> i1
// CHECK: call @SimpleTest_doWork() : () -> i1
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: return %[[FALSE]] : i1

// CHECK-LABEL: func.func @SimpleTest_rule_rule2() -> i1
// CHECK: call @SimpleTest_doWork() : () -> i1
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: return %[[FALSE]] : i1

// CHECK-LABEL: func.func @SimpleTest_scheduler()
// CHECK: %[[FIRED1:.*]] = memref.alloc() : memref<i1>
// CHECK: %[[FIRED2:.*]] = memref.alloc() : memref<i1>

// Execute rule1
// CHECK: scf.if %{{.*}} {
// CHECK:   %[[ABORTED1:.*]] = func.call @SimpleTest_rule_rule1() : () -> i1
// CHECK:   %[[NOT_ABORTED1:.*]] = arith.xori %[[ABORTED1]], %{{.*}} : i1
// CHECK:   memref.store %[[NOT_ABORTED1]], %[[FIRED1]]
// CHECK: }

// Execute rule2
// CHECK: scf.if %{{.*}} {
// CHECK:   %[[ABORTED2:.*]] = func.call @SimpleTest_rule_rule2() : () -> i1
// CHECK:   %[[NOT_ABORTED2:.*]] = arith.xori %[[ABORTED2]], %{{.*}} : i1
// CHECK:   memref.store %[[NOT_ABORTED2]], %[[FIRED2]]
// CHECK: }