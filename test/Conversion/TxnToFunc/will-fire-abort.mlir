// RUN: sharp-opt %s --convert-txn-to-func | FileCheck %s

// Test will-fire logic with abort handling
txn.module @WillFireTest {
  txn.rule @rule1 {
    // This rule always aborts
    txn.abort
  }
  
  txn.rule @rule2 {
    // This rule should execute since rule1 aborts
    txn.yield
  }
  
  txn.schedule [@rule1, @rule2]
}

// CHECK: func.func @WillFireTest_rule_rule1() -> i1 {
// CHECK:   %[[TRUE:.*]] = arith.constant true
// CHECK:   return %[[TRUE]] : i1
// CHECK: }

// CHECK: func.func @WillFireTest_rule_rule2() -> i1 {
// CHECK:   %[[FALSE:.*]] = arith.constant false
// CHECK:   return %[[FALSE]] : i1
// CHECK: }

// CHECK-LABEL: func.func @WillFireTest_scheduler()
// CHECK-DAG: %[[ALLOC1:.*]] = memref.alloc() : memref<i1>
// CHECK-DAG: %[[ALLOC2:.*]] = memref.alloc() : memref<i1>

// Execute rule1 - check that it returns true (aborted)
// CHECK: scf.if %{{.*}} {
// CHECK-NEXT:   %[[ABORTED:.*]] = func.call @WillFireTest_rule_rule1() : () -> i1
// CHECK-NEXT:   %[[NOT_ABORTED:.*]] = arith.xori %[[ABORTED]], %{{.*}} : i1
// CHECK-NEXT:   memref.store %[[NOT_ABORTED]], %[[ALLOC1]]
// CHECK-NEXT: }

// Execute rule2 - check that it returns false (success)
// CHECK: scf.if %{{.*}} {
// CHECK-NEXT:   %[[SUCCESS:.*]] = func.call @WillFireTest_rule_rule2() : () -> i1
// CHECK-NEXT:   %[[NOT_ABORTED2:.*]] = arith.xori %[[SUCCESS]], %{{.*}} : i1
// CHECK-NEXT:   memref.store %[[NOT_ABORTED2]], %[[ALLOC2]]
// CHECK-NEXT: }