// RUN: sharp-opt %s -convert-txn-to-func | FileCheck %s

// Test rule conversion

// CHECK-LABEL: module
// CHECK: func.func @RuleModule_rule_alwaysFire() -> i1 {
// CHECK:   %[[FALSE:.*]] = arith.constant false
// CHECK:   return %[[FALSE]] : i1
// CHECK: }

// CHECK: func.func @RuleModule_rule_conditionalFire() -> i1 {
// CHECK:   %[[C10:.*]] = arith.constant 10 : i32
// CHECK:   %[[C5:.*]] = arith.constant 5 : i32
// CHECK:   %[[CMP:.*]] = arith.cmpi sgt, %[[C10]], %[[C5]]
// CHECK:   %[[FALSE:.*]] = arith.constant false
// CHECK:   return %[[FALSE]] : i1
// CHECK: }

// CHECK: func.func @RuleModule_scheduler() {

txn.module @RuleModule {
  txn.rule @alwaysFire {
    // Rules now just execute and return abort status
    txn.yield
  }
  
  txn.rule @conditionalFire {
    %c10 = arith.constant 10 : i32
    %c5 = arith.constant 5 : i32
    %cond = arith.cmpi sgt, %c10, %c5 : i32
    // Do something based on condition
    txn.yield
  }
  
  txn.schedule [@alwaysFire, @conditionalFire]
}