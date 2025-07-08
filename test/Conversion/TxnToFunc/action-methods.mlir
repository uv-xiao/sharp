// RUN: sharp-opt %s -convert-txn-to-func | FileCheck %s

// Test action method conversion

// CHECK-LABEL: module
// CHECK: func.func @Actions_doNothing() -> i1 {
// CHECK:   %[[FALSE:.*]] = arith.constant false
// CHECK:   return %[[FALSE]] : i1
// CHECK: }

// CHECK: func.func @Actions_processValue(%arg0: i32) -> i1 {
// CHECK:   %[[C1:.*]] = arith.constant 1 : i32
// CHECK:   %[[ADD:.*]] = arith.addi %arg0, %[[C1]]
// CHECK:   %[[FALSE:.*]] = arith.constant false
// CHECK:   return %[[FALSE]] : i1
// CHECK: }

// CHECK: func.func @Actions_rule_callDoNothing() -> i1 {
// CHECK:   call @Actions_doNothing() : () -> i1
// CHECK:   %[[FALSE:.*]] = arith.constant false
// CHECK:   return %[[FALSE]] : i1
// CHECK: }
// CHECK: func.func @Actions_rule_callProcessValue() -> i1 {
// CHECK:   %[[C42:.*]] = arith.constant 42 : i32
// CHECK:   call @Actions_processValue(%[[C42]]) : (i32) -> i1
// CHECK:   %[[FALSE:.*]] = arith.constant false
// CHECK:   return %[[FALSE]] : i1
// CHECK: }
// CHECK: func.func @Actions_scheduler() {

txn.module @Actions {
  txn.action_method @doNothing() {
    txn.return
  }
  
  txn.action_method @processValue(%val: i32) {
    %c1 = arith.constant 1 : i32
    %newval = arith.addi %val, %c1 : i32
    txn.return
  }
  
  txn.rule @callDoNothing {
    txn.call @doNothing() : () -> ()
    txn.yield
  }
  
  txn.rule @callProcessValue {
    %c42 = arith.constant 42 : i32
    txn.call @processValue(%c42) : (i32) -> ()
    txn.yield
  }
  
  txn.schedule [@callDoNothing, @callProcessValue]
}