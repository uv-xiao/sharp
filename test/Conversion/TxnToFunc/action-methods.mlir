// RUN: sharp-opt %s -convert-txn-to-func | FileCheck %s

// Test action method conversion

// CHECK-LABEL: module
// CHECK: func.func @Actions_main() {
// CHECK:   return
// CHECK: }
// CHECK: func.func @Actions_doNothing() {
// CHECK:   txn.yield
// CHECK: }

// CHECK: func.func @Actions_processValue(%arg0: i32) {
// CHECK:   %[[C1:.*]] = arith.constant 1 : i32
// CHECK:   %[[ADD:.*]] = arith.addi %arg0, %[[C1]]
// CHECK:   txn.yield
// CHECK: }

txn.module @Actions {
  txn.action_method @doNothing() {
    txn.yield
  }
  
  txn.action_method @processValue(%val: i32) {
    %c1 = arith.constant 1 : i32
    %newval = arith.addi %val, %c1 : i32
    txn.yield
  }
  
  txn.schedule [@doNothing, @processValue]
}