// RUN: sharp-opt %s -convert-txn-to-func | FileCheck %s

// Test basic conversion from txn module to func module

// CHECK-LABEL: module
// CHECK: func.func @Counter_getValue() -> i32 {
// CHECK:   %[[C0:.*]] = arith.constant 0 : i32
// CHECK:   return %[[C0]] : i32
// CHECK: }
// CHECK: func.func @Counter_increment() -> i1 {
// CHECK:   %[[FALSE:.*]] = arith.constant false
// CHECK:   return %[[FALSE]] : i1
// CHECK: }
// CHECK: func.func @Counter_rule_doIncrement() -> i1 {
// CHECK:   call @Counter_increment() : () -> i1
// CHECK:   %[[FALSE:.*]] = arith.constant false
// CHECK:   return %[[FALSE]] : i1
// CHECK: }
// CHECK: func.func @Counter_scheduler() {
// CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<i1>
// CHECK:   return
// CHECK: }

txn.module @Counter {
  txn.value_method @getValue() -> i32 {
    %c0 = arith.constant 0 : i32
    txn.return %c0 : i32
  }
  
  txn.action_method @increment() {
    txn.return
  }
  
  txn.rule @doIncrement {
    txn.call @increment() : () -> ()
    txn.yield
  }
  
  txn.schedule [@doIncrement]
}