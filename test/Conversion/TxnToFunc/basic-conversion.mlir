// RUN: sharp-opt %s -convert-txn-to-func | FileCheck %s

// Test basic conversion from txn module to func module

// CHECK-LABEL: module
// CHECK: func.func @Counter_main() {
// CHECK:   return
// CHECK: }
// CHECK: func.func @Counter_getValue() -> i32 {
// CHECK:   %[[C0:.*]] = arith.constant 0 : i32
// CHECK:   txn.return %[[C0]] : i32
// CHECK: }
// CHECK: func.func @Counter_increment() {
// CHECK:   txn.yield
// CHECK: }

txn.module @Counter {
  txn.value_method @getValue() -> i32 {
    %c0 = arith.constant 0 : i32
    txn.return %c0 : i32
  }
  
  txn.action_method @increment() {
    txn.yield
  }
  
  txn.schedule [@getValue, @increment]
}