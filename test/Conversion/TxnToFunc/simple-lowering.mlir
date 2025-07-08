// RUN: sharp-opt %s --convert-txn-to-func | FileCheck %s

// Test simple value method conversion

// CHECK-LABEL: module
// CHECK: func.func @SimpleTest_getValue() -> i32
// CHECK-NOT: txn.module
// CHECK-NOT: txn.value_method
// CHECK: func.func @SimpleTest_rule_doWork() -> i1
// CHECK: call @SimpleTest_getValue() : () -> i32
// CHECK: func.func @SimpleTest_scheduler()

txn.module @SimpleTest {
  txn.value_method @getValue() -> i32 {
    %c42 = arith.constant 42 : i32
    txn.return %c42 : i32
  }
  
  txn.rule @doWork {
    %val = txn.call @getValue() : () -> i32
    // Use the value somehow
    txn.yield
  }
  
  txn.schedule [@doWork]
}