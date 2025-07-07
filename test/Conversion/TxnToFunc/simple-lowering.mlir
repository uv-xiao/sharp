// RUN: sharp-opt %s --convert-txn-to-func | FileCheck %s

// Test simple value method conversion

// CHECK-LABEL: func.func @SimpleTest_getValue() -> i32
// CHECK-NOT: txn.module
// CHECK-NOT: txn.value_method
txn.module @SimpleTest {
  txn.value_method @getValue() -> i32 {
    %c42 = arith.constant 42 : i32
    // CHECK: return %{{.*}} : i32
    // CHECK-NOT: txn.return
    txn.return %c42 : i32
  }
  
  txn.schedule [@getValue]
}