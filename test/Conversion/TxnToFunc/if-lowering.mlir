// RUN: sharp-opt %s --convert-txn-to-func | FileCheck %s
// XFAIL: *
// TODO: Fix handling of txn.return/txn.abort inside if branches
// The conversion currently creates invalid IR with scf.yield before func.return

// Test that txn.if converts to scf.if properly

// CHECK-LABEL: func.func @conditionalValue
txn.module @ConditionalTest {
  txn.value_method @conditionalValue(%cond: i1) -> i32 {
    // CHECK: scf.if %arg0 -> (i32)
    %result = txn.if %cond -> i32 {
      %c10 = arith.constant 10 : i32
      // CHECK: scf.yield %{{.*}} : i32
      txn.yield %c10 : i32
    } else {
      %c20 = arith.constant 20 : i32
      // CHECK: scf.yield %{{.*}} : i32
      txn.yield %c20 : i32
    }
    // CHECK: return
    txn.return %result : i32
  }
  
  // CHECK-LABEL: func.func @withAbort
  txn.action_method @withAbort(%guard: i1) {
    // CHECK: scf.if %arg0 {
    txn.if %guard {
      %val = arith.constant 42 : i32
      // CHECK: return
      txn.return
    } else {
      // CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
      // CHECK: return
      txn.abort
    }
    txn.return
  }
  
  txn.schedule [@withAbort]
}