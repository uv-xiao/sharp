// RUN: sharp-opt %s -allow-unregistered-dialect | sharp-opt -allow-unregistered-dialect | FileCheck %s

// Test control flow operations with standard terminators

// CHECK-LABEL: func.func @test_if_with_yield
func.func @test_if_with_yield(%cond: i1) -> i32 {
  // CHECK: %{{.*}} = txn.if %{{.*}} -> i32 {
  %result = txn.if %cond -> i32 {
    %c10 = arith.constant 10 : i32
    // CHECK: txn.yield %{{.*}} : i32
    txn.yield %c10 : i32
  } else {
    %c20 = arith.constant 20 : i32
    // CHECK: txn.yield %{{.*}} : i32
    txn.yield %c20 : i32
  }
  return %result : i32
}

// CHECK-LABEL: func.func @test_empty_if_regions
func.func @test_empty_if_regions(%cond: i1) {
  // Both regions empty - still valid
  // CHECK: txn.if %{{.*}} {
  // CHECK-NEXT: } else {
  // CHECK-NEXT: }
  txn.if %cond {
  } else {
  }
  
  // If with side effects (no results)
  // CHECK: txn.if %{{.*}} {
  // CHECK: "test.action"
  // CHECK-NEXT: } else {
  // CHECK: "test.other_action"
  // CHECK-NEXT: }
  txn.if %cond {
    "test.action"() : () -> ()
  } else {
    "test.other_action"() : () -> ()
  }
  
  return
}

// CHECK-LABEL: func.func @test_abort_patterns
func.func @test_abort_patterns(%ready: i1) {
  // Guard pattern with abort
  %true = arith.constant true
  %not_ready = arith.xori %ready, %true : i1
  
  // CHECK: txn.if %{{.*}} {
  // CHECK: txn.abort
  // CHECK: } else {
  txn.if %not_ready {
    txn.abort
  } else {
    // Continue
  }
  
  // Action pattern
  // CHECK: txn.if %{{.*}} {
  // CHECK: "test.action"
  // CHECK: } else {
  // CHECK-NEXT: }
  txn.if %ready {
    "test.action"() : () -> ()
  } else {
    // Do nothing
  }
  
  return
}

// CHECK-LABEL: txn.module @ControlFlowInMethods
txn.module @ControlFlowInMethods {
  txn.value_method @conditionalValue(%cond: i1) -> i32 {
    // CHECK: %{{.*}} = txn.if %{{.*}} -> i32 {
    %result = txn.if %cond -> i32 {
      %c1 = arith.constant 1 : i32
      // CHECK: txn.yield
      txn.yield %c1 : i32
    } else {
      %c2 = arith.constant 2 : i32
      // CHECK: txn.yield
      txn.yield %c2 : i32
    }
    // CHECK: txn.return
    txn.return %result : i32
  }
  
  txn.action_method @guardedAction(%guard: i1) {
    // CHECK: txn.if %{{.*}} {
    txn.if %guard {
      // CHECK: txn.call @doWork
      txn.call @doWork() : () -> ()
      txn.return
    } else {
      // CHECK: txn.abort
      txn.abort
    }
    // CHECK: txn.return
    txn.return
  }
  
  txn.action_method @doWork() {
    txn.return
  }
  
  // CHECK: txn.schedule [@conditionalValue, @guardedAction, @doWork]
  txn.schedule [@conditionalValue, @guardedAction, @doWork]
}