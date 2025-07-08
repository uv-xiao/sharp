// RUN: sharp-opt %s -convert-txn-to-firrtl | FileCheck %s

// Test that rule guards are properly evaluated

txn.module @GuardEvaluation {
  // State for testing
  %counter = txn.instance @counter of @Register<i32> : !txn.module<"Register">
  
  // Value method that returns current count
  txn.value_method @getCount() -> i32 {
    %val = txn.call @counter::@read() : () -> i32
    txn.return %val : i32
  }
  
  // Rule with guard that checks if count > 5
  txn.rule @incrementWhenReady {
    %count = txn.call @getCount() : () -> i32
    %c5 = arith.constant 5 : i32
    %ready = arith.cmpi sgt, %count, %c5 : i32
    
    txn.if %ready {
      %c1 = arith.constant 1 : i32
      %new_count = arith.addi %count, %c1 : i32
      txn.call @counter::@write(%new_count) : (i32) -> ()
      txn.yield
    } else {
      // Do nothing when not ready
      txn.yield
    }
    txn.return
  }
  
  // Rule without guard (always enabled)
  txn.rule @alwaysEnabled {
    %c0 = arith.constant 0 : i32
    txn.call @counter::@write(%c0) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@incrementWhenReady, @alwaysEnabled]
}

// CHECK-LABEL: firrtl.circuit "GuardEvaluation"
// CHECK: firrtl.module @GuardEvaluation

// The key test: verify that incrementWhenReady's will-fire signal
// is based on a comparison (the guard), not a constant true
// CHECK-DAG: %[[GUARD_RESULT:[0-9]+]] = firrtl.gt %{{.*}}, %{{.*}} :
// CHECK-DAG: %[[TRUE:.*]] = firrtl.constant 1 : !firrtl.uint<1>
// CHECK-DAG: %incrementWhenReady_wf = firrtl.node %[[GUARD_RESULT]] : !firrtl.uint<1>

// Verify that alwaysEnabled uses constant true
// CHECK-DAG: %alwaysEnabled_wf = firrtl.node %[[TRUE]] : !firrtl.uint<1>

// Verify both rules execute conditionally based on their will-fire signals
// CHECK-DAG: firrtl.when %incrementWhenReady_wf
// CHECK-DAG: firrtl.when %alwaysEnabled_wf