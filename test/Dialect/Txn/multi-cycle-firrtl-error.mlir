// RUN: sharp-opt %s --convert-txn-to-firrtl 2>&1 | FileCheck %s

// Test that multi-cycle operations emit proper errors in FIRRTL conversion
txn.module @MultiCycleFIRRTL {
  %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
  
  txn.action_method @withFuture() attributes {multicycle = true} {
    %v = txn.call @reg::@read() : () -> i32
    
    // CHECK: error: FutureOp not yet supported in FIRRTL conversion
    txn.future {
      %done = txn.launch after 1 {
        %zero = arith.constant 0 : i32
        txn.call @reg::@write(%zero) : (i32) -> ()
        txn.yield
      }
    }
    txn.return
  }
  
  txn.schedule [@withFuture]
}