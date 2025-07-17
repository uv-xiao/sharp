// RUN: sharp-opt %s --convert-txn-to-firrtl | FileCheck %s

// Test guard condition with abort
txn.module @GuardAbort {
  %reg = txn.instance @reg of @Register<i1> : index
  
  // Rule with guard condition and abort
  txn.rule @guarded_rule {
    %flag = txn.call @reg::@read() : () -> i1
    
    txn.if %flag {
      // Just abort if flag is true
      txn.abort
    } else {
      txn.yield
    }
    txn.yield
  }
  
  txn.schedule [@guarded_rule]
}

// CHECK: firrtl.circuit "GuardAbort"
// CHECK: firrtl.module @GuardAbort
// The flag should be used as the guard condition
// CHECK: firrtl.when %{{.*}} :