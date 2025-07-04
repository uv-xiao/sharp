// RUN: sharp-opt %s -txn-simulate=mode=jit | FileCheck %s

// CHECK-LABEL: JIT execution successful
txn.module @SimpleCounter attributes {moduleName = "SimpleCounter"} {
  txn.value_method @read() -> i32 {
    %zero = arith.constant 0 : i32
    txn.return %zero : i32
  }
  
  txn.action_method @increment() {
    txn.yield
  }
  
  txn.rule @auto_incr {
    %true = arith.constant true
    txn.if %true {
      txn.call @increment() : () -> ()
      txn.yield
    } else {
      txn.yield
    }
  }
  
  txn.schedule [@auto_incr, @read, @increment]
}