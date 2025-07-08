// RUN: sharp-opt %s -sharp-simulate=mode=jit | FileCheck %s

// CHECK-DAG: llvm.func @SimpleCounter_scheduler()
// CHECK-DAG: llvm.func @SimpleCounter_read() -> i32
// CHECK-DAG: llvm.func @SimpleCounter_increment() -> i1
// CHECK-DAG: llvm.func @SimpleCounter_rule_auto_incr() -> i1
txn.module @SimpleCounter attributes {moduleName = "SimpleCounter"} {
  txn.value_method @read() -> i32 {
    %zero = arith.constant 0 : i32
    txn.return %zero : i32
  }
  
  txn.action_method @increment() {
    txn.return
  }
  
  txn.rule @auto_incr {
    txn.call @increment() : () -> ()
    txn.yield
  }
  
  txn.schedule [@auto_incr]
}