// RUN: sharp-opt %s -sharp-simulate=mode=jit | FileCheck %s

// CHECK-DAG: llvm.func @BasicCounter_scheduler()
// CHECK-DAG: llvm.func @BasicCounter_getValue() -> i32
// CHECK-DAG: llvm.func @BasicCounter_setValue(%arg0: i32) -> i1
// CHECK-DAG: llvm.func @BasicCounter_rule_doWork() -> i1
txn.module @BasicCounter attributes {moduleName = "BasicCounter"} {
  txn.value_method @getValue() -> i32 {
    %c42 = arith.constant 42 : i32
    txn.return %c42 : i32
  }
  
  txn.action_method @setValue(%val: i32) {
    txn.return
  }
  
  txn.rule @doWork {
    %c100 = arith.constant 100 : i32
    txn.call @setValue(%c100) : (i32) -> ()
    txn.yield
  }
  
  txn.schedule [@doWork]
}