// RUN: sharp-opt %s -sharp-simulate=mode=jit | FileCheck %s

// CHECK: JIT execution successful
txn.module @BasicCounter attributes {moduleName = "BasicCounter"} {
  txn.value_method @getValue() -> i32 {
    %c42 = arith.constant 42 : i32
    txn.return %c42 : i32
  }
  
  txn.action_method @setValue(%val: i32) {
    txn.yield
  }
  
  txn.schedule [@getValue, @setValue]
}