// RUN: sharp-opt %s | FileCheck %s

// Test simple action method generation

// CHECK-LABEL: txn.module @SimpleModule
txn.module @SimpleModule {
  %count = txn.instance @count of @Register<i32> : !txn.module<"Register">
  
  txn.action_method @increment() {
    %v = txn.call @count.read() : () -> i32
    %one = arith.constant 1 : i32
    %next = arith.addi %v, %one : i32
    txn.call @count.write(%next) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@increment]
}