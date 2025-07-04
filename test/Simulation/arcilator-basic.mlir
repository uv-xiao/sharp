// RUN: sharp-opt %s -sharp-arcilator | FileCheck %s

// CHECK: Successfully converted to Arc dialect for RTL simulation
// CHECK: To simulate this module:
// CHECK: arcilator

txn.module @Counter attributes {moduleName = "Counter"} {
  txn.value_method @getCount() -> i32 {
    %zero = arith.constant 0 : i32
    txn.return %zero : i32
  }
  
  txn.action_method @increment() {
    txn.yield
  }
  
  txn.rule @tick {
    %true = arith.constant true
    txn.if %true {
      txn.call @increment() : () -> ()
      txn.yield
    } else {
      txn.yield
    }
  }
  
  txn.schedule [@tick, @getCount, @increment]
}