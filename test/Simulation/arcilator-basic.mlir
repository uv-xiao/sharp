// RUN: sharp-opt %s -sharp-arcilator 2>&1 | FileCheck %s

// CHECK: Successfully converted to Arc dialect for RTL simulation
// CHECK: To simulate this module:
// CHECK: arcilator

txn.module @Counter attributes {moduleName = "Counter"} {
  txn.value_method @getCount() -> i32 {
    %zero = arith.constant 0 : i32
    txn.return %zero : i32
  }
  
  txn.value_method @getValue() -> i32 {
    %val = arith.constant 42 : i32
    txn.return %val : i32
  }
  
  txn.schedule [@getCount, @getValue]
}