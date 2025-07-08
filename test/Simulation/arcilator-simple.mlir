// RUN: sharp-opt %s -sharp-arcilator 2>&1 | FileCheck %s

// CHECK: Successfully converted to Arc dialect for RTL simulation
// CHECK: To simulate this module:

txn.module @Adder attributes {moduleName = "Adder"} {
  txn.value_method @add(%a: i32, %b: i32) -> i32 {
    %sum = arith.addi %a, %b : i32
    txn.return %sum : i32
  }
  
  txn.schedule []
}