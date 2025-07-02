// RUN: sharp-opt --txn-export-verilog %s -o - 2>&1 | FileCheck %s

// Simple test for Verilog export without primitives

// CHECK: module SimpleCounter

txn.module @SimpleCounter {
  txn.value_method @getValue() -> i32 {
    %c42 = arith.constant 42 : i32
    txn.return %c42 : i32
  }
  
  txn.action_method @reset() -> () {
    txn.return
  }
  
  txn.schedule [@getValue, @reset] {
    conflict_matrix = {}
  }
}