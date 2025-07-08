// RUN: sharp-opt --txn-export-verilog %s -o - 2>&1 | FileCheck %s

// Simple test for Verilog export without primitives

// CHECK: module SimpleCounter(
// CHECK-DAG: input         clock,
// CHECK-DAG:               reset,
// CHECK-DAG: output [31:0] getValueOUT,
// CHECK-DAG: input         getValue_EN,
// CHECK-DAG:               resetEN,
// CHECK-DAG: output        resetRDY

txn.module @SimpleCounter {
  txn.value_method @getValue() -> i32 {
    %c42 = arith.constant 42 : i32
    txn.return %c42 : i32
  }
  
  txn.action_method @reset() -> () {
    txn.return
  }
  
  txn.schedule [@reset] {
    conflict_matrix = {}
  }
}

// CHECK: assign getValueOUT = 32'h2A;
// CHECK: assign resetRDY = 1'h1;
// CHECK: endmodule