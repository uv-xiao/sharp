// RUN: sharp-opt %s -txn-export-verilog 2>&1 | FileCheck %s

// Test end-to-end Verilog export pipeline: Txn -> FIRRTL -> HW -> Verilog

// CHECK: module Counter(
// CHECK:   input{{.*}}clock,
// CHECK:{{.*}}reset,
// CHECK:   output [31:0] getValueOUT,
// CHECK:   input{{.*}}getValue_EN,
// CHECK:{{.*}}incrementEN,
// CHECK:   output{{.*}}incrementRDY

txn.module @Counter attributes {moduleName = "Counter"} {
  txn.value_method @getValue() -> i32 attributes {timing = "combinational"} {
    %c0 = arith.constant 0 : i32
    txn.return %c0 : i32
  }
  
  txn.action_method @increment() attributes {timing = "static(1)"} {
    %c1 = arith.constant 1 : i32
    txn.yield
  }
  
  txn.schedule [@getValue, @increment] {
    conflict_matrix = {
      "getValue,increment" = 3 : i32  // ConflictFree
    }
  }
}

// CHECK: module Adder(
// CHECK:   input{{.*}}clock,
// CHECK:{{.*}}reset,
// CHECK:   output [31:0] addOUT,
// CHECK:   input{{.*}}add_EN

txn.module @Adder attributes {moduleName = "Adder"} {
  txn.value_method @add(%a: i32, %b: i32) -> i32 attributes {timing = "combinational"} {
    %sum = arith.addi %a, %b : i32
    txn.return %sum : i32
  }
  
  txn.schedule [@add]
}