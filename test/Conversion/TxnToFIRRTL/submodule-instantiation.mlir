// RUN: sharp-opt --convert-txn-to-firrtl %s | FileCheck %s

// Test submodule instantiation and method calls
txn.module @Register {
  txn.value_method @read() -> i32 {
    %c0 = arith.constant 0 : i32
    txn.return %c0 : i32
  }
  
  txn.action_method @write(%val: i32) {
    txn.return
  }
  
  txn.schedule [@write] {
    conflict_matrix = {
      "write,write" = 2 : i32
    }
  }
}

txn.module @Top {
  %reg1 = txn.instance @reg1 of @Register : index
  %reg2 = txn.instance @reg2 of @Register : index
  
  txn.rule @copy_rule {
    // Read from reg1
    %val = txn.call @reg1::@read() : () -> i32
    // Write to reg2
    txn.call @reg2::@write(%val) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@copy_rule] {
    conflict_matrix = {}
  }
}

// CHECK: module {
// CHECK-NEXT: firrtl.circuit "Top" {
// CHECK: firrtl.module @Register
// CHECK: firrtl.module @Top

// Check instances are created
// CHECK-DAG: firrtl.instance reg1 interesting_name @Register
// CHECK-DAG: firrtl.instance reg2 interesting_name @Register

// Check clock/reset connections are made
// CHECK-DAG: firrtl.connect %reg1_clock, %clock
// CHECK-DAG: firrtl.connect %reg1_reset, %reset
// CHECK-DAG: firrtl.connect %reg2_clock, %clock
// CHECK-DAG: firrtl.connect %reg2_reset, %reset

// Check rule creates method calls via connections
// CHECK-DAG: firrtl.when %copy_rule_wf
// CHECK-DAG: firrtl.connect %reg1_read_EN
// CHECK-DAG: firrtl.connect %reg2_writeOUT
// CHECK-DAG: firrtl.connect %reg2_writeEN