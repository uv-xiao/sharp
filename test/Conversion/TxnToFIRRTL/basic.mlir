// RUN: sharp-opt --convert-txn-to-firrtl %s | FileCheck %s

// Test basic Txn to FIRRTL conversion

// CHECK-LABEL: firrtl.circuit "Counter"
// CHECK: firrtl.module @Register
// CHECK: firrtl.module @Counter

txn.module @Counter {
  %reg = txn.instance @count of @Register : !txn.module<"Register">
  
  txn.rule @increment {
    %val = txn.call @count::@read() : () -> i32
    %c1 = arith.constant 1 : i32
    %inc = arith.addi %val, %c1 : i32
    txn.call @count::@write(%inc) : (i32) -> ()
    txn.return
  }
  
  txn.value_method @getValue() -> i32 {
    %val = txn.call @count::@read() : () -> i32
    txn.return %val : i32
  }
  
  txn.action_method @reset() -> () {
    %c0 = arith.constant 0 : i32
    txn.call @count::@write(%c0) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@increment, @getValue, @reset] {
    conflict_matrix = {
      "increment,reset" = 2 : i32,  // C
      "getValue,reset" = 0 : i32    // SB
    }
  }
}

// Simplified Register primitive for testing
txn.module @Register {
  txn.value_method @read() -> i32 {
    %0 = arith.constant 0 : i32
    txn.return %0 : i32
  }
  txn.action_method @write(%val: i32) -> () {
    txn.return
  }
  txn.schedule [@read, @write] {
    conflict_matrix = {
      "write,write" = 2 : i32,
      "write,read" = 0 : i32
    }
  }
}