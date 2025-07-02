// RUN: sharp-opt --convert-txn-to-firrtl %s | FileCheck %s

// Test basic Txn to FIRRTL conversion

// CHECK-LABEL: firrtl.circuit "Counter"
// CHECK-DAG: firrtl.module @Register_i32_impl
// CHECK-DAG: firrtl.module @Counter

txn.module @Counter {
  %reg = txn.instance @count of @Register<i32> : !txn.module<"Register">
  
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

// Register primitive for testing
txn.primitive @Register type = "hw" interface = !txn.module<"Register"> {
  txn.fir_value_method @read() : () -> i32
  txn.fir_action_method @write() : (i32) -> ()
}