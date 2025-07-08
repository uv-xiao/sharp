// RUN: sharp-opt %s --sharp-collect-primitive-actions --convert-txn-to-firrtl=will-fire-mode="most-dynamic" | FileCheck %s

// Test most-dynamic mode which tracks primitive action calls

txn.primitive @Register type = "hw" interface = !txn.module<"Register"> {
  txn.fir_value_method @read() : () -> i32
  txn.fir_action_method @write() : (i32) -> ()
  txn.clock_by @clk
  txn.reset_by @rst
  txn.schedule [@write] {
    conflict_matrix = {
      "read,read" = 3 : i32,
      "read,write" = 3 : i32,
      "write,read" = 3 : i32,
      "write,write" = 2 : i32
    }
  }
} {firrtl.impl = "Register_impl"}

txn.module @Counter {
  %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
  
  txn.action_method @increment() {
    %val = txn.call @reg::@read() : () -> i32
    %c1 = arith.constant 1 : i32
    %new = arith.addi %val, %c1 : i32
    txn.call @reg::@write(%new) : (i32) -> ()
    txn.return
  }
  
  txn.action_method @reset() {
    %c0 = arith.constant 0 : i32
    txn.call @reg::@write(%c0) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@increment, @reset] {
    conflict_matrix = {
      "increment,reset" = 2 : i32,  // C - both write to reg
      "reset,increment" = 2 : i32
    }
  }
}

txn.module @TestMostDynamic {
  %counter1 = txn.instance @counter1 of @Counter : !txn.module<"Counter">
  %counter2 = txn.instance @counter2 of @Counter : !txn.module<"Counter">
  
  // Action that increments counter1
  txn.rule @incrementFirst {
    txn.call @counter1::@increment() : () -> ()
    txn.return
  }
  
  // Action that resets counter1 - should conflict with incrementFirst
  txn.rule @resetFirst {
    txn.call @counter1::@reset() : () -> ()
    txn.return
  }
  
  // Action that increments counter2 - should NOT conflict with counter1 actions
  txn.rule @incrementSecond {
    txn.call @counter2::@increment() : () -> ()
    txn.return
  }
  
  txn.schedule [@incrementFirst, @resetFirst, @incrementSecond]
}

// CHECK-LABEL: firrtl.circuit "TestMostDynamic"
// CHECK: firrtl.module @TestMostDynamic

// In most-dynamic mode, the pass should detect that:
// - incrementFirst and resetFirst both access counter1's register (conflict)
// - incrementSecond accesses counter2's register (no conflict with others)

// The will-fire signals should reflect primitive-level tracking:
// CHECK-DAG: %incrementFirst_wf = firrtl.node
// CHECK-DAG: %resetFirst_wf = firrtl.node
// CHECK-DAG: %incrementSecond_wf = firrtl.node

// resetFirst should be blocked when incrementFirst fires (both access counter1::reg)
// CHECK-DAG: firrtl.not %incrementFirst_wf
// CHECK-DAG: firrtl.and %{{.*}}, %{{.*}}