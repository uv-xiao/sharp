// RUN: sharp-opt --convert-txn-to-firrtl %s | FileCheck %s --check-prefix=STATIC
// RUN: sharp-opt --convert-txn-to-firrtl="will-fire-mode=static" %s | FileCheck %s --check-prefix=STATIC
// RUN: sharp-opt --convert-txn-to-firrtl="will-fire-mode=dynamic" %s | FileCheck %s --check-prefix=DYNAMIC

// Test that both static and dynamic will-fire modes work correctly

// STATIC-LABEL: firrtl.circuit "WillFireModes"
// DYNAMIC-LABEL: firrtl.circuit "WillFireModes"
txn.module @WillFireModes {
  %reg = txn.instance @r of @Register : index
  
  txn.rule @simple {
    %c0 = arith.constant 0 : i32
    txn.call @r::@write(%c0) : (i32) -> ()
    txn.return
  }
  
  // Both static and dynamic modes should generate will-fire signals
  // STATIC: %simple_wf = firrtl.node
  // DYNAMIC: %simple_wf = firrtl.node
  
  txn.schedule [@simple] {
    conflict_matrix = {}
  }
}

txn.primitive @Register type = "hw" interface = index {
  txn.fir_value_method @read() : () -> i32
  txn.fir_action_method @write() : (i32) -> ()
}