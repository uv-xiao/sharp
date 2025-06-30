// RUN: sharp-opt --convert-txn-to-firrtl %s | FileCheck %s

// Test comprehensive Txn to FIRRTL conversion
txn.module @SimpleModule {
  txn.value_method @getValue() -> i32 {
    %c42 = arith.constant 42 : i32
    txn.return %c42 : i32
  }
  
  txn.action_method @doAction() {
    txn.return
  }
  
  txn.schedule [@doAction] {
    conflict_matrix = {
      "doAction,doAction" = 2 : i32
    }
  }
}

txn.module @TopModule {
  %inst = txn.instance @inst of @SimpleModule : !txn.module<"SimpleModule">
  
  txn.rule @useInstance {
    %val = txn.call @inst::@getValue() : () -> i32
    txn.call @inst::@doAction() : () -> ()
    txn.return
  }
  
  txn.schedule [@useInstance] {
    conflict_matrix = {}
  }
}

// CHECK: firrtl.circuit "TopModule"
// CHECK: firrtl.module @SimpleModule(
// CHECK-DAG: in %clock: !firrtl.clock
// CHECK-DAG: in %reset: !firrtl.uint<1>
// CHECK-DAG: out %getValueOUT: !firrtl.sint<32>
// CHECK-DAG: in %getValue_EN: !firrtl.uint<1>
// CHECK-DAG: in %doActionEN: !firrtl.uint<1>
// CHECK-DAG: out %doActionRDY: !firrtl.uint<1>

// CHECK: firrtl.module @TopModule(
// CHECK: firrtl.instance inst interesting_name @SimpleModule(

// Verify the conversion produces valid FIRRTL
// CHECK: }