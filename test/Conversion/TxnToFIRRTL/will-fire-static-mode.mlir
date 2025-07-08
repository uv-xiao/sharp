// RUN: sharp-opt --convert-txn-to-firrtl %s | FileCheck %s

// Test static mode will-fire logic with conflicting actions

// CHECK-LABEL: firrtl.circuit "StaticWillFire"
txn.module @StaticWillFire {
  %reg = txn.instance @r of @Register : !txn.module<"Register">
  
  txn.rule @rule1 {
    %c0 = arith.constant 0 : i32
    txn.call @r::@write(%c0) : (i32) -> ()
    txn.return
  }
  
  txn.rule @rule2 {
    %c1 = arith.constant 1 : i32
    txn.call @r::@write(%c1) : (i32) -> ()
    txn.return
  }
  
  // In static mode, rule2 should be prevented from firing if rule1 fires
  // CHECK: %rule1_wf = firrtl.node %c1_ui1 : !firrtl.uint<1>
  // CHECK: %rule1_wf_{{[0-9]+}} = firrtl.node %rule1_wf {name = "rule1_wf"} : !firrtl.uint<1>
  // CHECK: %[[NOT_RULE1:.*]] = firrtl.not %rule1_wf_{{[0-9]+}} : (!firrtl.uint<1>) -> !firrtl.uint<1>
  // CHECK: %[[RULE2_ENABLED:.*]] = firrtl.and %c1_ui1{{.*}}, %[[NOT_RULE1]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  // CHECK: %rule2_wf = firrtl.node %[[RULE2_ENABLED]] : !firrtl.uint<1>
  
  txn.schedule [@rule1, @rule2] {
    conflict_matrix = {
      "r::write,r::write" = 2 : i32  // C (conflict)
    }
  }
}

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