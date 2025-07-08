// RUN: sharp-opt %s | FileCheck %s

// Test rule attributes for FIRRTL generation

// Test rule with custom prefix attribute
txn.module @TestRuleAttrs {
  %reg = txn.instance @r of @Register : !txn.module<"Register">
  
  // CHECK: txn.rule @resetRule {
  // CHECK: } {prefix = "rl_reset"}
  txn.rule @resetRule {
    %c0 = arith.constant 0 : i32
    txn.call @r::@write(%c0) : (i32) -> ()
    txn.return
  } {prefix = "rl_reset"}
  
  // CHECK: txn.rule @computeRule {
  // CHECK: } {prefix = "do_compute", timing = "static(2)"}
  txn.rule @computeRule {
    %val = txn.call @r::@read() : () -> i32
    %c1 = arith.constant 1 : i32
    %inc = arith.addi %val, %c1 : i32
    txn.call @r::@write(%inc) : (i32) -> ()
    txn.return
  } {timing = "static(2)", prefix = "do_compute"}
  
  // CHECK: txn.rule @normalRule {
  txn.rule @normalRule {
    %val = txn.call @r::@read() : () -> i32
    txn.return
  }
  
  txn.schedule [@resetRule, @computeRule, @normalRule] {conflict_matrix = {}}
}

// Support primitive for testing
txn.primitive @Register type = "hw" interface = !txn.module<"Register"> {
  txn.fir_value_method @read() {firrtl.port = "read_data"} : () -> i32
  txn.fir_action_method @write() {firrtl.data_port = "write_data", firrtl.enable_port = "write_enable"} : (i32) -> ()
  txn.clock_by @clk
  txn.reset_by @rst
  txn.schedule [@read, @write] {conflict_matrix = {
    "read,read" = 3 : i32,
    "read,write" = 3 : i32,
    "write,read" = 3 : i32,
    "write,write" = 2 : i32
  }}
} {firrtl.impl = "Register_impl"}