// RUN: sharp-opt %s | FileCheck %s

// Test method attributes for FIRRTL generation

// Test value method with custom attributes
txn.module @TestValueMethodAttrs {
  %reg = txn.instance @r of @Register : !txn.module<"Register">
  
  // CHECK: txn.value_method @getValue() -> i32 attributes {prefix = "get", result = "_data"}
  txn.value_method @getValue() -> i32 attributes {prefix = "get", result = "_data"} {
    %val = txn.call @r::@read() : () -> i32
    txn.return %val : i32
  }
  
  // CHECK: txn.value_method @alwaysReady() -> i32 attributes {always_ready}
  txn.value_method @alwaysReady() -> i32 attributes {always_ready} {
    %val = txn.call @r::@read() : () -> i32
    txn.return %val : i32
  }
  
  txn.schedule [@getValue, @alwaysReady] {conflict_matrix = {}}
}

// Test action method with custom attributes  
txn.module @TestActionMethodAttrs {
  %reg = txn.instance @r of @Register : !txn.module<"Register">
  
  // CHECK: txn.action_method @doReset() attributes {enable = "_en", prefix = "do", ready = "_rdy", result = "_out"}
  txn.action_method @doReset() -> () attributes {prefix = "do", ready = "_rdy", enable = "_en", result = "_out"} {
    %c0 = arith.constant 0 : i32
    txn.call @r::@write(%c0) : (i32) -> ()
    txn.return
  }
  
  // CHECK: txn.action_method @alwaysReadyAction() attributes {always_ready}
  txn.action_method @alwaysReadyAction() -> () attributes {always_ready} {
    %c1 = arith.constant 1 : i32
    txn.call @r::@write(%c1) : (i32) -> ()
    txn.return
  }
  
  // CHECK: txn.action_method @alwaysEnabledAction() attributes {always_enable}
  txn.action_method @alwaysEnabledAction() -> () attributes {always_enable} {
    %c2 = arith.constant 2 : i32
    txn.call @r::@write(%c2) : (i32) -> ()
    txn.return
  }
  
  // CHECK: txn.action_method @fullyOptimized() attributes {always_enable, always_ready}
  txn.action_method @fullyOptimized() -> () attributes {always_ready, always_enable} {
    %c3 = arith.constant 3 : i32
    txn.call @r::@write(%c3) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@doReset, @alwaysReadyAction, @alwaysEnabledAction, @fullyOptimized] {conflict_matrix = {}}
}


// Test FIR method attributes on primitives
txn.primitive @TestPrimitive type = "hw" interface = !txn.module<"TestPrimitive"> {
  // CHECK: txn.fir_value_method @getValue() {prefix = "val", result = "_out"} : () -> i32
  txn.fir_value_method @getValue() {prefix = "val", result = "_out"} : () -> i32
  
  // CHECK: txn.fir_action_method @doAction() {enable = "_e", prefix = "act", ready = "_r", result = "_d"} : (i32) -> ()
  txn.fir_action_method @doAction() {prefix = "act", ready = "_r", enable = "_e", result = "_d"} : (i32) -> ()
  
  txn.clock_by @clk
  txn.reset_by @rst
  
  txn.schedule [@getValue, @doAction] {conflict_matrix = {
    "getValue,getValue" = 3 : i32,
    "getValue,doAction" = 3 : i32,
    "doAction,getValue" = 3 : i32,
    "doAction,doAction" = 2 : i32
  }}
} {firrtl.impl = "TestPrimitive_impl"}

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