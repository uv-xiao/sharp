// RUN: sharp-opt %s --convert-txn-to-firrtl 2>&1 | FileCheck %s

// Define primitives
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

// Test that multi-cycle operations emit proper errors in FIRRTL conversion
txn.module @MultiCycleFIRRTL {
  %reg = txn.instance @reg of @Register : !txn.module<"Register">
  
  txn.action_method @withFuture() {
    %v = txn.call @reg::@read() : () -> i32
    
    // CHECK: error: future operations are not yet supported in FIRRTL conversion. Multi-cycle execution requires additional synthesis infrastructure.
    txn.future {
      %done = txn.launch after 1 {
        %zero = arith.constant 0 : i32
        txn.call @reg::@write(%zero) : (i32) -> ()
        txn.yield
      }
    }
    txn.return
  }
  
  txn.schedule [@withFuture]
}