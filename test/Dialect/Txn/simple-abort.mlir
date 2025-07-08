// RUN: sharp-opt %s --convert-txn-to-firrtl | FileCheck %s

// Simple test for abort handling
txn.module @SimpleAbort {
  %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
  
  // Simple rule with abort
  txn.rule @rule_abort {
    // Just abort immediately
    txn.abort
  }
  
  // Another rule
  txn.rule @rule_normal {
    %c10 = arith.constant 10 : i32
    txn.call @reg::@write(%c10) : (i32) -> ()
    txn.yield
  }
  
  txn.schedule [@rule_abort, @rule_normal]
}

// CHECK: firrtl.circuit "SimpleAbort"