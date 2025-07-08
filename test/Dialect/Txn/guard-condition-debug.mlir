// RUN: sharp-opt %s --convert-txn-to-firrtl | FileCheck %s

// Debug test for guard condition conversion
txn.module @GuardDebug {
  %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
  
  // Rule with simple guard
  txn.rule @rule_simple {
    %c10 = arith.constant 10 : i32
    %c5 = arith.constant 5 : i32
    %cond = arith.cmpi slt, %c5, %c10 : i32
    
    txn.if %cond {
      %c20 = arith.constant 20 : i32
      txn.call @reg::@write(%c20) : (i32) -> ()
      txn.yield
    } else {
      txn.yield
    }
    txn.yield
  }
  
  txn.schedule [@rule_simple]
}

// CHECK: firrtl.circuit "GuardDebug"