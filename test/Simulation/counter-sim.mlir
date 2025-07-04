// RUN: sharp-opt %s --sharp-simulate | FileCheck %s

// Simple counter module for transaction-level simulation
// CHECK-LABEL: txn.module @Counter
txn.module @Counter {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  
  // State variable
  %count = txn.reg %zero : i32
  
  // CHECK: txn.value_method @getValue
  txn.value_method @getValue() -> i32 {
    %val = txn.read %count : i32
    txn.return %val : i32
  }
  
  // CHECK: txn.action_method @increment
  txn.action_method @increment() {
    %old = txn.read %count : i32
    %new = arith.addi %old, %one : i32
    txn.write %count, %new : i32
  }
  
  // CHECK: txn.action_method @decrement
  txn.action_method @decrement() {
    %old = txn.read %count : i32
    %new = arith.subi %old, %one : i32
    txn.write %count, %new : i32
  }
  
  // Conflict matrix - increment and decrement conflict
  txn.schedule @explicit_schedule conflicts {
    "increment" -> "decrement" : "C",
    "getValue" -> "increment" : "CF",
    "getValue" -> "decrement" : "CF"
  }
}

// Testbench module
// CHECK-LABEL: txn.module @CounterTestBench
txn.module @CounterTestBench {
  %counter = txn.instance @counter of @Counter
  
  // Test sequence
  txn.rule @test_sequence {
    // Increment 3 times
    txn.call %counter::@increment() : () -> ()
    txn.call %counter::@increment() : () -> ()
    txn.call %counter::@increment() : () -> ()
    
    // Get value
    %val = txn.call %counter::@getValue() : () -> i32
    
    // Assert value is 3
    %three = arith.constant 3 : i32
    %eq = arith.cmpi eq, %val, %three : i32
    cf.assert %eq, "Counter value should be 3"
    
    // Decrement once
    txn.call %counter::@decrement() : () -> ()
    
    // Final value should be 2
    %val2 = txn.call %counter::@getValue() : () -> i32
    %two = arith.constant 2 : i32
    %eq2 = arith.cmpi eq, %val2, %two : i32
    cf.assert %eq2, "Counter value should be 2"
  }
}

// Simulation attributes
// CHECK: sharp.sim
sharp.sim @main {
  module = @CounterTestBench,
  max_cycles = 100 : i64,
  debug = true
}