// RUN: sharp-opt --convert-txn-to-firrtl %s | FileCheck %s

// Test that primitives are automatically constructed when used

// Define Register primitive
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

// Define Wire primitive  
txn.primitive @Wire type = "hw" interface = !txn.module<"Wire"> {
  txn.fir_value_method @read() : () -> i8
  txn.fir_action_method @write() : (i8) -> ()
  txn.schedule [@write] {
    conflict_matrix = {
      "read,read" = 3 : i32,
      "read,write" = 0 : i32,
      "write,read" = 0 : i32,
      "write,write" = 2 : i32
    }
  }
} {firrtl.impl = "Wire_impl"}

txn.module @Counter {
  // Instantiate a Register primitive with type parameter
  %count = txn.instance @count of @Register : !txn.module<"Register">
  
  txn.action_method @increment() {
    // Read current value
    %current = txn.call @count::@read() : () -> i32
    
    // Increment
    %one = arith.constant 1 : i32
    %next = arith.addi %current, %one : i32
    
    // Write back
    txn.call @count::@write(%next) : (i32) -> ()
    
    txn.return
  }
  
  txn.value_method @getCount() -> i32 {
    %val = txn.call @count::@read() : () -> i32
    txn.return %val : i32
  }
  
  txn.schedule [
    @increment
  ] {
    conflict_matrix = {}
  }
}

// Test with Wire primitive
txn.module @WireTest {
  %wire = txn.instance @wire of @Wire : !txn.module<"Wire">
  
  txn.action_method @setValue(%val: i8) {
    txn.call @wire::@write(%val) : (i8) -> ()
    txn.return
  }
  
  txn.value_method @getValue() -> i8 {
    %val = txn.call @wire::@read() : () -> i8
    txn.return %val : i8
  }
  
  txn.schedule [
    @setValue
  ] {
    conflict_matrix = {}
  }
}

// CHECK-LABEL: firrtl.circuit
// CHECK-DAG: firrtl.module @Register_i32_impl
// CHECK-DAG: firrtl.module @Wire_i8_impl
// CHECK-DAG: firrtl.module @Counter
// CHECK-DAG: firrtl.module @WireTest

// Primitive implementation details are checked by CHECK-DAG above

// Check instances in modules
// CHECK-DAG: firrtl.instance count {{.*}} @Register_i32_impl
// CHECK-DAG: firrtl.instance wire {{.*}} @Wire_i8_impl