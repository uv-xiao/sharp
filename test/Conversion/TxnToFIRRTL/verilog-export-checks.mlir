// RUN: sharp-opt --txn-export-verilog %s -o - 2>&1 | FileCheck %s

// Test comprehensive Verilog export validation

// Test 1: Basic module with value and action methods
txn.module @BasicModule {
  txn.value_method @getValue() -> i32 {
    %c42 = arith.constant 42 : i32
    txn.return %c42 : i32
  }
  
  txn.action_method @doAction() -> () {
    txn.return
  }
  
  txn.schedule [@doAction] {
    conflict_matrix = {}
  }
}

// CHECK-DAG: module BasicModule(
// CHECK-DAG: assign getValueOUT = 32'h2A;
// CHECK-DAG: assign doActionRDY = 1'h1;

// Test 2: Module with arithmetic operations
txn.module @ArithModule {
  txn.value_method @compute(%a: i32, %b: i32) -> i32 {
    %sum = arith.addi %a, %b : i32
    txn.return %sum : i32
  }
  
  txn.schedule [] {
    conflict_matrix = {}
  }
}

// CHECK-DAG: module ArithModule(

// Test 3: Different integer widths
txn.module @IntWidths {
  txn.value_method @getByte() -> i8 {
    %c255 = arith.constant 255 : i8
    txn.return %c255 : i8
  }
  
  txn.value_method @getShort() -> i16 {
    %c1000 = arith.constant 1000 : i16
    txn.return %c1000 : i16
  }
  
  txn.schedule [] {
    conflict_matrix = {}
  }
}

// CHECK-DAG: module IntWidths(
// CHECK-DAG: assign getByteOUT = 8'hFF;
// CHECK-DAG: assign getShortOUT = 16'h3E8;

// Test 4: Conflict handling
txn.module @ConflictModule {
  txn.action_method @write1() -> () {
    txn.return
  }
  
  txn.action_method @write2() -> () {
    txn.return
  }
  
  txn.schedule [@write1, @write2] {
    conflict_matrix = {
      "write1,write2" = 2 : i32  // Conflict
    }
  }
}

// CHECK-DAG: module ConflictModule(
// CHECK-DAG: write1RDY
// CHECK-DAG: write2RDY