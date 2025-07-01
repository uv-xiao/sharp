// RUN: sharp-opt %s | FileCheck %s

// Test parametric primitive instance creation
// We can't define true parametric primitives with type variables like T
// But we can test instances of primitives with type arguments

txn.module @TestParametric {
  // CHECK: txn.instance @reg32 of @Register<i32>
  %reg32 = txn.instance @reg32 of @Register<i32> : !txn.module<"Register">
  
  // CHECK: txn.instance @reg8 of @Register<i8>
  %reg8 = txn.instance @reg8 of @Register<i8> : !txn.module<"Register">
  
  // CHECK: txn.instance @wire64 of @Wire<i64>
  %wire64 = txn.instance @wire64 of @Wire<i64> : !txn.module<"Wire">
  
  txn.action_method @test() {
    // Use i32 register
    %val32 = txn.call @reg32::@read() : () -> i32
    %c1_i32 = arith.constant 1 : i32
    %new32 = arith.addi %val32, %c1_i32 : i32
    txn.call @reg32::@write(%new32) : (i32) -> ()
    
    // Use i8 register
    %val8 = txn.call @reg8::@read() : () -> i8
    %c1_i8 = arith.constant 1 : i8
    %new8 = arith.addi %val8, %c1_i8 : i8
    txn.call @reg8::@write(%new8) : (i8) -> ()
    
    // Use i64 wire
    %c42_i64 = arith.constant 42 : i64
    txn.call @wire64::@write(%c42_i64) : (i64) -> ()
    
    txn.return
  }
  
  txn.value_method @read() -> i64 {
    %val = txn.call @wire64::@read() : () -> i64
    txn.return %val : i64
  }
  
  txn.schedule [@test, @read] {
    conflict_matrix = {}
  }
}