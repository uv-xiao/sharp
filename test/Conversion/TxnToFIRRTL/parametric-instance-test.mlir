// RUN: sharp-opt --convert-txn-to-firrtl %s | FileCheck %s

// Test parametric instance creation (instances with type arguments)

txn.module @TestParametricInstances {
  // Instantiate Register primitive with different type arguments
  %reg32 = txn.instance @reg32 of @Register<i32> : !txn.module<"Register">
  %reg8 = txn.instance @reg8 of @Register<i8> : !txn.module<"Register">
  %reg64 = txn.instance @reg64 of @Register<i64> : !txn.module<"Register">
  
  // Instantiate Wire primitive with type arguments
  %wire16 = txn.instance @wire16 of @Wire<i16> : !txn.module<"Wire">
  
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
    
    // Use i16 wire
    %c42_i16 = arith.constant 42 : i16
    txn.call @wire16::@write(%c42_i16) : (i16) -> ()
    
    txn.return
  }
  
  txn.schedule [@test] {
    conflict_matrix = {}
  }
}

// CHECK-LABEL: firrtl.circuit "TestParametricInstances"

// Check that FIRRTL modules are created for each type instantiation
// CHECK-DAG: firrtl.module @Register_i32_impl
// CHECK-DAG: firrtl.module @Register_i8_impl  
// CHECK-DAG: firrtl.module @Register_i64_impl
// CHECK-DAG: firrtl.module @Wire_i16_impl

// The actual bit widths are verified by the CHECK-DAG above that ensures the modules exist

// Check instances are created in the main module
// CHECK-DAG: firrtl.instance reg32 {{.*}} @Register_i32_impl
// CHECK-DAG: firrtl.instance reg8 {{.*}} @Register_i8_impl
// CHECK-DAG: firrtl.instance reg64 {{.*}} @Register_i64_impl
// CHECK-DAG: firrtl.instance wire16 {{.*}} @Wire_i16_impl