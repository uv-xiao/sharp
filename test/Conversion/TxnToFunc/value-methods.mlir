// RUN: sharp-opt %s -convert-txn-to-func | FileCheck %s

// Test value method conversion with various signatures

// CHECK-LABEL: module
// CHECK: func.func @Math_add(%arg0: i32, %arg1: i32) -> i32 {
// CHECK:   %[[SUM:.*]] = arith.addi %arg0, %arg1
// CHECK:   return %[[SUM]] : i32
// CHECK: }

// CHECK: func.func @Math_isPositive(%arg0: i32) -> i1 {
// CHECK:   %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:   %[[CMP:.*]] = arith.cmpi sgt, %arg0, %[[ZERO]]
// CHECK:   return %[[CMP]] : i1
// CHECK: }

// CHECK: func.func @Math_getConstant() -> i32 {
// CHECK:   %[[C42:.*]] = arith.constant 42 : i32
// CHECK:   return %[[C42]] : i32
// CHECK: }

txn.module @Math {
  txn.value_method @add(%a: i32, %b: i32) -> i32 {
    %sum = arith.addi %a, %b : i32
    txn.return %sum : i32
  }
  
  txn.value_method @isPositive(%x: i32) -> i1 {
    %zero = arith.constant 0 : i32
    %cmp = arith.cmpi sgt, %x, %zero : i32
    txn.return %cmp : i1
  }
  
  txn.value_method @getConstant() -> i32 {
    %c42 = arith.constant 42 : i32
    txn.return %c42 : i32
  }
  
  txn.schedule [@add, @isPositive, @getConstant]
}