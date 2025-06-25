// RUN: sharp-opt %s | sharp-opt | FileCheck %s

// CHECK-LABEL: func.func @test_constant
func.func @test_constant() -> i32 {
  // CHECK: sharp.core.constant 42 : i32
  %0 = sharp.core.constant 42 : i32
  return %0 : i32
}