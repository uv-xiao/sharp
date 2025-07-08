// RUN: sharp-opt %s | sharp-opt | FileCheck %s

// CHECK-LABEL: txn.module @FunctionExample
txn.module @FunctionExample {
  // CHECK: txn.func @add(%arg0: i32, %arg1: i32) -> i32
  txn.func @add(%a: i32, %b: i32) -> i32 {
    %sum = arith.addi %a, %b : i32
    txn.return %sum : i32
  }
  
  // CHECK: txn.func @multiply(%arg0: i32, %arg1: i32) -> i32
  txn.func @multiply(%x: i32, %y: i32) -> i32 {
    %prod = arith.muli %x, %y : i32
    txn.return %prod : i32
  }
  
  // CHECK: txn.func @complexCalc(%arg0: i32, %arg1: i32, %arg2: i32) -> i32
  txn.func @complexCalc(%a: i32, %b: i32, %c: i32) -> i32 {
    // Nested function calls
    %sum = txn.func_call @add(%a, %b) : (i32, i32) -> i32
    %result = txn.func_call @multiply(%sum, %c) : (i32, i32) -> i32
    txn.return %result : i32
  }
  
  // CHECK: txn.value_method @compute() -> i32
  txn.value_method @compute() -> i32 {
    %c5 = arith.constant 5 : i32
    %c3 = arith.constant 3 : i32
    %c2 = arith.constant 2 : i32
    
    // CHECK: %{{.*}} = txn.func_call @add(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
    %sum = txn.func_call @add(%c5, %c3) : (i32, i32) -> i32
    
    // CHECK: %{{.*}} = txn.func_call @multiply(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
    %prod = txn.func_call @multiply(%sum, %c2) : (i32, i32) -> i32
    
    // CHECK: %{{.*}} = txn.func_call @complexCalc(%{{.*}}, %{{.*}}, %{{.*}}) : (i32, i32, i32) -> i32
    %complex = txn.func_call @complexCalc(%c5, %c3, %c2) : (i32, i32, i32) -> i32
    
    %total = arith.addi %prod, %complex : i32
    txn.return %total : i32
  }
  
  // Functions with no return value
  // CHECK: txn.func @printValue(%arg0: i32)
  txn.func @printValue(%val: i32) {
    // In real hardware, this would be some side-effect
    txn.return
  }
  
  // CHECK: txn.action_method @doAction()
  txn.action_method @doAction() {
    %c42 = arith.constant 42 : i32
    // CHECK: txn.func_call @printValue(%{{.*}}) : (i32) -> ()
    txn.func_call @printValue(%c42) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@doAction]
}

// CHECK-LABEL: txn.module @ConditionalFunctions
txn.module @ConditionalFunctions {
  // CHECK: txn.func @max(%arg0: i32, %arg1: i32) -> i32
  txn.func @max(%a: i32, %b: i32) -> i32 {
    %cmp = arith.cmpi sgt, %a, %b : i32
    %result = arith.select %cmp, %a, %b : i32
    txn.return %result : i32
  }
  
  // CHECK: txn.func @abs(%arg0: i32) -> i32
  txn.func @abs(%x: i32) -> i32 {
    %zero = arith.constant 0 : i32
    %neg = arith.subi %zero, %x : i32
    %isNeg = arith.cmpi slt, %x, %zero : i32
    %result = arith.select %isNeg, %neg, %x : i32
    txn.return %result : i32
  }
  
  // CHECK: txn.value_method @getMaxAbs(%arg0: i32, %arg1: i32) -> i32
  txn.value_method @getMaxAbs(%a: i32, %b: i32) -> i32 {
    // CHECK: %{{.*}} = txn.func_call @abs(%{{.*}}) : (i32) -> i32
    %abs_a = txn.func_call @abs(%a) : (i32) -> i32
    // CHECK: %{{.*}} = txn.func_call @abs(%{{.*}}) : (i32) -> i32
    %abs_b = txn.func_call @abs(%b) : (i32) -> i32
    // CHECK: %{{.*}} = txn.func_call @max(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
    %result = txn.func_call @max(%abs_a, %abs_b) : (i32, i32) -> i32
    txn.return %result : i32
  }
  
  txn.schedule []
}