// RUN: sharp-opt %s --sharp-inline-functions | FileCheck %s

// CHECK-LABEL: txn.module @InliningExample
txn.module @InliningExample {
  // Functions should be removed after inlining
  // CHECK-NOT: txn.func
  txn.func @add(%a: i32, %b: i32) -> i32 {
    %sum = arith.addi %a, %b : i32
    txn.return %sum : i32
  }
  
  txn.func @multiply(%x: i32, %y: i32) -> i32 {
    %prod = arith.muli %x, %y : i32
    txn.return %prod : i32
  }
  
  // CHECK: txn.value_method @compute() -> i32
  txn.value_method @compute() -> i32 {
    %c5 = arith.constant 5 : i32
    %c3 = arith.constant 3 : i32
    
    // CHECK-NOT: txn.func_call
    // CHECK: %[[SUM:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
    %sum = txn.func_call @add(%c5, %c3) : (i32, i32) -> i32
    
    // CHECK: %[[PROD:.*]] = arith.muli %[[SUM]], %{{.*}} : i32
    %prod = txn.func_call @multiply(%sum, %c3) : (i32, i32) -> i32
    
    // CHECK: txn.return %[[PROD]] : i32
    txn.return %prod : i32
  }
  
  txn.schedule []
}

// CHECK-LABEL: txn.module @NestedInlining
txn.module @NestedInlining {
  // CHECK-NOT: txn.func
  txn.func @double(%x: i32) -> i32 {
    %two = arith.constant 2 : i32
    %result = arith.muli %x, %two : i32
    txn.return %result : i32
  }
  
  txn.func @quadruple(%x: i32) -> i32 {
    // Nested function call
    %doubled = txn.func_call @double(%x) : (i32) -> i32
    %result = txn.func_call @double(%doubled) : (i32) -> i32
    txn.return %result : i32
  }
  
  // CHECK: txn.value_method @compute() -> i32
  txn.value_method @compute() -> i32 {
    %c10 = arith.constant 10 : i32
    
    // CHECK-NOT: txn.func_call
    // Should expand to: double(double(10)) = 2 * (2 * 10) = 40
    // CHECK: %[[TWO1:.*]] = arith.constant 2 : i32
    // CHECK: %[[MUL1:.*]] = arith.muli %{{.*}}, %[[TWO1]] : i32
    // CHECK: %[[TWO2:.*]] = arith.constant 2 : i32
    // CHECK: %[[MUL2:.*]] = arith.muli %[[MUL1]], %[[TWO2]] : i32
    %result = txn.func_call @quadruple(%c10) : (i32) -> i32
    
    // CHECK: txn.return %[[MUL2]] : i32
    txn.return %result : i32
  }
  
  txn.schedule []
}

// CHECK-LABEL: txn.module @ConditionalInlining
txn.module @ConditionalInlining {
  // CHECK-NOT: txn.func
  txn.func @abs(%x: i32) -> i32 {
    %zero = arith.constant 0 : i32
    %neg = arith.subi %zero, %x : i32
    %isNeg = arith.cmpi slt, %x, %zero : i32
    %result = arith.select %isNeg, %neg, %x : i32
    txn.return %result : i32
  }
  
  // CHECK: txn.value_method @getAbs(%arg0: i32) -> i32
  txn.value_method @getAbs(%x: i32) -> i32 {
    // CHECK-NOT: txn.func_call
    // CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
    // CHECK: %[[NEG:.*]] = arith.subi %[[ZERO]], %arg0 : i32
    // CHECK: %[[CMP:.*]] = arith.cmpi slt, %arg0, %[[ZERO]] : i32
    // CHECK: %[[SELECT:.*]] = arith.select %[[CMP]], %[[NEG]], %arg0 : i32
    %result = txn.func_call @abs(%x) : (i32) -> i32
    
    // CHECK: txn.return %[[SELECT]] : i32
    txn.return %result : i32
  }
  
  txn.schedule []
}

// CHECK-LABEL: txn.module @MultiReturnInlining
txn.module @MultiReturnInlining {
  // CHECK-NOT: txn.func
  txn.func @divmod(%a: i32, %b: i32) -> (i32, i32) {
    %quot = arith.divsi %a, %b : i32
    %rem = arith.remsi %a, %b : i32
    txn.return %quot, %rem : i32, i32
  }
  
  // CHECK: txn.value_method @compute() -> (i32, i32)
  txn.value_method @compute() -> (i32, i32) {
    %c17 = arith.constant 17 : i32
    %c5 = arith.constant 5 : i32
    
    // CHECK-NOT: txn.func_call
    // CHECK: %[[QUOT:.*]] = arith.divsi %{{.*}}, %{{.*}} : i32
    // CHECK: %[[REM:.*]] = arith.remsi %{{.*}}, %{{.*}} : i32
    %quot, %rem = txn.func_call @divmod(%c17, %c5) : (i32, i32) -> (i32, i32)
    
    // CHECK: txn.return %[[QUOT]], %[[REM]] : i32, i32
    txn.return %quot, %rem : i32, i32
  }
  
  txn.schedule []
}

// CHECK-LABEL: txn.module @VoidFunctionInlining
txn.module @VoidFunctionInlining {
  
  // CHECK-NOT: txn.func
  txn.func @doNothing() {
    txn.return
  }
  
  // CHECK: txn.action_method @doAction()
  txn.action_method @doAction() {
    // CHECK-NOT: txn.func_call
    txn.func_call @doNothing() : () -> ()
    
    // CHECK: txn.return
    txn.return
  }
  
  txn.schedule [@doAction]
}