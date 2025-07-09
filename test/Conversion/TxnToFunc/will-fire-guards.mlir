// RUN: sharp-opt %s --convert-txn-to-func | FileCheck %s

// Test will-fire logic with guard conditions
txn.module @GuardTest {
  %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
  
  txn.value_method @getValue() -> i32 {
    %val = txn.call @reg::@read() : () -> i32
    txn.return %val : i32
  }
  
  txn.rule @conditionalRule {
    %val = txn.call @getValue() : () -> i32
    %c10 = arith.constant 10 : i32
    %cond = arith.cmpi ult, %val, %c10 : i32
    
    txn.if %cond {
      %c20 = arith.constant 20 : i32
      txn.call @reg::@write(%c20) : (i32) -> ()
    } else {
      txn.abort
    }
    txn.yield
  }
  
  txn.schedule [@conditionalRule]
}

// CHECK-LABEL: func.func @GuardTest_rule_conditionalRule() -> i1
// CHECK: %[[VAL:.*]] = func.call @GuardTest_getValue() : () -> i32
// CHECK: %[[C10:.*]] = arith.constant 10 : i32
// CHECK: %[[COND:.*]] = arith.cmpi ult, %[[VAL]], %[[C10]] : i32
// CHECK: scf.if %[[COND]] {
// CHECK:   %[[C20:.*]] = arith.constant 20 : i32
// CHECK:   func.call @GuardTest_reg_write(%[[C20]]) : (i32) -> ()
// CHECK: } else {
// CHECK:   %[[TRUE:.*]] = arith.constant true
// CHECK:   return %[[TRUE]] : i1
// CHECK: }
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: return %[[FALSE]] : i1

// CHECK-LABEL: func.func @GuardTest_scheduler()
// Execute the conditional rule
// CHECK: scf.if %{{.*}} {
// CHECK:   %[[ABORTED:.*]] = func.call @GuardTest_rule_conditionalRule() : () -> i1
// CHECK:   %[[NOT_ABORTED:.*]] = arith.xori %[[ABORTED]], %{{.*}} : i1
// CHECK:   memref.store %[[NOT_ABORTED]], %{{.*}}[] : memref<i1>
// CHECK: }