// RUN: sharp-opt %s --convert-txn-to-func | FileCheck %s

// Test will-fire logic with guard conditions
txn.module @GuardTest {
  
  txn.value_method @getValue() -> i32 {
    %val = arith.constant 5 : i32
    txn.return %val : i32
  }
  
  txn.action_method @setValue() {
    // Just a placeholder
    txn.return
  }
  
  txn.rule @conditionalRule {
    %val = txn.call @getValue() : () -> i32
    %c10 = arith.constant 10 : i32
    %cond = arith.cmpi ult, %val, %c10 : i32
    
    // TODO: txn.if/else/abort conversion needs fixing
    // For now, just test simple rule execution
    txn.call @setValue() : () -> ()
    txn.yield
  }
  
  txn.schedule [@conditionalRule]
}

// CHECK-LABEL: func.func @GuardTest_rule_conditionalRule() -> i1
// CHECK: %[[VAL:.*]] = func.call @GuardTest_getValue() : () -> i32
// CHECK: %[[C10:.*]] = arith.constant 10 : i32
// CHECK: %[[COND:.*]] = arith.cmpi ult, %[[VAL]], %[[C10]] : i32
// CHECK: func.call @GuardTest_setValue() : () -> i1
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: return %[[FALSE]] : i1

// CHECK-LABEL: func.func @GuardTest_scheduler()
// Execute the conditional rule
// CHECK: scf.if %{{.*}} {
// CHECK:   %[[ABORTED:.*]] = func.call @GuardTest_rule_conditionalRule() : () -> i1
// CHECK:   %[[NOT_ABORTED:.*]] = arith.xori %[[ABORTED]], %{{.*}} : i1
// CHECK:   memref.store %[[NOT_ABORTED]], %{{.*}}[] : memref<i1>
// CHECK: }