// RUN: sharp-opt %s -convert-txn-to-func | FileCheck %s

// Test rule conversion

// CHECK-LABEL: module
// CHECK: func.func @RuleModule_main() {
// CHECK:   return
// CHECK: }
// CHECK: func.func @RuleModule_rule_alwaysFire() {
// CHECK:   %[[TRUE:.*]] = arith.constant true
// CHECK:   txn.yield %[[TRUE]] : i1
// CHECK: }

// CHECK: func.func @RuleModule_rule_conditionalFire() {
// CHECK:   %[[C10:.*]] = arith.constant 10 : i32
// CHECK:   %[[C5:.*]] = arith.constant 5 : i32
// CHECK:   %[[CMP:.*]] = arith.cmpi sgt, %[[C10]], %[[C5]]
// CHECK:   txn.yield %[[CMP]] : i1
// CHECK: }

txn.module @RuleModule {
  txn.rule @alwaysFire {
    %true = arith.constant true
    txn.yield %true : i1
  }
  
  txn.rule @conditionalFire {
    %c10 = arith.constant 10 : i32
    %c5 = arith.constant 5 : i32
    %cond = arith.cmpi sgt, %c10, %c5 : i32
    txn.yield %cond : i1
  }
  
  txn.schedule [@alwaysFire, @conditionalFire]
}