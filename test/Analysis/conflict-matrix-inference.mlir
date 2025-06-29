// RUN: sharp-opt %s -sharp-infer-conflict-matrix | FileCheck %s

// CHECK-LABEL: txn.module @TestConflictInference
txn.module @TestConflictInference {
  // Rule that should conflict with itself
  txn.rule @rule1 {
    txn.yield
  }
  
  // Rule that should conflict with itself  
  txn.rule @rule2 {
    txn.yield
  }
  
  // Schedule with empty conflict matrix - should infer conflicts
  // CHECK: txn.schedule [@rule1, @rule2] {
  // CHECK: conflict_matrix = {
  // CHECK-DAG: "rule1,rule1" = 2 : i32
  // CHECK-DAG: "rule1,rule2" = 3 : i32
  // CHECK-DAG: "rule2,rule1" = 3 : i32
  // CHECK-DAG: "rule2,rule2" = 2 : i32
  // CHECK: }
  // CHECK: }
  txn.schedule [@rule1, @rule2] {
    conflict_matrix = {}
  }
}

// CHECK-LABEL: txn.module @TestSequenceConflict
txn.module @TestSequenceConflict {
  txn.rule @ruleA {
    txn.yield
  }
  
  txn.rule @ruleB {
    txn.yield
  }
  
  // Test that SA + SB = C (rule 2 in inference)
  // Initial: ruleA SA ruleB (1) and ruleB SB ruleA (0)
  // Should infer: both conflict with each other
  // CHECK: txn.schedule [@ruleA, @ruleB] {
  // CHECK: conflict_matrix = {
  // CHECK-DAG: "ruleA,ruleA" = 2 : i32
  // CHECK-DAG: "ruleA,ruleB" = 2 : i32
  // CHECK-DAG: "ruleB,ruleA" = 2 : i32
  // CHECK-DAG: "ruleB,ruleB" = 2 : i32
  // CHECK: }
  // CHECK: }
  txn.schedule [@ruleA, @ruleB] {
    conflict_matrix = {
      "ruleA,ruleB" = 1 : i32,   // SA
      "ruleB,ruleA" = 0 : i32    // SB
    }
  }
}