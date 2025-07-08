// RUN: sharp-opt %s --convert-txn-to-func --mlir-print-ir-after-all 2>&1 | FileCheck %s

// Test abort conversion specifically
txn.module @AbortTest {
  txn.rule @rule1 {
    txn.abort
  }
  
  txn.schedule [@rule1]
}

// CHECK-LABEL: func.func @AbortTest_rule_rule1() -> i1
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: return %[[TRUE]] : i1