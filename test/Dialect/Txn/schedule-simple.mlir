// RUN: sharp-opt %s -allow-unregistered-dialect | sharp-opt -allow-unregistered-dialect | FileCheck %s

// Test schedule operation with method/rule lists

// CHECK-LABEL: txn.module @Simple
txn.module @Simple {
  txn.value_method @getValue() -> i32 {
    %c0 = arith.constant 0 : i32
    txn.return %c0 : i32
  }
  
  txn.action_method @setValue(%val: i32) {
    txn.return
  }
  
  // CHECK: txn.schedule [@setValue]
  txn.schedule [@setValue]
}

// CHECK-LABEL: txn.module @WithRule
txn.module @WithRule {
  txn.rule @autoProcess {
    %c1 = arith.constant 1 : i32
  }
  
  txn.action_method @process() {
    txn.return
  }
  
  // CHECK: txn.schedule [@autoProcess, @process]
  txn.schedule [@autoProcess, @process]
}

// CHECK-LABEL: txn.module @EmptySchedule
txn.module @EmptySchedule {
  // CHECK: txn.schedule []
  txn.schedule []
}

// CHECK-LABEL: txn.module @SingleMethod
txn.module @SingleMethod {
  txn.action_method @onlyMethod() -> i1 {
    %true = arith.constant true
    txn.return %true : i1
  }
  
  // CHECK: txn.schedule [@onlyMethod]
  txn.schedule [@onlyMethod]
}