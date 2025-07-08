// RUN: sharp-opt %s -sharp-concurrent-sim 2>&1 | FileCheck %s

// Test basic concurrent simulation pass

// CHECK: remark: Generated concurrent simulation code using DAM methodology

txn.module @Producer {
  txn.value_method @produce() -> i32 {
    %val = arith.constant 42 : i32
    txn.return %val : i32
  }
  
  txn.schedule []
}

txn.module @Consumer {
  txn.action_method @consume(%val: i32) {
    txn.return
  }
  
  txn.schedule [@consume]
}