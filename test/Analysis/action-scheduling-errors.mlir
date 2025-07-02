// RUN: not sharp-opt %s -sharp-action-scheduling 2>&1 | FileCheck %s

// Test cyclic dependencies - should fail
txn.module @CyclicDependencies {
  txn.rule @r1 {
    txn.return
  }
  
  txn.rule @r2 {
    txn.return
  }
  
  txn.rule @r3 {
    txn.return
  }
  
  txn.schedule [] {
    conflict_matrix = {
      "r1,r2" = 1 : i32,  // r1 SA r2 (r2 before r1)
      "r2,r3" = 1 : i32,  // r2 SA r3 (r3 before r2)
      "r3,r1" = 1 : i32   // r3 SA r1 (r1 before r3) - creates cycle
    }
  }
}
// CHECK: error: Failed to compute valid schedule - possible cyclic dependencies