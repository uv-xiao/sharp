// RUN: sharp-opt %s | FileCheck %s

// Test conflict matrix on schedule operation with enum values

// CHECK-LABEL: txn.module @ConflictExample
txn.module @ConflictExample {
  txn.rule @r1 {
    txn.return
  }
  
  txn.rule @r2 {
    txn.return
  }
  
  txn.action_method @m1() {
    txn.return
  }
  
  // CHECK: txn.schedule [@r1, @r2, @m1]
  // The conflict matrix uses enum values:
  // 0 = SequenceBefore (SB)
  // 1 = SequenceAfter (SA) 
  // 2 = Conflict (C)
  // 3 = ConflictFree (CF) - default if not specified
  txn.schedule [@r1, @r2, @m1] {
    conflict_matrix = {
      "r1,r2" = 2 : i32,   // r1 conflicts with r2
      "r1,m1" = 0 : i32,   // r1 must execute before m1
      "r2,m1" = 1 : i32    // r2 must execute after m1
    }
  }
}