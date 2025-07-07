// RUN: sharp-opt %s -sharp-action-scheduling | FileCheck %s

// Test 1: Complete missing schedule with no conflicts
txn.module @NoConflicts {
  txn.rule @r1 {
    txn.return
  }
  
  txn.rule @r2 {
    txn.return
  }
  
  txn.action_method @m1() {
    txn.return
  }
  
  txn.schedule [] {}
}
// CHECK-LABEL: txn.module @NoConflicts
// CHECK: txn.schedule [@r1, @r2, @m1]

// Test 2: Preserve partial schedule order
txn.module @PartialSchedule {
  txn.rule @r1 {
    txn.return
  }
  
  txn.rule @r2 {
    txn.return
  }
  
  txn.action_method @m1() {
    txn.return
  }
  
  txn.action_method @m2() {
    txn.return
  }
  
  // Partial schedule: r1 before m1
  txn.schedule [@r1, @m1] {
    conflict_matrix = {}
  }
}
// CHECK-LABEL: txn.module @PartialSchedule
// CHECK: txn.schedule [@r1, @r2, @m1, @m2]

// Test 3: Handle conflicts (C)
txn.module @WithConflicts {
  txn.rule @r1 {
    txn.return
  }
  
  txn.rule @r2 {
    txn.return
  }
  
  txn.action_method @m1() {
    txn.return
  }
  
  txn.action_method @m2() {
    txn.return
  }
  
  txn.schedule [] {
    conflict_matrix = {
      "r1,r2" = 2 : i32,  // r1 C r2
      "m1,m2" = 2 : i32   // m1 C m2
    }
  }
}
// CHECK-LABEL: txn.module @WithConflicts
// CHECK: txn.schedule [@r1, @r2, @m1, @m2] {conflict_matrix = {"m1,m2" = 2 : i32, "r1,r2" = 2 : i32}}

// Test 4: Conflict-free (CF) relationships
txn.module @ConflictFree {
  txn.rule @r1 {
    txn.return
  }
  
  txn.rule @r2 {
    txn.return
  }
  
  txn.action_method @m1() {
    txn.return
  }
  
  txn.schedule [] {
    conflict_matrix = {
      "r1,r2" = 3 : i32,  // r1 CF r2
      "r2,m1" = 3 : i32   // r2 CF m1
    }
  }
}
// CHECK-LABEL: txn.module @ConflictFree
// CHECK: txn.schedule [@r1, @r2, @m1] {conflict_matrix = {"r1,r2" = 3 : i32, "r2,m1" = 3 : i32}}

// Test 5: Module with no actions
txn.module @NoActions {
  %reg = txn.instance @reg of @Register : !txn.module<"Register">
  txn.schedule [] {}
}
// CHECK-LABEL: txn.module @NoActions
// CHECK: txn.schedule []

// Test 6: Single action module
txn.module @SingleAction {
  txn.rule @r1 {
    txn.return
  }
  txn.schedule [] {}
}
// CHECK-LABEL: txn.module @SingleAction
// CHECK: txn.schedule [@r1]