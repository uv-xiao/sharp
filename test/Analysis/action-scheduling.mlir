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
// Note: r1 comes before m1 as required by partial schedule

// Test 3: Handle SB (Sequential Before) constraints
txn.module @SBConstraints {
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
      "r1,r2" = 0 : i32,  // r1 SB r2
      "r2,m1" = 0 : i32   // r2 SB m1
    }
  }
}
// CHECK-LABEL: txn.module @SBConstraints
// CHECK: txn.schedule [@r1, @r2, @m1] {conflict_matrix = {"r1,r2" = 0 : i32, "r2,m1" = 0 : i32}}

// Test 4: Handle SA (Sequential After) constraints
txn.module @SAConstraints {
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
      "r1,r2" = 1 : i32,  // r1 SA r2 (r2 before r1)
      "r1,m1" = 1 : i32   // r1 SA m1 (m1 before r1)
    }
  }
}
// CHECK-LABEL: txn.module @SAConstraints
// CHECK: txn.schedule [@r2, @m1, @r1] {conflict_matrix = {"r1,m1" = 1 : i32, "r1,r2" = 1 : i32}}

// Test 5: Handle conflicts (C) - minimize but don't prevent scheduling
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

// Test 6: Complex constraints with partial schedule
txn.module @ComplexConstraints {
  txn.rule @r1 {
    txn.return
  }
  
  txn.rule @r2 {
    txn.return
  }
  
  txn.rule @r3 {
    txn.return
  }
  
  txn.action_method @m1() {
    txn.return
  }
  
  txn.action_method @m2() {
    txn.return
  }
  
  // Partial schedule: r2 before m1
  txn.schedule [@r2, @m1] {
    conflict_matrix = {
      "r1,r2" = 0 : i32,  // r1 SB r2
      "r3,m2" = 0 : i32,  // r3 SB m2
      "m1,m2" = 2 : i32   // m1 C m2
    }
  }
}
// CHECK-LABEL: txn.module @ComplexConstraints
// CHECK: txn.schedule [@r1, @r2, @r3, @m1, @m2] {conflict_matrix = {"m1,m2" = 2 : i32, "r1,r2" = 0 : i32, "r3,m2" = 0 : i32}}

// Test 7: Complete schedule should not be modified
txn.module @CompleteSchedule {
  txn.rule @r1 {
    txn.return
  }
  
  txn.rule @r2 {
    txn.return
  }
  
  txn.action_method @m1() {
    txn.return
  }
  
  // Already complete schedule
  txn.schedule [@m1, @r2, @r1] {
    conflict_matrix = {
      "r1,r2" = 0 : i32  // r1 SB r2 (violated in schedule)
    }
  }
}
// CHECK-LABEL: txn.module @CompleteSchedule
// CHECK: txn.schedule [@m1, @r2, @r1] {conflict_matrix = {"r1,r2" = 0 : i32}}

// Test 8: Module with no actions
txn.module @NoActions {
  %reg = txn.instance @reg of @Register : !txn.module<"Register">
  txn.schedule [] {}
}
// CHECK-LABEL: txn.module @NoActions
// CHECK: txn.schedule []

// Test 9: Single action module
txn.module @SingleAction {
  txn.rule @r1 {
    txn.return
  }
  txn.schedule [] {}
}
// CHECK-LABEL: txn.module @SingleAction
// CHECK: txn.schedule [@r1]

// Test 10: Conflict-free (CF) relationships
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