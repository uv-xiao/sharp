// RUN: sharp-opt %s -sharp-action-scheduling | FileCheck %s

// Test heuristic algorithm with >10 actions
txn.module @LargeModule {
  txn.rule @r1 { txn.return }
  txn.rule @r2 { txn.return }
  txn.rule @r3 { txn.return }
  txn.rule @r4 { txn.return }
  txn.rule @r5 { txn.return }
  
  txn.action_method @m1() { txn.return }
  txn.action_method @m2() { txn.return }
  txn.action_method @m3() { txn.return }
  txn.action_method @m4() { txn.return }
  txn.action_method @m5() { txn.return }
  txn.action_method @m6() { txn.return }
  
  // Partial schedule with some constraints
  txn.schedule [@r1, @m1] {
    conflict_matrix = {
      "r1,r2" = 0 : i32,  // r1 SB r2
      "r2,r3" = 0 : i32,  // r2 SB r3
      "m1,m2" = 0 : i32,  // m1 SB m2
      "m2,m3" = 0 : i32,  // m2 SB m3
      "r5,m6" = 1 : i32   // r5 SA m6 (m6 before r5)
    }
  }
}
// CHECK-LABEL: txn.module @LargeModule
// CHECK: txn.schedule [@r1, @m1, {{.*}}] {conflict_matrix =
// Note: partial schedule [@r1, @m1] is preserved

// Test heuristic with complex conflict patterns
txn.module @ComplexConflictPattern {
  txn.rule @a { txn.return }
  txn.rule @b { txn.return }
  txn.rule @c { txn.return }
  txn.rule @d { txn.return }
  txn.rule @e { txn.return }
  txn.rule @f { txn.return }
  txn.rule @g { txn.return }
  txn.rule @h { txn.return }
  txn.rule @i { txn.return }
  txn.rule @j { txn.return }
  txn.rule @k { txn.return }
  
  txn.schedule [] {
    conflict_matrix = {
      // Create a chain of dependencies
      "a,b" = 0 : i32,  // a SB b
      "b,c" = 0 : i32,  // b SB c
      "c,d" = 0 : i32,  // c SB d
      "d,e" = 0 : i32,  // d SB e
      // Create some conflicts
      "f,g" = 2 : i32,  // f C g
      "h,i" = 2 : i32,  // h C i
      // Create some SA relationships
      "j,k" = 1 : i32,  // j SA k (k before j)
      "k,a" = 0 : i32   // k SB a
    }
  }
}
// CHECK-LABEL: txn.module @ComplexConflictPattern
// CHECK: txn.schedule [{{.*}}@k, {{.*}}@a, @b, @c, @d, @e, {{.*}}@j{{.*}}] {conflict_matrix =
// Note: k before a, a-b-c-d-e chain preserved, j after k