// RUN: sharp-opt --convert-txn-to-firrtl %s | FileCheck %s

// Test complex conflict matrix with multiple relationships
txn.module @ComplexConflicts {
  // Create instances to demonstrate conflicts
  %reg1 = txn.instance @reg1 of @Register<i32> : index
  %reg2 = txn.instance @reg2 of @Register<i32> : index
  
  // Four actions with different conflict relationships
  txn.action_method @a1() {
    %c1 = arith.constant 1 : i32
    txn.call @reg1::@write(%c1) : (i32) -> ()
    txn.return
  }
  
  txn.action_method @a2() {
    %c2 = arith.constant 2 : i32
    txn.call @reg2::@write(%c2) : (i32) -> ()
    txn.return
  }
  
  txn.action_method @a3() {
    %val = txn.call @reg1::@read() : () -> i32
    txn.return
  }
  
  txn.rule @r1 {
    %val = txn.call @reg1::@read() : () -> i32
    %c10 = arith.constant 10 : i32
    %sum = arith.addi %val, %c10 : i32
    txn.call @reg1::@write(%sum) : (i32) -> ()
    txn.return
  }
  
  txn.rule @r2 {
    %c20 = arith.constant 20 : i32
    txn.call @reg2::@write(%c20) : (i32) -> ()
    txn.return
  }
  
  txn.rule @r3 {
    %val1 = txn.call @reg1::@read() : () -> i32
    %val2 = txn.call @reg2::@read() : () -> i32
    %sum = arith.addi %val1, %val2 : i32
    txn.call @reg1::@write(%sum) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@a1, @a2, @a3, @r1, @r2, @r3] {
    conflict_matrix = {
      // Self conflicts
      "a1,a1" = 2 : i32,
      "a2,a2" = 2 : i32,
      "a3,a3" = 2 : i32,
      "r1,r1" = 2 : i32,
      "r2,r2" = 2 : i32,
      "r3,r3" = 2 : i32,
      
      // Sequential relationships
      "a1,a2" = 0 : i32,  // a1 before a2
      "a2,a3" = 1 : i32,  // a2 after a3 (a3 before a2)
      
      // Conflicts
      "a1,a3" = 2 : i32,  // a1 conflicts with a3
      "r1,r2" = 2 : i32,  // r1 conflicts with r2
      
      // Conflict-free
      "r1,r3" = 3 : i32,  // r1 conflict-free with r3
      "r2,r3" = 3 : i32,  // r2 conflict-free with r3
      
      // Rules calling methods inherit conflicts
      "r1,a1" = 3 : i32,  // r1 calls a1, so CF
      "r2,a2" = 3 : i32,  // r2 calls a2, so CF
      "r3,a3" = 3 : i32,  // r3 calls a3, so CF
      
      // Derived conflicts
      "r1,a2" = 0 : i32,  // r1->a1, a1<a2, so r1<a2
      "r1,a3" = 2 : i32,  // r1->a1, a1 C a3, so r1 C a3
      "r2,a3" = 1 : i32,  // r2->a2, a2>a3, so r2>a3
      "r2,a1" = 1 : i32,  // r2->a2, a2>a1, so r2>a1
      "r3,a1" = 2 : i32,  // r3->a3, a3 C a1, so r3 C a1
      "r3,a2" = 0 : i32   // r3->a3, a3<a2, so r3<a2
    }
  }
}

// CHECK: module {
// CHECK-NEXT: firrtl.circuit "ComplexConflicts" {
// CHECK: firrtl.module @ComplexConflicts

// Check will-fire signals exist
// CHECK: %a1_wf = firrtl.node
// CHECK: %a2_wf = firrtl.node
// CHECK: %a3_wf = firrtl.node
// CHECK: %r1_wf = firrtl.node
// CHECK: %r2_wf = firrtl.node
// CHECK: %r3_wf = firrtl.node

// Check ready signals are connected
// CHECK-DAG: firrtl.connect %a1RDY
// CHECK-DAG: firrtl.connect %a2RDY
// CHECK-DAG: firrtl.connect %a3RDY

// Check when blocks exist for all actions
// CHECK-DAG: firrtl.when %a1_wf
// CHECK-DAG: firrtl.when %a2_wf
// CHECK-DAG: firrtl.when %a3_wf
// CHECK-DAG: firrtl.when %r1_wf
// CHECK-DAG: firrtl.when %r2_wf
// CHECK-DAG: firrtl.when %r3_wf