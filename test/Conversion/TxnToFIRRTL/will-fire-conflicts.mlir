// RUN: sharp-opt --convert-txn-to-firrtl %s | FileCheck %s

// Test will-fire logic with conflict matrix

// CHECK-LABEL: firrtl.circuit "ConflictTest"
// CHECK: firrtl.module @ConflictTest

txn.module @ConflictTest {
  %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
  
  txn.action_method @a1() {
    %c1 = arith.constant 1 : i32
    txn.call @reg::@write(%c1) : (i32) -> ()
    txn.return
  }
  
  txn.action_method @a2() {
    %c2 = arith.constant 2 : i32
    txn.call @reg::@write(%c2) : (i32) -> ()
    txn.return
  }
  
  txn.action_method @a3() {
    %c3 = arith.constant 3 : i32
    txn.call @reg::@write(%c3) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@a1, @a2, @a3] {
    conflict_matrix = {
      "a1,a1" = 2 : i32,  // C (conflict)
      "a2,a2" = 2 : i32,  // C
      "a3,a3" = 2 : i32,  // C
      "a1,a2" = 0 : i32,  // SB (sequence before)
      "a2,a3" = 2 : i32,  // C
      "a1,a3" = 3 : i32   // CF (conflict free)
    }
  }
}

// Check will-fire generation
// CHECK: %a1_wf = firrtl.node
// CHECK-DAG: firrtl.not %a1_wf
// CHECK: %a2_wf = firrtl.node
// CHECK-DAG: firrtl.not %a2_wf
// CHECK: %a3_wf = firrtl.node

// Check ready signals are generated
// CHECK-DAG: firrtl.connect %a1RDY
// CHECK-DAG: firrtl.connect %a2RDY
// CHECK-DAG: firrtl.connect %a3RDY

// Check when blocks for actions
// CHECK-DAG: firrtl.when %a1_wf
// CHECK-DAG: firrtl.when %a2_wf
// CHECK-DAG: firrtl.when %a3_wf