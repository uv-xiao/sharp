// RUN: sharp-opt --convert-txn-to-firrtl %s | FileCheck %s

// Test will-fire logic with conflict matrix
txn.module @ConflictTest {
  txn.action_method @a1() {
    txn.return
  }
  
  txn.action_method @a2() {
    txn.return
  }
  
  txn.action_method @a3() {
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

// CHECK-LABEL: firrtl.circuit "ConflictTest"
// CHECK: firrtl.module @ConflictTest

// Check will-fire generation
// CHECK: %a1_wf = firrtl.node %a1EN
// CHECK: firrtl.not %a1_wf
// CHECK: %a2_wf = firrtl.node
// CHECK: firrtl.not %a2_wf
// CHECK: %a3_wf = firrtl.node

// Check ready signals are generated
// CHECK: firrtl.connect %a1RDY
// CHECK: firrtl.connect %a2RDY
// CHECK: firrtl.connect %a3RDY

// Check when blocks for actions
// CHECK: firrtl.when %a1_wf
// CHECK: firrtl.when %a2_wf
// CHECK: firrtl.when %a3_wf