// RUN: sharp-opt %s -convert-txn-to-firrtl | FileCheck %s

// Test that empty action bodies don't generate empty when blocks

txn.module @EmptyActions {
  // Empty action method
  txn.action_method @emptyAction() {
    txn.return
  }
  
  // Action method with only abort
  txn.action_method @abortOnlyAction() {
    txn.abort
  }
  
  // Empty rule
  txn.rule @emptyRule {
    txn.return
  }
  
  // Non-empty action method for comparison
  txn.action_method @nonEmptyAction(%val: i32) {
    %c1 = arith.constant 1 : i32
    %result = arith.addi %val, %c1 : i32
    txn.return
  }
  
  txn.schedule [@emptyAction, @abortOnlyAction, @emptyRule, @nonEmptyAction]
}

// CHECK-LABEL: firrtl.circuit "EmptyActions"
// CHECK: firrtl.module @EmptyActions

// Check that will-fire signals are still created
// CHECK: %emptyAction_wf = firrtl.node
// CHECK: %abortOnlyAction_wf = firrtl.node
// CHECK: %emptyRule_wf = firrtl.node
// CHECK: %nonEmptyAction_wf = firrtl.node

// Check that empty actions don't have when blocks
// CHECK-NOT: firrtl.when %emptyAction_wf
// CHECK-NOT: firrtl.when %abortOnlyAction_wf
// CHECK-NOT: firrtl.when %emptyRule_wf

// Check that non-empty action has a when block
// CHECK: firrtl.when %nonEmptyAction_wf
// CHECK: firrtl.constant
// CHECK: firrtl.add
// CHECK: }