// RUN: sharp-opt --convert-txn-to-firrtl %s | FileCheck %s

// Test edge cases and special scenarios

// Empty module with no methods
txn.module @EmptyModule {
  txn.schedule [] {
    conflict_matrix = {}
  }
}

// Module with only value methods
txn.module @OnlyValueMethods {
  txn.value_method @get1() -> i32 {
    %c1 = arith.constant 1 : i32
    txn.return %c1 : i32
  }
  
  txn.value_method @get2() -> i32 {
    %c2 = arith.constant 2 : i32
    txn.return %c2 : i32
  }
  
  txn.schedule [] {
    conflict_matrix = {}
  }
}

// Module with empty action methods
txn.module @EmptyActions {
  txn.action_method @noop1() {
    txn.return
  }
  
  txn.action_method @noop2() {
    txn.return
  }
  
  txn.schedule [@noop1, @noop2] {
    conflict_matrix = {
      "noop1,noop2" = 3 : i32  // Conflict-free
    }
  }
}

// Module with complex arithmetic in value methods
txn.module @ComplexArithmetic {
  txn.value_method @compute(%a: i32, %b: i32) -> i32 {
    // Test various arithmetic operations
    %sum = arith.addi %a, %b : i32
    %diff = arith.subi %a, %b : i32
    %prod = arith.muli %sum, %diff : i32
    
    // Test comparisons
    %c0 = arith.constant 0 : i32
    %is_positive = arith.cmpi sgt, %prod, %c0 : i32
    
    // Conditional return
    %result = arith.select %is_positive, %prod, %c0 : i32
    txn.return %result : i32
  }
  
  txn.schedule [] {
    conflict_matrix = {}
  }
}

// Module with no schedule (should still convert)
txn.module @NoSchedule {
  txn.action_method @action() {
    txn.return
  }
  
  txn.schedule [@action] {
    conflict_matrix = {
      "action,action" = 2 : i32
    }
  }
}

// CHECK: module {
// CHECK-NEXT: firrtl.circuit

// Check all modules are converted
// CHECK-DAG: firrtl.module @EmptyModule
// CHECK-DAG: firrtl.module @OnlyValueMethods
// CHECK-DAG: firrtl.module @EmptyActions
// CHECK-DAG: firrtl.module @ComplexArithmetic
// CHECK-DAG: firrtl.module @NoSchedule
// Check key conversions:
// Value methods produce constants in OnlyValueMethods
// CHECK-DAG: firrtl.constant 1
// CHECK-DAG: firrtl.constant 2

// Empty actions don't generate when blocks anymore
// CHECK-NOT: firrtl.when %noop1_wf
// CHECK-NOT: firrtl.when %noop2_wf

// Empty action methods don't have when blocks
// CHECK-NOT: firrtl.when %action_wf