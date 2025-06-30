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

// CHECK-LABEL: firrtl.circuit "ComplexArithmetic"

// Check empty module conversion
// CHECK: firrtl.module @EmptyModule
// CHECK-SAME: in %clock: !firrtl.clock
// CHECK-SAME: in %reset: !firrtl.uint<1>

// Check module with only value methods
// CHECK: firrtl.module @OnlyValueMethods
// CHECK: firrtl.constant 1
// CHECK: firrtl.connect %get1OUT
// CHECK: firrtl.constant 2
// CHECK: firrtl.connect %get2OUT

// Check empty actions
// CHECK: firrtl.module @EmptyActions
// CHECK: %noop1_wf = firrtl.node %noop1EN
// CHECK: %noop2_wf = firrtl.node %noop2EN
// CHECK: firrtl.when %noop1_wf
// CHECK: firrtl.when %noop2_wf

// Check complex arithmetic
// CHECK: firrtl.module @ComplexArithmetic
// CHECK: firrtl.add
// CHECK: firrtl.sub
// CHECK: firrtl.mul
// CHECK: firrtl.gt
// CHECK: firrtl.mux

// Check module with schedule
// CHECK: firrtl.module @NoSchedule
// CHECK: %action_wf = firrtl.node %actionEN
// CHECK: firrtl.when %action_wf