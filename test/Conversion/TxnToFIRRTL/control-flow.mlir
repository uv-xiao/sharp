// RUN: sharp-opt --convert-txn-to-firrtl %s | FileCheck %s

// Test control flow conversion (txn.if -> firrtl.when)
txn.module @ControlFlowTest {
  txn.action_method @write(%arg0: i32) {
    txn.return
  }
  
  txn.rule @conditional_write {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c42 = arith.constant 42 : i32
    %cond = arith.cmpi eq, %c0, %c1 : i32
    
    txn.if %cond {
      // This write only happens if condition is true
      txn.call @write(%c42) : (i32) -> ()
      txn.yield
    } else {
      // Different value in else branch
      %c99 = arith.constant 99 : i32
      txn.call @write(%c99) : (i32) -> ()
      txn.yield
    }
    
    txn.return
  }
  
  txn.schedule [@write, @conditional_write] {
    conflict_matrix = {
      "write,write" = 2 : i32,
      "conditional_write,write" = 2 : i32
    }
  }
}

// CHECK-LABEL: firrtl.circuit "ControlFlowTest"
// CHECK: firrtl.module @ControlFlowTest

// Check will-fire logic
// CHECK: %write_wf = firrtl.node %writeEN
// CHECK: %conditional_write_wf = firrtl.node

// Check rule body with conditional
// CHECK: firrtl.when %conditional_write_wf
// CHECK: firrtl.constant 0
// CHECK: firrtl.constant 1
// CHECK: firrtl.constant 42
// CHECK: firrtl.eq

// Check nested when for if-then-else
// CHECK: firrtl.when
// CHECK: } else {
// CHECK: firrtl.constant 99