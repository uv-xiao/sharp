// RUN: sharp-opt --convert-txn-to-firrtl %s | FileCheck %s

// Test action methods with guards
txn.module @GuardedMethods {
  txn.action_method @conditional_write(%val: i32) {
    %c0 = arith.constant 0 : i32
    %cond = arith.cmpi sgt, %val, %c0 : i32
    
    txn.if %cond {
      // Only write if value > 0
      txn.yield
    } else {
      // This path should affect the ready signal
      txn.yield
    }
    
    txn.return
  }
  
  txn.value_method @is_ready() -> i1 {
    %true = arith.constant true
    txn.return %true : i1
  }
  
  txn.rule @check_ready {
    %ready = txn.call @is_ready() : () -> i1
    txn.if %ready {
      %c42 = arith.constant 42 : i32
      txn.call @conditional_write(%c42) : (i32) -> ()
      txn.yield
    } else {
      txn.yield
    }
    txn.return
  }
  
  txn.schedule [@conditional_write, @check_ready] {
    conflict_matrix = {
      "conditional_write,conditional_write" = 2 : i32,
      "check_ready,conditional_write" = 3 : i32
    }
  }
}

// CHECK-LABEL: firrtl.circuit "GuardedMethods"
// CHECK: firrtl.module @GuardedMethods

// Check ports exist
// CHECK-SAME: in %conditional_writeOUT: !firrtl.sint<32>
// CHECK-SAME: in %conditional_writeEN: !firrtl.uint<1>
// CHECK-SAME: out %conditional_writeRDY: !firrtl.uint<1>
// CHECK-SAME: out %is_readyOUT: !firrtl.uint<1>

// Check will-fire signals
// CHECK: %conditional_write_wf = firrtl.node %conditional_writeEN
// CHECK: %check_ready_wf = firrtl.node

// Check conditional in method
// CHECK: firrtl.when %conditional_write_wf
// CHECK: firrtl.gt
// CHECK: firrtl.when
// CHECK: } else {

// Check value method implementation
// CHECK: firrtl.constant 1
// CHECK: firrtl.connect %is_readyOUT

// Check rule with local method call
// CHECK: firrtl.when %check_ready_wf
// CHECK: %is_ready_call = firrtl.node %is_readyOUT
// CHECK: firrtl.when
// CHECK: firrtl.constant 42