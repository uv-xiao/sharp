// RUN: sharp-opt --convert-txn-to-firrtl %s | FileCheck %s

// Test basic module conversion without instances
txn.module @SimpleCounter {
  txn.value_method @read() -> i32 {
    %c42_i32 = arith.constant 42 : i32
    txn.return %c42_i32 : i32
  }
  
  txn.action_method @increment() {
    txn.return
  }
  
  txn.rule @auto_inc {
    txn.return
  }
  
  txn.schedule [@increment, @auto_inc] {
    conflict_matrix = {
      "increment,increment" = 2 : i32,
      "auto_inc,increment" = 2 : i32
    }
  }
}

// CHECK-LABEL: firrtl.circuit "SimpleCounter"
// CHECK: firrtl.module @SimpleCounter
// CHECK-SAME: in %clock: !firrtl.clock
// CHECK-SAME: in %reset: !firrtl.uint<1>
// CHECK-SAME: out %readOUT: !firrtl.sint<32>
// CHECK-SAME: in %incrementEN: !firrtl.uint<1>
// CHECK-SAME: out %incrementRDY: !firrtl.uint<1>

// Check will-fire logic
// CHECK: %increment_wf = firrtl.node %incrementEN
// CHECK: %auto_inc_wf = firrtl.node

// Check value method implementation
// CHECK: firrtl.constant 42
// CHECK: firrtl.connect %readOUT

// Check ready signal
// CHECK: firrtl.connect %incrementRDY

// Check when blocks for actions
// CHECK: firrtl.when %increment_wf
// CHECK: firrtl.when %auto_inc_wf