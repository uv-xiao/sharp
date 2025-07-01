// RUN: sharp-opt --convert-txn-to-firrtl %s | FileCheck %s

// Test conversion with wider integer types

// CHECK-LABEL: firrtl.circuit "WiderTypes"
// CHECK: firrtl.module @WiderTypes

txn.module @WiderTypes {
  // Test 8-bit operations
  txn.value_method @get8bit() -> i8 {
    %c42 = arith.constant 42 : i8
    txn.return %c42 : i8
  }
  
  // Test 16-bit operations
  txn.action_method @process16bit(%val: i16) {
    txn.return
  }
  
  // Test 64-bit operations
  txn.rule @compute64bit {
    %c1 = arith.constant 1 : i64
    %c2 = arith.constant 2 : i64
    %sum = arith.addi %c1, %c2 : i64
    txn.return
  }
  
  txn.schedule [@get8bit, @process16bit, @compute64bit] {
    conflict_matrix = {}
  }
}

// CHECK: out %get8bitOUT: !firrtl.sint<8>
// CHECK: in %process16bitOUT: !firrtl.sint<16>
// CHECK: %c42_si8 = firrtl.constant 42 : !firrtl.sint<8>
// CHECK: %c1_si64 = firrtl.constant 1 : !firrtl.sint<64>
// CHECK: %c2_si64 = firrtl.constant 2 : !firrtl.sint<64>
// CHECK: firrtl.add %{{.*}}, %{{.*}} : (!firrtl.sint<64>, !firrtl.sint<64>) -> !firrtl.sint<65>