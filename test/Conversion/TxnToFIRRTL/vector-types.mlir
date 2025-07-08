// RUN: sharp-opt --convert-txn-to-firrtl %s | FileCheck %s

// Test conversion with vector types

// CHECK-LABEL: firrtl.circuit "VectorOps"
// CHECK: firrtl.module @VectorOps

txn.module @VectorOps {
  // Test vector operations
  txn.value_method @getVector() -> vector<4xi32> {
    // Create a vector of zeros
    %c0 = arith.constant 0 : i32
    %vec = vector.broadcast %c0 : i32 to vector<4xi32>
    txn.return %vec : vector<4xi32>
  }
  
  txn.action_method @processVector(%vec: vector<8xi16>) {
    txn.return
  }
  
  txn.schedule [@processVector] {
    conflict_matrix = {}
  }
}

// CHECK: out %getVectorOUT: !firrtl.vector<{{.*}}, 4>
// CHECK: in %processVectorOUT: !firrtl.vector<{{.*}}, 8>