// RUN: sharp-opt %s -convert-txn-to-func | FileCheck %s

// Test JIT lowering of txn.if and txn.return operations

txn.module @TestJIT {
  txn.action_method @testIf(%cond: i1) -> i32 {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    
    // Test txn.if with both then and else regions
    %result = txn.if %cond -> i32 {
      txn.yield %c1 : i32
    } else {
      txn.yield %c2 : i32
    }
    
    txn.return %result : i32
  }
  
  txn.action_method @testReturn(%val: i32) -> i32 {
    txn.return %val : i32
  }
  
  txn.action_method @testAbort(%cond: i1) {
    txn.if %cond {
      txn.abort
    } else {
      // Empty else block
    }
    txn.return
  }
  
  txn.schedule []
}

// CHECK: module {
// CHECK:   func.func @testIf(%arg0: i1) -> i32
// CHECK:     %[[C1:.*]] = arith.constant 1 : i32
// CHECK:     %[[C2:.*]] = arith.constant 2 : i32
// CHECK:     %[[RESULT:.*]] = scf.if %arg0 -> (i32) {
// CHECK:       scf.yield %[[C1]] : i32
// CHECK:     } else {
// CHECK:       scf.yield %[[C2]] : i32
// CHECK:     }
// CHECK:     return %[[RESULT]] : i32
// CHECK:   }
// CHECK:   func.func @testReturn(%arg0: i32) -> i32
// CHECK:     return %arg0 : i32
// CHECK:   }
// CHECK:   func.func @testAbort(%arg0: i1)
// CHECK:     scf.if %arg0 {
// CHECK:       func.return
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }