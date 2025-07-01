// RUN: sharp-opt --sharp-reachability-analysis %s | FileCheck %s

// This example demonstrates the dominance issue with reachability analysis

// CHECK-LABEL: txn.module @DominanceExample
txn.module @DominanceExample {
  %reg = txn.instance @r of @Register : !txn.module<"Register">
  
  // CHECK-LABEL: txn.action_method @problematic
  txn.action_method @problematic(%arg0: i1) {
    // After reachability analysis, it tries to create:
    // %0 = arith.constant true
    // %1 = arith.xori %arg0, %0   <-- Problem: %0 is created after %arg0 but used here
    // %2 = arith.andi %0, %1      
    // %3 = arith.andi %0, %arg0   
    
    txn.if %arg0 {
      %c5 = arith.constant 5 : i32
      // CHECK: txn.call @r::@write if %arg0 : i1 then(%{{.*}}) : (i32) -> ()
      txn.call @r::@write(%c5) : (i32) -> ()
      txn.yield
    } else {
      txn.yield
    }
    txn.return
  }
  
  txn.schedule [@problematic] {
    conflict_matrix = {
      "r::write,r::write" = 2 : i32
    }
  }
}

txn.primitive @Register type = "hw" interface = !txn.module<"Register"> {
  txn.fir_action_method @write() : (i32) -> ()
}

// The issue occurs because:
// 1. ReachabilityAnalysis creates new arith operations (constant, xor, and)
// 2. These operations are inserted at the beginning of the block
// 3. But they reference %arg0 which is a block argument
// 4. The new operations try to use %arg0 before they are defined in the IR order
//
// Solution approaches:
// 1. Insert operations after all block arguments
// 2. Create operations lazily when needed
// 3. Use a different approach for tracking conditions