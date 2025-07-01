// RUN: sharp-opt --sharp-reachability-analysis --convert-txn-to-firrtl %s | FileCheck %s

// Test simple conflict_inside calculation without reachability analysis

// CHECK-LABEL: firrtl.circuit "SimpleConflictInside"
// CHECK: firrtl.module @SimpleConflictInside

txn.module @SimpleConflictInside {
  %reg = txn.instance @r of @Register : !txn.module<"Register">
  
  // Rule with multiple writes that conflict
  txn.rule @multiWrite {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    
    // Two writes to same register - always conflict
    txn.call @r::@write(%c0) : (i32) -> ()
    txn.call @r::@write(%c1) : (i32) -> ()
    
    txn.return
  }
  
  // CHECK: %[[AND:.*]] = firrtl.and %{{.*}}, %{{.*}} : {{.*}} -> !firrtl.uint<1>
  // CHECK: %[[NOT:.*]] = firrtl.not %[[AND]] : {{.*}} -> !firrtl.uint<1>
  // CHECK: firrtl.and %{{.*}}, %[[NOT]] : {{.*}} -> !firrtl.uint<1>
  
  txn.schedule [@multiWrite] {
    conflict_matrix = {
      "r::write,r::write" = 2 : i32  // C (conflict)
    }
  }
}

txn.primitive @Register type = "hw" interface = !txn.module<"Register"> {
  txn.fir_value_method @read() : () -> i32
  txn.fir_action_method @write() : (i32) -> ()
}