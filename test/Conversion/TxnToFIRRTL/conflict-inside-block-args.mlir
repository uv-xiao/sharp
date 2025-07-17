// RUN: sharp-opt --sharp-reachability-analysis --convert-txn-to-firrtl %s | FileCheck %s

// Test conflict_inside with block arguments mapped to FIRRTL ports

// CHECK-LABEL: firrtl.circuit "BlockArgHandling"
txn.module @BlockArgHandling {
  %reg = txn.instance @r of @Register : index
  
  // Action method with multiple conditions derived from arguments
  txn.action_method @complexConditions(%cond1: i1, %cond2: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    
    // First write controlled by cond1
    txn.if %cond1 {
      txn.call @r::@write(%c0) : (i32) -> ()
      txn.yield
    } else {
      txn.yield
    }
    
    // Second write controlled by cond2
    txn.if %cond2 {
      txn.call @r::@write(%c1) : (i32) -> ()
      txn.yield
    } else {
      txn.yield
    }
    
    // Third write controlled by AND of both conditions
    %both = arith.andi %cond1, %cond2 : i1
    txn.if %both {
      txn.call @r::@write(%c2) : (i32) -> ()
      txn.yield
    } else {
      txn.yield
    }
    
    txn.return
  }
  
  // CHECK: firrtl.module @BlockArgHandling({{.*}}in %complexConditionsOUT_arg0: !firrtl.uint<1>{{.*}}in %complexConditionsOUT_arg1: !firrtl.uint<1>{{.*}}in %complexConditionsEN: !firrtl.uint<1>{{.*}}out %complexConditionsRDY: !firrtl.uint<1>{{.*}})
  
  // The conflict_inside calculation should use these mapped arguments
  // There are three pairs of conflicts:
  // 1. write(%c0) with write(%c1): cond1 && cond2  
  // 2. write(%c0) with write(%c2): cond1 && (cond1 && cond2) = cond1 && cond2
  // 3. write(%c1) with write(%c2): cond2 && (cond1 && cond2) = cond1 && cond2
  // conflict_inside = OR of all = cond1 && cond2
  // CHECK: %{{.*}} = firrtl.and %complexConditionsOUT_arg0, %complexConditionsOUT_arg1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  // CHECK: %{{.*}} = firrtl.not %{{.*}} : (!firrtl.uint<1>) -> !firrtl.uint<1>
  // CHECK: %{{.*}} = firrtl.and %complexConditionsEN, %{{.*}} : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  
  txn.schedule [@complexConditions] {
    conflict_matrix = {
      "r::write,r::write" = 2 : i32  // C (conflict)
    }
  }
}

txn.primitive @Register type = "hw" interface = index {
  txn.fir_value_method @read() : () -> i32
  txn.fir_action_method @write() : (i32) -> ()
}