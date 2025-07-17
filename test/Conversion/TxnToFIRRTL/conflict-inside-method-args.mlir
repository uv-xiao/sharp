// RUN: sharp-opt --sharp-reachability-analysis --convert-txn-to-firrtl %s | FileCheck %s

// Test conflict_inside with method arguments as conditions

// CHECK-LABEL: firrtl.circuit "MethodArgConditions"
txn.module @MethodArgConditions {
  %reg = txn.instance @r of @Register : index
  
  // Action method where condition is directly the method argument
  txn.action_method @conditionalWrite(%enable: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    
    // Two writes that conflict, both controlled by the same condition
    txn.if %enable {
      txn.call @r::@write(%c0) : (i32) -> ()
      txn.call @r::@write(%c1) : (i32) -> ()
      txn.yield
    } else {
      txn.yield
    }
    
    txn.return
  }
  
  // CHECK: firrtl.module @MethodArgConditions({{.*}}in %conditionalWriteOUT: !firrtl.uint<1>{{.*}}in %conditionalWriteEN: !firrtl.uint<1>{{.*}}out %conditionalWriteRDY: !firrtl.uint<1>{{.*}})
  
  // Since both writes have the same condition (%enable mapped to %conditionalWriteOUT),
  // conflict_inside should be: %conditionalWriteOUT && %conditionalWriteOUT = %conditionalWriteOUT
  // CHECK: %{{.*}} = firrtl.and %conditionalWriteOUT, %conditionalWriteOUT : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  // CHECK: %{{.*}} = firrtl.not %{{.*}} : (!firrtl.uint<1>) -> !firrtl.uint<1>
  // CHECK: %{{.*}} = firrtl.and %conditionalWriteEN, %{{.*}} : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  
  txn.schedule [@conditionalWrite] {
    conflict_matrix = {
      "r::write,r::write" = 2 : i32  // C (conflict)
    }
  }
}

txn.primitive @Register type = "hw" interface = index {
  txn.fir_value_method @read() : () -> i32
  txn.fir_action_method @write() : (i32) -> ()
}