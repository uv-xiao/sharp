// RUN: sharp-opt --sharp-reachability-analysis --convert-txn-to-firrtl %s | FileCheck %s

// Test conflict_inside calculation with reachability conditions

// CHECK-LABEL: firrtl.circuit "ConflictInsideWithConditions"
txn.module @ConflictInsideWithConditions {
  %reg = txn.instance @r of @Register : !txn.module<"Register">
  
  // Rule with conditional writes that may conflict
  txn.rule @conditionalWrites {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c5 = arith.constant 5 : i32
    
    %val = txn.call @r::@read() : () -> i32
    %cond = arith.cmpi eq, %val, %c0 : i32
    
    txn.if %cond {
      // Write if val == 0
      txn.call @r::@write(%c1) : (i32) -> ()
      txn.yield
    } else {
      // Write if val != 0
      txn.call @r::@write(%c5) : (i32) -> ()
      txn.yield
    }
    
    txn.return
  }
  
  // CHECK: firrtl.module @ConflictInsideWithConditions
  // No conflict_inside because the two writes are mutually exclusive
  // CHECK-NOT: firrtl.not
  
  txn.schedule [@conditionalWrites] {
    conflict_matrix = {
      "r::read,r::write" = 2 : i32,
      "r::write,r::write" = 2 : i32  // C (conflict)
    }
  }
}

// Test with nested conditions where writes can conflict
// CHECK-LABEL: firrtl.module @NestedConflicts
txn.module @NestedConflicts {
  %reg = txn.instance @r of @Register : !txn.module<"Register">
  
  txn.action_method @nestedWrites(%enable: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c10 = arith.constant 10 : i32
    
    txn.if %enable {
      %val = txn.call @r::@read() : () -> i32
      %inner = arith.cmpi sgt, %val, %c0 : i32
      
      // First write - always happens if enable is true
      txn.call @r::@write(%c1) : (i32) -> ()
      
      txn.if %inner {
        // Second write - happens if enable && (val > 0)
        // This conflicts with the first write
        txn.call @r::@write(%c10) : (i32) -> ()
        txn.yield
      } else {
        txn.yield
      }
      txn.yield
    } else {
      txn.yield
    }
    
    txn.return
  }
  
  // CHECK-DAG: %[[COND1:.*]] = firrtl.and %{{.*}}, %{{.*}} : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  // CHECK-DAG: %[[NOT:.*]] = firrtl.not %[[COND1]] : (!firrtl.uint<1>) -> !firrtl.uint<1>
  // CHECK-DAG: firrtl.and %{{.*}}, %[[NOT]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  
  txn.schedule [@nestedWrites] {
    conflict_matrix = {
      "r::read,r::write" = 2 : i32,
      "r::write,r::write" = 2 : i32
    }
  }
}

txn.primitive @Register type = "hw" interface = !txn.module<"Register"> {
  txn.fir_value_method @read() : () -> i32
  txn.fir_action_method @write() : (i32) -> ()
}