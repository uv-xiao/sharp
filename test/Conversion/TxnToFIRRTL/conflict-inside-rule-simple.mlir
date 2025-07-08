// RUN: sharp-opt --sharp-reachability-analysis --convert-txn-to-firrtl %s | FileCheck %s

// Test conflict_inside calculation for rules (no method arguments)

// CHECK-LABEL: firrtl.circuit "RuleConflictInside"
txn.module @RuleConflictInside {
  %reg = txn.instance @r of @Register<i32> : !txn.module<"Register">
  
  // Rule with simple constant conditions
  txn.rule @simpleConflict {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    
    // First write with true condition
    txn.if %true {
      txn.call @r::@write(%c0) : (i32) -> ()
      txn.yield
    } else {
      txn.yield
    }
    
    // Second write with false condition (won't execute)
    txn.if %false {
      txn.call @r::@write(%c1) : (i32) -> ()
      txn.yield
    } else {
      txn.yield
    }
    
    txn.return
  }
  
  // CHECK: firrtl.module @RuleConflictInside
  // The conflict_inside should detect that one condition is true and one is false
  // So conflict_inside = (true && false) = false
  // CHECK: %[[AND:.*]] = firrtl.and %{{.*}}, %{{.*}} :
  // CHECK: %[[NOT:.*]] = firrtl.not %[[AND]] :
  // CHECK: firrtl.and %{{.*}}, %[[NOT]] :
  
  txn.schedule [@simpleConflict] {
    conflict_matrix = {
      "r::write,r::write" = 2 : i32  // C (conflict)
    }
  }
}

txn.primitive @Register type = "hw" interface = !txn.module<"Register"> {
  txn.fir_value_method @read() : () -> i32
  txn.fir_action_method @write() : (i32) -> ()
  txn.clock_by @clk
  txn.reset_by @rst
  txn.schedule [@write] {
    conflict_matrix = {
      "read,read" = 3 : i32,
      "read,write" = 3 : i32,
      "write,read" = 3 : i32,
      "write,write" = 2 : i32
    }
  }
} {firrtl.impl = "Register_impl"}