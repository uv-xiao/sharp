// RUN: sharp-opt --sharp-reachability-analysis %s | FileCheck %s

// Test that reachability analysis adds condition operands to txn.call

txn.module @ReachabilityTest {
  %reg = txn.instance @r of @Register : !txn.module<"Register">
  
  // Rule with conditional method calls
  txn.rule @conditionalWrites {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c5 = arith.constant 5 : i32
    %c10 = arith.constant 10 : i32
    
    // Read value
    %val = txn.call @r::@read() : () -> i32
    
    // First condition
    %cond1 = arith.cmpi eq, %val, %c0 : i32
    txn.if %cond1 {
      // This call should have condition %cond1
      // CHECK: txn.call @r::@write if %{{.*}} then(%{{.*}}) : (i32) -> ()
      txn.call @r::@write(%c5) : (i32) -> ()
      txn.yield
    } else {
      txn.yield
    }
    
    // Second condition
    %cond2 = arith.cmpi sgt, %val, %c1 : i32
    txn.if %cond2 {
      // This call should have condition %cond2
      // CHECK: txn.call @r::@write if %{{.*}} then(%{{.*}}) : (i32) -> ()
      txn.call @r::@write(%c10) : (i32) -> ()
      txn.yield
    } else {
      txn.yield
    }
    
    txn.return
  }
  
  // Action method with nested conditions
  txn.action_method @nestedConditions(%enable: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    
    txn.if %enable {
      %val = txn.call @r::@read() : () -> i32
      %innerCond = arith.cmpi eq, %val, %c0 : i32
      
      txn.if %innerCond {
        // This call should have condition (%enable && %innerCond)
        // CHECK: txn.call @r::@write if %{{.*}} then(%{{.*}}) : (i32) -> ()
        txn.call @r::@write(%c1) : (i32) -> ()
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
  
  txn.schedule [@conditionalWrites, @nestedConditions] {
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