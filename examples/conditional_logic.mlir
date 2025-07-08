// Example demonstrating conditional logic and analysis passes
// This works without primitives

txn.module @ConditionalLogic {
  // Action that does computation based on mode
  txn.action_method @compute(%mode: i1, %val: i32) attributes {timing = "static(1)"} {
    txn.if %mode {
      // Mode 1: double the value
      %double = arith.addi %val, %val : i32
      txn.yield  
    } else {
      // Mode 0: halve the value
      %two = arith.constant 2 : i32
      %half = arith.divsi %val, %two : i32
      txn.yield
    }
    txn.return
  }
  
  // Value method that computes based on inputs
  txn.value_method @selectMax(%a: i32, %b: i32) -> i32 attributes {timing = "combinational"} {
    %cmp = arith.cmpi sgt, %a, %b : i32
    %max = arith.select %cmp, %a, %b : i32
    txn.return %max : i32
  }
  
  // Rule that fires conditionally
  txn.rule @conditionalRule {
    %ten = arith.constant 10 : i32
    %twenty = arith.constant 20 : i32
    %max = txn.call @selectMax(%ten, %twenty) : (i32, i32) -> i32
    
    // Check if max > 15
    %fifteen = arith.constant 15 : i32
    %should_fire = arith.cmpi sgt, %max, %fifteen : i32
    
    txn.if %should_fire {
      %mode = arith.constant true
      txn.call @compute(%mode, %max) : (i1, i32) -> ()
      txn.yield
    } else {
      txn.yield
    }
    
    txn.return
  }
  
  // Another action for testing conflicts
  txn.action_method @reset() attributes {timing = "static(1)"} {
    %zero = arith.constant 0 : i32
    txn.return
  }
  
  // Schedule with partial specification
  txn.schedule [@compute, @conditionalRule, @reset] {
    conflict_matrix = {
      // Specify that compute and reset conflict
      "compute,reset" = 2 : i32
    }
  }
}

// Run this example with:
// sharp-opt conditional_logic.mlir -sharp-infer-conflict-matrix
// sharp-opt conditional_logic.mlir -sharp-reachability-analysis