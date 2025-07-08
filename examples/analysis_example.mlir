// Example demonstrating various analysis passes

txn.module @AnalysisExample {
  // Action methods with different relationships
  txn.action_method @action1() {
    %c1 = arith.constant 1 : i32
    txn.yield
  }
  
  txn.action_method @action2() {
    %c2 = arith.constant 2 : i32
    txn.yield
  }
  
  txn.action_method @action3() {
    %c3 = arith.constant 3 : i32
    txn.yield
  }
  
  // Value method
  txn.value_method @getValue() -> i32 {
    %c42 = arith.constant 42 : i32
    txn.return %c42 : i32
  }
  
  // Rule that calls methods conditionally
  txn.rule @conditionalRule {
    %val = txn.call @getValue() : () -> i32
    %threshold = arith.constant 40 : i32
    %cond = arith.cmpi sgt, %val, %threshold : i32
    
    txn.if %cond {
      txn.call @action1() : () -> ()
      txn.yield
    } else {
      txn.call @action2() : () -> ()
      txn.yield
    }
    
    txn.return
  }
  
  // Partial schedule - will be completed by analysis
  txn.schedule [@action1, @conditionalRule] {
    conflict_matrix = {
      // Only specify one relationship
      "action1,action2" = 0 : i32  // SB - action1 must come before action2
    }
  }
}