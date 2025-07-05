// Three-stage pipeline example
txn.module @Pipeline {
  // Pipeline registers
  %stage1 = txn.instance @stage1 of @Register<i32> : !txn.module<"Register">
  %stage2 = txn.instance @stage2 of @Register<i32> : !txn.module<"Register">
  %stage3 = txn.instance @stage3 of @Register<i32> : !txn.module<"Register">
  
  // Input new data
  txn.action_method @input(%data: i32) {
    txn.call @stage1::@write(%data) : (i32) -> ()
    txn.yield
  }
  
  // Advance pipeline
  txn.action_method @advance() {
    // Read all stages
    %s1 = txn.call @stage1::@read() : () -> i32
    %s2 = txn.call @stage2::@read() : () -> i32
    %s3 = txn.call @stage3::@read() : () -> i32
    
    // Process
    %two = arith.constant 2 : i32
    %p1 = arith.addi %s1, %two : i32
    %p2 = arith.muli %s2, %two : i32
    
    // Write next stage
    txn.call @stage2::@write(%p1) : (i32) -> ()
    txn.call @stage3::@write(%p2) : (i32) -> ()
    txn.yield
  }
  
  // Get output
  txn.value_method @output() -> i32 {
    %val = txn.call @stage3::@read() : () -> i32
    txn.return %val : i32
  }
  
  // Auto-advance rule
  txn.rule @clock {
    txn.call @this.advance() : () -> ()
    txn.yield
  }
  
  txn.schedule [@input, @advance, @output, @clock] {
    conflict_matrix = {
      "input,advance" = 2 : i32,    // C
      "advance,clock" = 2 : i32,    // C
      "input,clock" = 2 : i32       // C
    }
  }
}