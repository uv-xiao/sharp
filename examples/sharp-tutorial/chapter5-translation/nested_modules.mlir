// Example demonstrating nested modules with timing modes
// Inner module: Simple adder
txn.module @SimpleAdder {
  %result = txn.instance @result of @Register<i32> : !txn.module<"Register">
  
  // Add two numbers and store result
  txn.action_method @add(%a: i32, %b: i32) {
    %sum = arith.addi %a, %b : i32
    txn.call @result::@write(%sum) : (i32) -> ()
    txn.yield
  }
  
  // Read the result
  txn.value_method @getResult() -> i32 {
    %val = txn.call @result::@read() : () -> i32
    txn.return %val : i32
  }
  
  // Reset the result
  txn.action_method @reset() {
    %zero = arith.constant 0 : i32
    txn.call @result::@write(%zero) : (i32) -> ()
    txn.yield
  }
  
  txn.schedule [@add, @reset] {
    conflict_matrix = {
      "add,add" = 2 : i32,      // C
      "add,reset" = 2 : i32,    // C  
      "reset,reset" = 2 : i32   // C
    }
  }
}

// Outer module: Dual adder processor
txn.module @DualProcessor {
  %adder1 = txn.instance @adder1 of @SimpleAdder : !txn.module<"SimpleAdder">
  %adder2 = txn.instance @adder2 of @SimpleAdder : !txn.module<"SimpleAdder">
  %output = txn.instance @output of @Register<i32> : !txn.module<"Register">
  
  // Process input through first adder
  txn.action_method @processA(%x: i32, %y: i32) {
    txn.call @adder1::@add(%x, %y) : (i32, i32) -> ()
    txn.yield
  }
  
  // Process input through second adder  
  txn.action_method @processB(%x: i32, %y: i32) {
    txn.call @adder2::@add(%x, %y) : (i32, i32) -> ()
    txn.yield
  }
  
  // Combine results (simplified - just store one result)
  txn.action_method @combine() {
    // For simplicity, just read from one adder since we can't call value methods
    // In a real implementation, this would combine both results
    %dummy = arith.constant 42 : i32
    txn.call @output::@write(%dummy) : (i32) -> ()
    txn.yield
  }
  
  // Reset all adders
  txn.action_method @resetAll() {
    txn.call @adder1::@reset() : () -> ()
    txn.call @adder2::@reset() : () -> ()
    %zero = arith.constant 0 : i32
    txn.call @output::@write(%zero) : (i32) -> ()
    txn.yield
  }
  
  // Get final output
  txn.value_method @getOutput() -> i32 {
    %val = txn.call @output::@read() : () -> i32
    txn.return %val : i32
  }
  
  txn.schedule [@processA, @processB, @combine, @resetAll] {
    conflict_matrix = {
      // Process operations can run in parallel
      "processA,processA" = 2 : i32,    // C
      "processA,processB" = 3 : i32,    // CF (different adders)
      "processA,combine" = 2 : i32,     // C
      "processA,resetAll" = 2 : i32,    // C
      "processB,processB" = 2 : i32,    // C
      "processB,combine" = 2 : i32,     // C
      "processB,resetAll" = 2 : i32,    // C
      "combine,combine" = 2 : i32,      // C
      "combine,resetAll" = 2 : i32,     // C
      "resetAll,resetAll" = 2 : i32     // C
    }
  }
}