// Example demonstrating full Verilog export pipeline
// This shows Txn -> FIRRTL -> HW -> Verilog conversion

// Simple traffic light controller
txn.module @TrafficLight {
  // Action to change to red
  txn.action_method @goRed() {
    %c0 = arith.constant 0 : i32
    txn.return
  }
  
  // Action to change to yellow  
  txn.action_method @goYellow() {
    %c1 = arith.constant 1 : i32
    txn.return
  }
  
  // Action to change to green
  txn.action_method @goGreen() {
    %c2 = arith.constant 2 : i32
    txn.return
  }
  
  // Value method to check if red
  txn.value_method @isRed() -> i1 {
    %true = arith.constant true
    txn.return %true : i1
  }
  
  // Schedule with mutual exclusion
  txn.schedule [@goRed, @goYellow, @goGreen] {
    conflict_matrix = {
      "goRed,goYellow" = 2 : i32,     // C - only one at a time
      "goRed,goGreen" = 2 : i32,      // C
      "goYellow,goGreen" = 2 : i32    // C
    }
  }
}