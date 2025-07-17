// Hardware counter for translation
txn.module @HardwareCounter attributes {top} {
  txn.instance @count of @Register<i32> 
  
  // Enable signal (input)
  txn.value_method @getCount() -> i32 {
    %val = txn.call @count::@read() : () -> i32
    txn.return %val : i32
  }
  
  // Increment action
  txn.action_method @increment() {
    %current = txn.call @count::@read() : () -> i32
    %one = arith.constant 1 : i32
    %next = arith.addi %current, %one : i32
    txn.call @count::@write(%next) : (i32) -> ()
    txn.return
  }
  
  // Decrement action
  txn.action_method @decrement() {
    %current = txn.call @count::@read() : () -> i32
    %one = arith.constant 1 : i32
    %next = arith.subi %current, %one : i32
    txn.call @count::@write(%next) : (i32) -> ()
    txn.return
  }
  
  // Reset action
  txn.action_method @reset() {
    %zero = arith.constant 0 : i32
    txn.call @count::@write(%zero) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@increment, @decrement, @reset] {
    conflict_matrix = {
      // Actions conflict with each other
      "increment,increment" = 2 : i32,     // C
      "increment,decrement" = 2 : i32,     // C
      "increment,reset" = 3 : i32,         // CF
      "decrement,decrement" = 2 : i32,     // C
      "decrement,reset" = 3 : i32,         // CF
      "reset,reset" = 2 : i32              // C
    }
  }
}