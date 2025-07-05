// Hardware counter for translation
txn.module @HardwareCounter {
  %count = txn.instance @count of @Register<i32> : !txn.module<"Register">
  
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
    txn.yield
  }
  
  // Decrement action
  txn.action_method @decrement() {
    %current = txn.call @count::@read() : () -> i32
    %one = arith.constant 1 : i32
    %next = arith.subi %current, %one : i32
    txn.call @count::@write(%next) : (i32) -> ()
    txn.yield
  }
  
  // Reset action
  txn.action_method @reset() {
    %zero = arith.constant 0 : i32
    txn.call @count::@write(%zero) : (i32) -> ()
    txn.yield
  }
  
  txn.schedule [@getCount, @increment, @decrement, @reset] {
    conflict_matrix = {
      // Value method doesn't conflict
      "getCount,getCount" = 3 : i32,       // CF
      "getCount,increment" = 3 : i32,      // CF
      "getCount,decrement" = 3 : i32,      // CF
      "getCount,reset" = 3 : i32,          // CF
      
      // Actions conflict with each other
      "increment,increment" = 2 : i32,     // C
      "increment,decrement" = 2 : i32,     // C
      "increment,reset" = 2 : i32,         // C
      "decrement,decrement" = 2 : i32,     // C
      "decrement,reset" = 2 : i32,         // C
      "reset,reset" = 2 : i32              // C
    }
  }
}