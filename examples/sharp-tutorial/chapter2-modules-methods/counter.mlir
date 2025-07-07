// A counter with increment, decrement, and configurable step
txn.module @Counter {
  // State: current value and step size
  %value = txn.instance @value of @Register<i32> : !txn.module<"Register">
  %step = txn.instance @step of @Register<i32> : !txn.module<"Register">
  
  // Value method: read current count
  txn.value_method @getValue() -> i32 {
    %val = txn.call @value::@read() : () -> i32
    txn.return %val : i32
  }
  
  // Value method: read step size
  txn.value_method @getStep() -> i32 {
    %val = txn.call @step::@read() : () -> i32
    txn.return %val : i32
  }
  
  // Action method: increment by step
  txn.action_method @increment() {
    %current = txn.call @value::@read() : () -> i32
    %step_val = txn.call @step::@read() : () -> i32
    %new = arith.addi %current, %step_val : i32
    txn.call @value::@write(%new) : (i32) -> ()
    txn.yield
  }
  
  // Action method: decrement by step
  txn.action_method @decrement() {
    %current = txn.call @value::@read() : () -> i32
    %step_val = txn.call @step::@read() : () -> i32
    %new = arith.subi %current, %step_val : i32
    txn.call @value::@write(%new) : (i32) -> ()
    txn.yield
  }
  
  // Action method: set custom step
  txn.action_method @setStep(%new_step: i32) {
    txn.call @step::@write(%new_step) : (i32) -> ()
    txn.yield
  }
  
  // Action method: reset to zero
  txn.action_method @reset() {
    %zero = arith.constant 0 : i32
    txn.call @value::@write(%zero) : (i32) -> ()
    txn.yield
  }
  
  // Schedule with conflict information
  txn.schedule [@increment, @decrement, @setStep, @reset] {
    conflict_matrix = {
      // Actions conflict with each other
      "increment,decrement" = 2 : i32,  // C
      "increment,setStep" = 2 : i32,    // C
      "increment,reset" = 2 : i32,      // C
      "decrement,setStep" = 2 : i32,    // C
      "decrement,reset" = 2 : i32,      // C
      "setStep,reset" = 2 : i32         // C
    }
  }
}