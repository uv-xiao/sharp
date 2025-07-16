// A counter with increment, decrement, and configurable step
txn.module @Counter attributes {top} {
  // State: current value and step size
  %value = txn.instance @value of @Register<!firrtl.uint<32>> : !txn.module<"Register">
  %step = txn.instance @step of @Register<!firrtl.uint<32>> : !txn.module<"Register">
  
  // Value method: read current count
  txn.value_method @getValue() -> !firrtl.uint<32> {
    %val = txn.call @value::@read() : () -> !firrtl.uint<32>
    txn.return %val : !firrtl.uint<32>
  }
  
  // Value method: read step size
  txn.value_method @getStep() -> !firrtl.uint<32> {
    %val = txn.call @step::@read() : () -> !firrtl.uint<32>
    txn.return %val : !firrtl.uint<32>
  }
  
  // Action method: increment by step
  txn.action_method @increment() {
    %current = txn.call @value::@read() : () -> !firrtl.uint<32>
    %step_val = txn.call @step::@read() : () -> !firrtl.uint<32>
    %new_wide = firrtl.add %current, %step_val : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<33>
    %new = firrtl.bits %new_wide 31 to 0 : (!firrtl.uint<33>) -> !firrtl.uint<32>
    txn.call @value::@write(%new) : (!firrtl.uint<32>) -> ()
    txn.return
  }
  
  // Action method: decrement by step
  txn.action_method @decrement() {
    %current = txn.call @value::@read() : () -> !firrtl.uint<32>
    %step_val = txn.call @step::@read() : () -> !firrtl.uint<32>
    %new_wide = firrtl.sub %current, %step_val : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<33>
    %new = firrtl.bits %new_wide 31 to 0 : (!firrtl.uint<33>) -> !firrtl.uint<32>
    txn.call @value::@write(%new) : (!firrtl.uint<32>) -> ()
    txn.return
  }
  
  // Action method: set custom step
  txn.action_method @setStep(%new_step: !firrtl.uint<32>) {
    txn.call @step::@write(%new_step) : (!firrtl.uint<32>) -> ()
    txn.return
  }
  
  // Action method: reset to zero
  txn.action_method @reset() {
    %zero = firrtl.constant 0 : !firrtl.uint<32>
    txn.call @value::@write(%zero) : (!firrtl.uint<32>) -> ()
    txn.return
  }
  
  // Schedule with conflict information
  txn.schedule [@reset, @increment, @decrement, @setStep] {
    conflict_matrix = {
       "increment,setStep" = 2 : i32, 
       "decrement,setStep" = 2 : i32
    }
  }
}