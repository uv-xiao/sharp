module {
  txn.module @Counter {
    %0 = txn.instance @value of @Register<!firrtl.uint<32>> : !txn.module<"Register">
    %1 = txn.instance @step of @Register<!firrtl.uint<32>> : !txn.module<"Register">
    txn.value_method @getValue() -> !firrtl.uint<32> {
      %2 = txn.call @value::@read() : () -> !firrtl.uint<32>
      txn.return %2 : !firrtl.uint<32>
    }
    txn.value_method @getStep() -> !firrtl.uint<32> {
      %2 = txn.call @step::@read() : () -> !firrtl.uint<32>
      txn.return %2 : !firrtl.uint<32>
    }
    txn.action_method @increment() {
      %2 = txn.call @value::@read() : () -> !firrtl.uint<32>
      %3 = txn.call @step::@read() : () -> !firrtl.uint<32>
      %4 = firrtl.add %2, %3 : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<33>
      %5 = firrtl.bits %4 31 to 0 : (!firrtl.uint<33>) -> !firrtl.uint<32>
      txn.call @value::@write(%5) : (!firrtl.uint<32>) -> ()
      txn.return
    }
    txn.action_method @decrement() {
      %2 = txn.call @value::@read() : () -> !firrtl.uint<32>
      %3 = txn.call @step::@read() : () -> !firrtl.uint<32>
      %4 = firrtl.sub %2, %3 : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<33>
      %5 = firrtl.bits %4 31 to 0 : (!firrtl.uint<33>) -> !firrtl.uint<32>
      txn.call @value::@write(%5) : (!firrtl.uint<32>) -> ()
      txn.return
    }
    txn.action_method @setStep(%arg0: !firrtl.uint<32>) {
      txn.call @step::@write(%arg0) : (!firrtl.uint<32>) -> ()
      txn.return
    }
    txn.action_method @reset() {
      %c0_ui32 = firrtl.constant 0 : !firrtl.uint<32>
      txn.call @value::@write(%c0_ui32) : (!firrtl.uint<32>) -> ()
      txn.return
    }
    txn.schedule [@reset, @increment, @decrement, @setStep] {conflict_matrix = {"decrement,setStep" = 3 : i32, "increment,setStep" = 3 : i32}}
  }
}

