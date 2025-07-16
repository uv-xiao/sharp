module attributes {sharp.conflict_matrix_inferred, sharp.general_checked, sharp.pre_synthesis_checked, sharp.primitive_gen_complete, sharp.reachability_analyzed} {
  txn.module @HardwareCounter attributes {top} {
    %0 = txn.instance @count of @Register<!firrtl.uint<32>> : !txn.module<"Register">
    txn.value_method @getCount() -> !firrtl.uint<32> {
      %1 = txn.call @count::@read() : () -> !firrtl.uint<32>
      txn.return %1 : !firrtl.uint<32>
    }
    txn.action_method @increment() {
      %c1_ui32 = firrtl.constant 1 : !firrtl.uint<32>
      %1 = txn.call @count::@read() : () -> !firrtl.uint<32>
      %2 = firrtl.add %1, %c1_ui32 : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<33>
      %3 = firrtl.bits %2 31 to 0 : (!firrtl.uint<33>) -> !firrtl.uint<32>
      txn.call @count::@write(%3) : (!firrtl.uint<32>) -> ()
      txn.return
    }
    txn.action_method @decrement() {
      %c1_ui32 = firrtl.constant 1 : !firrtl.uint<32>
      %1 = txn.call @count::@read() : () -> !firrtl.uint<32>
      %2 = firrtl.sub %1, %c1_ui32 : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<33>
      %3 = firrtl.bits %2 31 to 0 : (!firrtl.uint<33>) -> !firrtl.uint<32>
      txn.call @count::@write(%3) : (!firrtl.uint<32>) -> ()
      txn.return
    }
    txn.action_method @reset() {
      %c0_ui32 = firrtl.constant 0 : !firrtl.uint<32>
      txn.call @count::@write(%c0_ui32) : (!firrtl.uint<32>) -> ()
      txn.return
    }
    txn.schedule [@increment, @decrement, @reset] {conflict_matrix = {"decrement,decrement" = 2 : i32, "decrement,increment" = 2 : i32, "decrement,reset" = 3 : i32, "increment,decrement" = 2 : i32, "increment,increment" = 2 : i32, "increment,reset" = 3 : i32, "reset,decrement" = 3 : i32, "reset,increment" = 3 : i32, "reset,reset" = 2 : i32}}
  }
  txn.primitive @Register <!firrtl.uint<32>> type = "hw" interface = !txn.module<"Register<i32>"> {
    txn.fir_value_method @read() {firrtl.port = "read_data"} : () -> !firrtl.uint<32>
    txn.fir_action_method @write() {firrtl.data_port = "write_data", firrtl.enable_port = "write_enable"} : (!firrtl.uint<32>) -> ()
    txn.clock_by @clk
    txn.reset_by @rst
    txn.schedule [@write] {conflict_matrix = {"write,write" = 2 : i32}}
  } {firrtl.impl = "Register<i32>_impl"}
}

