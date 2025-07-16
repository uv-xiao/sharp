module attributes {sharp.conflict_matrix_inferred, sharp.general_checked, sharp.pre_synthesis_checked, sharp.primitive_gen_complete, sharp.reachability_analyzed} {
  txn.module @HardwareCounter attributes {top} {
    %0 = txn.instance @count of @Register<i32> : !txn.module<"Register">
    txn.value_method @getCount() -> i32 {
      %1 = txn.call @count::@read() : () -> i32
      txn.return %1 : i32
    }
    txn.action_method @increment() {
      %1 = txn.call @count::@read() : () -> i32
      %c1_i32 = arith.constant 1 : i32
      %2 = arith.addi %1, %c1_i32 : i32
      txn.call @count::@write(%2) : (i32) -> ()
      txn.return
    }
    txn.action_method @decrement() {
      %1 = txn.call @count::@read() : () -> i32
      %c1_i32 = arith.constant 1 : i32
      %2 = arith.subi %1, %c1_i32 : i32
      txn.call @count::@write(%2) : (i32) -> ()
      txn.return
    }
    txn.action_method @reset() {
      %c0_i32 = arith.constant 0 : i32
      txn.call @count::@write(%c0_i32) : (i32) -> ()
      txn.return
    }
    txn.schedule [@increment, @decrement, @reset] {conflict_matrix = {"decrement,decrement" = 2 : i32, "decrement,increment" = 2 : i32, "decrement,reset" = 3 : i32, "increment,decrement" = 2 : i32, "increment,increment" = 2 : i32, "increment,reset" = 3 : i32, "reset,decrement" = 3 : i32, "reset,increment" = 3 : i32, "reset,reset" = 2 : i32}}
  }
  txn.primitive @Register <i32> type = "hw" interface = !txn.module<"Register<i32>"> {
    txn.fir_value_method @read() {firrtl.port = "read_data"} : () -> i32
    txn.fir_action_method @write() {firrtl.data_port = "write_data", firrtl.enable_port = "write_enable"} : (i32) -> ()
    txn.clock_by @clk
    txn.reset_by @rst
    txn.schedule [@write] {conflict_matrix = {"write,write" = 2 : i32}}
  } {firrtl.impl = "Register<i32>_impl"}
}

