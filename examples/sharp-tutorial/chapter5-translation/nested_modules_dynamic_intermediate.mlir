module attributes {sharp.conflict_matrix_inferred, sharp.general_checked, sharp.pre_synthesis_checked, sharp.primitive_gen_complete, sharp.reachability_analyzed} {
  txn.module @SimpleAdder {
    %0 = txn.instance @result of @Register<!firrtl.uint<32>> : !txn.module<"Register">
    txn.action_method @add(%arg0: !firrtl.uint<32>, %arg1: !firrtl.uint<32>) {
      %1 = firrtl.add %arg0, %arg1 : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<33>
      %2 = firrtl.bits %1 31 to 0 : (!firrtl.uint<33>) -> !firrtl.uint<32>
      txn.call @result::@write(%2) : (!firrtl.uint<32>) -> ()
      txn.return
    }
    txn.value_method @getResult() -> !firrtl.uint<32> {
      %1 = txn.call @result::@read() : () -> !firrtl.uint<32>
      txn.return %1 : !firrtl.uint<32>
    }
    txn.action_method @reset() {
      %c0_ui32 = firrtl.constant 0 : !firrtl.uint<32>
      txn.call @result::@write(%c0_ui32) : (!firrtl.uint<32>) -> ()
      txn.return
    }
    txn.schedule [@add, @reset] {conflict_matrix = {"add,add" = 2 : i32, "add,reset" = 2 : i32, "reset,add" = 2 : i32, "reset,reset" = 2 : i32}}
  }
  txn.module @DualProcessor attributes {top} {
    %0 = txn.instance @adder1 of @SimpleAdder : !txn.module<"SimpleAdder">
    %1 = txn.instance @adder2 of @SimpleAdder : !txn.module<"SimpleAdder">
    %2 = txn.instance @output of @Register<!firrtl.uint<32>> : !txn.module<"Register">
    txn.action_method @processA(%arg0: !firrtl.uint<32>, %arg1: !firrtl.uint<32>) {
      txn.call @adder1::@add(%arg0, %arg1) : (!firrtl.uint<32>, !firrtl.uint<32>) -> ()
      txn.return
    }
    txn.action_method @processB(%arg0: !firrtl.uint<32>, %arg1: !firrtl.uint<32>) {
      txn.call @adder2::@add(%arg0, %arg1) : (!firrtl.uint<32>, !firrtl.uint<32>) -> ()
      txn.return
    }
    txn.action_method @combine() {
      %c42_ui32 = firrtl.constant 42 : !firrtl.uint<32>
      txn.call @output::@write(%c42_ui32) : (!firrtl.uint<32>) -> ()
      txn.return
    }
    txn.action_method @resetAll() {
      %c0_ui32 = firrtl.constant 0 : !firrtl.uint<32>
      txn.call @adder1::@reset() : () -> ()
      txn.call @adder2::@reset() : () -> ()
      txn.call @output::@write(%c0_ui32) : (!firrtl.uint<32>) -> ()
      txn.return
    }
    txn.value_method @getOutput() -> !firrtl.uint<32> {
      %3 = txn.call @output::@read() : () -> !firrtl.uint<32>
      txn.return %3 : !firrtl.uint<32>
    }
    txn.schedule [@processA, @processB, @combine, @resetAll] {conflict_matrix = {"combine,combine" = 2 : i32, "combine,processA" = 2 : i32, "combine,processB" = 2 : i32, "combine,resetAll" = 2 : i32, "processA,combine" = 2 : i32, "processA,processA" = 2 : i32, "processA,processB" = 3 : i32, "processA,resetAll" = 2 : i32, "processB,combine" = 2 : i32, "processB,processA" = 3 : i32, "processB,processB" = 2 : i32, "processB,resetAll" = 2 : i32, "resetAll,combine" = 2 : i32, "resetAll,processA" = 2 : i32, "resetAll,processB" = 2 : i32, "resetAll,resetAll" = 2 : i32}}
  }
  txn.primitive @Register <!firrtl.uint<32>> type = "hw" interface = !txn.module<"Register<i32>"> {
    txn.fir_value_method @read() {firrtl.port = "read_data"} : () -> !firrtl.uint<32>
    txn.fir_action_method @write() {firrtl.data_port = "write_data", firrtl.enable_port = "write_enable"} : (!firrtl.uint<32>) -> ()
    txn.clock_by @clk
    txn.reset_by @rst
    txn.schedule [@write] {conflict_matrix = {"write,write" = 2 : i32}}
  } {firrtl.impl = "Register<i32>_impl"}
}

