module {
  txn.module @Counter {
    %0 = txn.instance @count of @Register<i32> : !txn.module<"Register">
    txn.value_method @getValue() -> i32 {
      %1 = txn.call @count.read() : () -> i32
      txn.return %1 : i32
    }
    txn.action_method @increment() {
      %1 = txn.call @count.read() : () -> i32
      %c1_i32 = arith.constant 1 : i32
      %2 = arith.addi %1, %c1_i32 : i32
      txn.call @count.write(%2) : (i32) -> ()
      txn.return
    }
    txn.rule @autoIncrement {
      %1 = txn.call @getValue() : () -> i32
      %c100_i32 = arith.constant 100 : i32
      %2 = arith.cmpi ult, %1, %c100_i32 : i32
      txn.if %2 {
        txn.call @increment() : () -> ()
        txn.yield
      } else {
        txn.yield
      }
      txn.yield
    }
    txn.schedule [@autoIncrement, @getValue, @increment] {conflict_matrix = {}}
  }
}

