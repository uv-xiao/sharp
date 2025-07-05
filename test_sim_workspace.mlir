// Test module for simulation workspace generation
txn.module @Counter {
  %count = txn.instance @count of @Register<i32> : !txn.module<"Register">
  
  txn.value_method @getValue() -> i32 {
    %val = txn.call @count.read() : () -> i32
    txn.return %val : i32
  }
  
  txn.action_method @increment() {
    %old = txn.call @count.read() : () -> i32
    %one = arith.constant 1 : i32
    %new = arith.addi %old, %one : i32
    txn.call @count.write(%new) : (i32) -> ()
    txn.return
  }
  
  txn.rule @autoIncrement {
    %val = txn.call @getValue() : () -> i32
    %limit = arith.constant 100 : i32
    %cond = arith.cmpi ult, %val, %limit : i32
    txn.if %cond {
      txn.call @increment() : () -> ()
      txn.yield
    } else {
      txn.yield
    }
    txn.yield
  }
  
  txn.schedule [@autoIncrement, @getValue, @increment] {
    conflict_matrix = {}
  }
}