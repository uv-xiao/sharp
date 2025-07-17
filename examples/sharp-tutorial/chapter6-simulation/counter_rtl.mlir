// Simple counter for RTL simulation
txn.module @RTLCounter {
  txn.instance @count of @Register<i8> 
  
  txn.value_method @read() -> i8 {
    %val = txn.call @count::@read() : () -> i8
    txn.return %val : i8
  }
  
  txn.action_method @increment() {
    %val = txn.call @count::@read() : () -> i8
    %one = arith.constant 1 : i8
    %next = arith.addi %val, %one : i8
    txn.call @count::@write(%next) : (i8) -> ()
    txn.yield
  }
  
  txn.rule @tick {
    txn.call @this.increment() : () -> ()
    txn.yield
  }
  
  txn.schedule [@read, @increment, @tick]
}