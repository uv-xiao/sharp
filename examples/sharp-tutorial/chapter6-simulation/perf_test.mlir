// Module for performance testing
txn.module @PerfTest {
  txn.instance @acc of @Register<i64> 
  
  // Computation-heavy method
  txn.action_method @compute(%n: i32) {
    %acc_val = txn.call @acc::@read() : () -> i64
    %n_ext = arith.extsi %n : i32 to i64
    
    // Simulate heavy computation
    %c2 = arith.constant 2 : i64
    %r1 = arith.muli %n_ext, %c2 : i64
    %r2 = arith.addi %acc_val, %r1 : i64
    
    txn.call @acc::@write(%r2) : (i64) -> ()
    txn.yield
  }
  
  txn.value_method @result() -> i64 {
    %val = txn.call @acc::@read() : () -> i64
    txn.return %val : i64
  }
  
  txn.schedule [@compute, @result]
}