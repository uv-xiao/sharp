// Example demonstrating JIT simulation mode
// This compiles the module to LLVM IR for fast execution

txn.module @Accumulator {
  // Value method to compute sum
  txn.value_method @sum(%a: i32, %b: i32) -> i32 {
    %result = arith.addi %a, %b : i32
    txn.return %result : i32
  }
  
  // Value method to compute product
  txn.value_method @product(%a: i32, %b: i32) -> i32 {
    %result = arith.muli %a, %b : i32
    txn.return %result : i32
  }
  
  // Action method that does nothing (for testing)
  txn.action_method @noop() {
    txn.return
  }
  
  // Rule that always fires
  txn.rule @always_ready {
    %true = arith.constant true
    txn.return
  }
  
  txn.schedule [@always_ready, @noop]
}