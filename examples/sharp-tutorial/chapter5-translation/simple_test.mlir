// Simple test for conversion
txn.module @SimpleTest {
  txn.action_method @test() {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %add = arith.addi %zero, %one : i32
    txn.return
  }
}