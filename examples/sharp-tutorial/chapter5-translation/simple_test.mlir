module {
  txn.module @SimpleTest {
    txn.action_method @test() {
      %true = arith.constant true
      %false = arith.constant false
      %result = arith.xori %true, %false : i1
      txn.if %result {
        txn.yield
      } else {
        txn.yield
      }
      txn.return
    }
    txn.schedule [@test] {conflict_matrix = {"test,test" = 2 : i32}}
  }
}