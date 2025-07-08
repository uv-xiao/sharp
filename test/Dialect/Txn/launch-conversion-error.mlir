// RUN: not sharp-opt --convert-txn-to-firrtl %s 2>&1 | FileCheck %s

// CHECK: error: future operations are not yet supported in FIRRTL conversion. Multi-cycle execution requires additional synthesis infrastructure.

txn.module @TestLaunch {
  txn.action_method @test() {
    txn.future {
      %done = txn.launch after 2 {
        txn.yield
      }
    }
    txn.return
  }
  
  txn.schedule [@test]
}