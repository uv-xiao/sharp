// RUN: sharp-opt %s -sharp-concurrent-sim 2>&1 | FileCheck %s

// Test DAM methodology concurrent simulation pass

// CHECK: remark: Generated concurrent simulation code using DAM methodology

txn.module @DataProcessor {
  txn.value_method @getData() -> i32 {
    %data = arith.constant 100 : i32
    txn.return %data : i32
  }
  
  txn.action_method @processData(%val: i32) attributes {timing = "static(5)"} {
    %c2 = arith.constant 2 : i32
    %result = arith.muli %val, %c2 : i32
    txn.yield
  }
  
  txn.rule @compute {
    %ready = arith.constant true
    txn.yield %ready : i1
  }
  
  txn.schedule [@compute, @getData, @processData] {
    conflict_matrix = {
      "compute,processData" = 2 : i32  // Conflict
    }
  }
}

txn.module @Controller {
  txn.action_method @start() {
    txn.yield
  }
  
  txn.action_method @stop() {
    txn.yield
  }
  
  txn.rule @monitor {
    %check = arith.constant true
    txn.yield %check : i1
  }
  
  txn.schedule [@monitor, @start, @stop] {
    conflict_matrix = {
      "start,stop" = 2 : i32  // Conflict
    }
  }
}