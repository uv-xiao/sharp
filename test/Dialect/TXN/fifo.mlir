// RUN: sharp-opt %s -allow-unregistered-dialect | sharp-opt -allow-unregistered-dialect | FileCheck %s

// Test FIFO implementation with updated syntax

// CHECK-LABEL: txn.module @FIFO
txn.module @FIFO {
  // CHECK: %{{.*}} = txn.instance @storage of @Storage : !txn.module<"Storage">
  %storage = txn.instance @storage of @Storage : !txn.module<"Storage">
  
  // CHECK: txn.value_method @isEmpty() -> i1
  txn.value_method @isEmpty() -> i1 {
    %empty = txn.call @storage.isEmpty() : () -> i1
    txn.return %empty : i1
  }
  
  // CHECK: txn.value_method @isFull() -> i1
  txn.value_method @isFull() -> i1 {
    %full = txn.call @storage.isFull() : () -> i1
    txn.return %full : i1
  }
  
  // CHECK: txn.action_method @enqueue(%{{.*}}: i32)
  txn.action_method @enqueue(%data: i32) {
    %full = txn.call @isFull() : () -> i1
    txn.if %full {
      txn.abort
    } else {
      txn.call @storage.write(%data) : (i32) -> ()
      txn.yield
    }
    txn.return
  }
  
  // CHECK: txn.action_method @dequeue() -> i32
  txn.action_method @dequeue() -> i32 {
    %empty = txn.call @isEmpty() : () -> i1
    txn.if %empty {
      txn.abort
    } else {
      txn.yield
    }
    %data = txn.call @storage.read() : () -> i32
    txn.return %data : i32
  }
  
  // CHECK: txn.rule @autoProcess
  txn.rule @autoProcess {
    %ready = txn.call @isProcessReady() : () -> i1
    txn.if %ready {
      %data = txn.call @dequeue() : () -> i32
      txn.call @process(%data) : (i32) -> ()
      txn.yield
    } else {
      // Do nothing
      txn.yield
    }
  }
  
  txn.value_method @isProcessReady() -> i1 {
    %false = arith.constant false
    txn.return %false : i1
  }
  
  txn.action_method @process(%data: i32) {
    txn.return
  }
  
  // CHECK: txn.schedule [@isEmpty, @isFull, @enqueue, @dequeue, @autoProcess]
  txn.schedule [@isEmpty, @isFull, @enqueue, @dequeue, @autoProcess]
}

// CHECK-LABEL: txn.module @Storage
txn.module @Storage {
  txn.value_method @isEmpty() -> i1 {
    %true = arith.constant true
    txn.return %true : i1
  }
  
  txn.value_method @isFull() -> i1 {
    %false = arith.constant false
    txn.return %false : i1
  }
  
  txn.action_method @write(%data: i32) {
    txn.return
  }
  
  txn.value_method @read() -> i32 {
    %c0 = arith.constant 0 : i32
    txn.return %c0 : i32
  }
  
  txn.schedule [@isEmpty, @isFull, @write, @read]
}

// CHECK-LABEL: func.func @test_fifo_usage
func.func @test_fifo_usage() -> i32 {
  // Test using FIFO in a schedule-like context
  %c42 = arith.constant 42 : i32
  
  // Enqueue data
  txn.call @FIFO.enqueue(%c42) : (i32) -> ()
  
  // Dequeue data
  %data = txn.call @FIFO.dequeue() : () -> i32
  
  return %data : i32
}