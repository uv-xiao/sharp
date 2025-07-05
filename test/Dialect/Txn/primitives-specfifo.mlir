// RUN: sharp-opt %s | FileCheck %s

// CHECK-LABEL: txn.module @SpecFIFOTest
txn.module @SpecFIFOTest {
  // CHECK: %fifo = txn.instance @fifo of @SpecFIFO<i32>
  %fifo = txn.instance @fifo of @SpecFIFO<i32> : !txn.module<"SpecFIFO">
  
  // Test enqueue method
  txn.action_method @enqueue_test(%data: i32) {
    // CHECK: txn.call @fifo::@enqueue(%{{.*}}) : (i32) -> ()
    txn.call @fifo::@enqueue(%data) : (i32) -> ()
    txn.yield
  }
  
  // Test dequeue method
  txn.action_method @dequeue_test() -> i32 {
    // CHECK: txn.call @fifo::@dequeue() : () -> i32
    %val = txn.call @fifo::@dequeue() : () -> i32
    txn.return %val : i32
  }
  
  // Test isEmpty method
  txn.value_method @is_empty_test() -> i1 {
    // CHECK: txn.call @fifo::@isEmpty() : () -> i1
    %empty = txn.call @fifo::@isEmpty() : () -> i1
    txn.return %empty : i1
  }
  
  // Test size method
  txn.value_method @size_test() -> i32 {
    // CHECK: txn.call @fifo::@size() : () -> i32
    %size = txn.call @fifo::@size() : () -> i32
    txn.return %size : i32
  }
  
  // Test peek method
  txn.value_method @peek_test() -> i32 {
    // CHECK: txn.call @fifo::@peek() : () -> i32
    %val = txn.call @fifo::@peek() : () -> i32
    txn.return %val : i32
  }
  
  // Complex operation: dequeue if not empty
  txn.action_method @safe_dequeue() -> i32 {
    %empty = txn.call @fifo::@isEmpty() : () -> i1
    %true = arith.constant true
    %not_empty = arith.xori %empty, %true : i1
    
    %zero = arith.constant 0 : i32
    %result = scf.if %not_empty -> i32 {
      %val = txn.call @fifo::@dequeue() : () -> i32
      scf.yield %val : i32
    } else {
      scf.yield %zero : i32
    }
    
    txn.return %result : i32
  }
  
  // CHECK: txn.schedule
  txn.schedule [@enqueue_test, @dequeue_test, @is_empty_test, @size_test, @peek_test, @safe_dequeue]
}

// CHECK-LABEL: txn.module @SpecFIFOProducerConsumer
txn.module @SpecFIFOProducerConsumer {
  %data_fifo = txn.instance @data_fifo of @SpecFIFO<i64> : !txn.module<"SpecFIFO">
  %cmd_fifo = txn.instance @cmd_fifo of @SpecFIFO<i8> : !txn.module<"SpecFIFO">
  
  // Producer sends data and command
  txn.action_method @produce(%data: i64, %cmd: i8) {
    txn.call @data_fifo::@enqueue(%data) : (i64) -> ()
    txn.call @cmd_fifo::@enqueue(%cmd) : (i8) -> ()
    txn.yield
  }
  
  // Consumer processes if both FIFOs have data
  txn.action_method @consume() -> (i64, i8) {
    %data_empty = txn.call @data_fifo::@isEmpty() : () -> i1
    %cmd_empty = txn.call @cmd_fifo::@isEmpty() : () -> i1
    %either_empty = arith.ori %data_empty, %cmd_empty : i1
    %true = arith.constant true
    %both_ready = arith.xori %either_empty, %true : i1
    
    %zero64 = arith.constant 0 : i64
    %zero8 = arith.constant 0 : i8
    
    %res:2 = scf.if %both_ready -> (i64, i8) {
      %data = txn.call @data_fifo::@dequeue() : () -> i64
      %cmd = txn.call @cmd_fifo::@dequeue() : () -> i8
      scf.yield %data, %cmd : i64, i8
    } else {
      scf.yield %zero64, %zero8 : i64, i8
    }
    
    txn.return %res#0, %res#1 : i64, i8
  }
  
  txn.schedule [@produce, @consume]
}