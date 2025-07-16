// Producer-Consumer pattern using FIFO
txn.module @ProducerConsumer attributes {top} {
  // FIFO buffer between producer and consumer
  %buffer = txn.instance @buffer of @FIFO<i32> : !txn.module<"FIFO">
  
  // State for producer
  %prod_count = txn.instance @prod_count of @Register<i32> : !txn.module<"Register">
  
  // State for consumer  
  %cons_sum = txn.instance @cons_sum of @Register<i32> : !txn.module<"Register">
  
  // Producer action: generate sequential values
  txn.action_method @produce() {
    // Check if FIFO has space
    %full = txn.call @buffer::@isFull() : () -> i1
    %true = arith.constant true
    %not_full = arith.xori %full, %true : i1
    
    // Only produce if not full (simplified - real hardware would use guards)
    %count = txn.call @prod_count::@read() : () -> i32
    txn.call @buffer::@enqueue(%count) : (i32) -> ()
    
    // Increment counter
    %one = arith.constant 1 : i32
    %next = arith.addi %count, %one : i32
    txn.call @prod_count::@write(%next) : (i32) -> ()
    
    txn.yield
  }
  
  // Consumer action: sum received values
  txn.action_method @consume() {
    // Check if FIFO has data
    %empty = txn.call @buffer::@isEmpty() : () -> i1
    %true = arith.constant true
    %not_empty = arith.xori %empty, %true : i1
    
    // Only consume if not empty
    %value = txn.call @buffer::@dequeue() : () -> i32
    
    // Add to running sum
    %sum = txn.call @cons_sum::@read() : () -> i32
    %new_sum = arith.addi %sum, %value : i32
    txn.call @cons_sum::@write(%new_sum) : (i32) -> ()
    
    txn.yield
  }
  
  // Value methods for monitoring
  txn.value_method @getProducerCount() -> i32 {
    %val = txn.call @prod_count::@read() : () -> i32
    txn.return %val : i32
  }
  
  txn.value_method @getConsumerSum() -> i32 {
    %val = txn.call @cons_sum::@read() : () -> i32
    txn.return %val : i32
  }
  
  txn.value_method @getBufferEmpty() -> i1 {
    %val = txn.call @buffer::@isEmpty() : () -> i1
    txn.return %val : i1
  }
  
  // Rules for autonomous operation
  txn.rule @autoProducer {
    // Producer runs when buffer not full
    %full = txn.call @buffer::@isFull() : () -> i1
    %true = arith.constant true
    %not_full = arith.xori %full, %true : i1
    // In real implementation, would use guards
    txn.call @this.produce() : () -> ()
    txn.yield
  }
  
  txn.rule @autoConsumer {
    // Consumer runs when buffer not empty
    %empty = txn.call @buffer::@isEmpty() : () -> i1
    %true = arith.constant true  
    %not_empty = arith.xori %empty, %true : i1
    // In real implementation, would use guards
    txn.call @this.consume() : () -> ()
    txn.yield
  }
  
  // Schedule all methods
  txn.schedule [@produce, @consume, @getProducerCount, @getConsumerSum, 
                @getBufferEmpty, @autoProducer, @autoConsumer] {
    // Key conflicts:
    // - produce conflicts with consume (both modify FIFO)
    // - Rules conflict with their respective actions
    conflict_matrix = {
      "produce,consume" = 2 : i32,      // C
      "consume,produce" = 2 : i32,      // C
      "autoProducer,produce" = 2 : i32, // C
      "autoConsumer,consume" = 2 : i32  // C
    }
  }
}