// Simple FIFO module using basic operations only
txn.module @SimpleFifo {
  txn.instance @data_reg of @Register<i32> 
  txn.instance @count of @Register<i32> 
  
  // Enqueue data into FIFO
  txn.action_method @enqueue(%data: i32) {
    // Store the data
    txn.call @data_reg::@write(%data) : (i32) -> ()
    
    // Increment count
    %current_count = txn.call @count::@read() : () -> i32
    %one = arith.constant 1 : i32
    %new_count = arith.addi %current_count, %one : i32
    txn.call @count::@write(%new_count) : (i32) -> ()
    
    txn.return
  }
  
  // Dequeue data from FIFO - THIS CAUSES THE ISSUE
  txn.action_method @dequeue() -> i32 {
    // Read the data
    %data = txn.call @data_reg::@read() : () -> i32
    
    // Decrement count
    %current_count = txn.call @count::@read() : () -> i32
    %one = arith.constant 1 : i32
    %new_count = arith.subi %current_count, %one : i32
    txn.call @count::@write(%new_count) : (i32) -> ()
    
    txn.return %data : i32
  }
  
  txn.schedule [@enqueue, @dequeue] {
    conflict_matrix = {
      // Enqueue and dequeue conflict with each other
      "enqueue,enqueue" = 2 : i32,     // C
      "enqueue,dequeue" = 2 : i32,     // C
      "dequeue,dequeue" = 2 : i32      // C
    }
  }
}

// Datapath with FIFO and processing - DEMONSTRATES NESTED MODULE LIMITATION
txn.module @Datapath attributes {top} {
  txn.instance @input_fifo of @SimpleFifo 
  txn.instance @output_reg of @Register<i32> 
  txn.instance @status of @Register<i1> 
  
  // Input data - this works (action method with no return value)
  txn.action_method @pushData(%data: i32) {
    txn.call @input_fifo::@enqueue(%data) : (i32) -> ()
    txn.return
  }
  
  // Process and store - THIS FAILS: action method with return value from child module
  txn.action_method @process() {
    // This line causes the error: "Could not find output port for instance method: input_fifo::dequeue"
    %data = txn.call @input_fifo::@dequeue() : () -> i32
    %two = arith.constant 2 : i32
    %processed = arith.muli %data, %two : i32
    
    // Store result
    txn.call @output_reg::@write(%processed) : (i32) -> ()
    %true = arith.constant true
    txn.call @status::@write(%true) : (i1) -> ()
    txn.return
  }
  
  // Read output
  txn.value_method @getOutput() -> i32 {
    %val = txn.call @output_reg::@read() : () -> i32
    txn.return %val : i32
  }
  
  // Check status
  txn.value_method @isReady() -> i1 {
    %val = txn.call @status::@read() : () -> i1
    txn.return %val : i1
  }
  
  txn.schedule [@pushData, @process] {
    conflict_matrix = {
      // pushData and process conflict (both access FIFO)
      "pushData,pushData" = 2 : i32,   // C
      "pushData,process" = 2 : i32,    // C
      "process,process" = 2 : i32      // C
    }
  }
}