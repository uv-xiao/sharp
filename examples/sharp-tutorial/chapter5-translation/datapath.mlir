// Datapath with FIFO and processing
txn.module @Datapath {
  %input_fifo = txn.instance @input_fifo of @FIFO<i32> : !txn.module<"FIFO">
  %output_reg = txn.instance @output_reg of @Register<i32> : !txn.module<"Register">
  %status = txn.instance @status of @Register<i1> : !txn.module<"Register">
  
  // Input data
  txn.action_method @pushData(%data: i32) {
    txn.call @input_fifo::@enqueue(%data) : (i32) -> ()
    txn.yield
  }
  
  // Process and store
  txn.action_method @process() {
    // Check if data available
    %empty = txn.call @input_fifo::@isEmpty() : () -> i1
    %true = arith.constant true
    %has_data = arith.xori %empty, %true : i1
    
    // Get data and process
    %data = txn.call @input_fifo::@dequeue() : () -> i32
    %two = arith.constant 2 : i32
    %processed = arith.muli %data, %two : i32
    
    // Store result
    txn.call @output_reg::@write(%processed) : (i32) -> ()
    txn.call @status::@write(%has_data) : (i1) -> ()
    txn.yield
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
  
  txn.schedule [@pushData, @process, @getOutput, @isReady]
}