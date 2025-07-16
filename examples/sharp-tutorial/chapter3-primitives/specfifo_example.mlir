// Example demonstrating SpecFIFO for verification
txn.module @NetworkInterface attributes {top} {
  // Unbounded packet queue for specification
  %rx_queue = txn.instance @rx_queue of @SpecFIFO<i64> : !txn.module<"SpecFIFO">
  %tx_queue = txn.instance @tx_queue of @SpecFIFO<i64> : !txn.module<"SpecFIFO">
  
  // Statistics
  %rx_count = txn.instance @rx_count of @Register<i32> : !txn.module<"Register">
  %tx_count = txn.instance @tx_count of @Register<i32> : !txn.module<"Register">
  
  // Receive packet
  txn.action_method @receive(%packet: i64) {
    // Always succeeds with SpecFIFO
    txn.call @rx_queue::@enqueue(%packet) : (i64) -> ()
    
    // Update counter
    %count = txn.call @rx_count::@read() : () -> i32
    %one = arith.constant 1 : i32
    %new_count = arith.addi %count, %one : i32
    txn.call @rx_count::@write(%new_count) : (i32) -> ()
    
    txn.yield
  }
  
  // Process packets
  txn.action_method @process() {
    %rx_empty = txn.call @rx_queue::@isEmpty() : () -> i1
    %true = arith.constant true
    %has_packet = arith.xori %rx_empty, %true : i1
    
    txn.if %has_packet {
      // Get packet
      %packet = txn.call @rx_queue::@dequeue() : () -> i64
      
      // Simple processing: add header
      %header = arith.constant 0xFF00000000000000 : i64
      %processed = arith.ori %packet, %header : i64
      
      // Send to TX queue
      txn.call @tx_queue::@enqueue(%processed) : (i64) -> ()
    }
    
    txn.yield
  }
  
  // Transmit packet
  txn.action_method @transmit() -> i64 {
    %tx_empty = txn.call @tx_queue::@isEmpty() : () -> i1
    %true = arith.constant true
    %has_packet = arith.xori %tx_empty, %true : i1
    
    %zero = arith.constant 0 : i64
    %result = scf.if %has_packet -> i64 {
      %packet = txn.call @tx_queue::@dequeue() : () -> i64
      
      // Update counter
      %count = txn.call @tx_count::@read() : () -> i32
      %one = arith.constant 1 : i32
      %new_count = arith.addi %count, %one : i32
      txn.call @tx_count::@write(%new_count) : (i32) -> ()
      
      scf.yield %packet : i64
    } else {
      scf.yield %zero : i64
    }
    
    txn.return %result : i64
  }
  
  // Get queue depths
  txn.value_method @get_queue_status() -> (i32, i32) {
    %rx_size = txn.call @rx_queue::@size() : () -> i32
    %tx_size = txn.call @tx_queue::@size() : () -> i32
    txn.return %rx_size, %tx_size : i32, i32
  }
  
  // Peek at next packet without removing
  txn.value_method @peek_rx() -> i64 {
    %empty = txn.call @rx_queue::@isEmpty() : () -> i1
    %zero = arith.constant 0 : i64
    
    %result = scf.if %empty -> i64 {
      scf.yield %zero : i64
    } else {
      %packet = txn.call @rx_queue::@peek() : () -> i64
      scf.yield %packet : i64
    }
    
    txn.return %result : i64
  }
  
  txn.schedule [@receive, @process, @transmit, @get_queue_status, @peek_rx]
}