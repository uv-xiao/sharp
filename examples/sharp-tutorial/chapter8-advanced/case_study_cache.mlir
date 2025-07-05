// High-performance cache controller
txn.module @CacheController {
  // Cache state - simplified for tutorial
  %data = txn.instance @data of @Register<i64> : !txn.module<"Register">
  %tag = txn.instance @tag of @Register<i32> : !txn.module<"Register">
  %valid = txn.instance @valid of @Register<i1> : !txn.module<"Register">
  %dirty = txn.instance @dirty of @Register<i1> : !txn.module<"Register">
  
  // Statistics
  %hits = txn.instance @hits of @Register<i64> : !txn.module<"Register">
  %misses = txn.instance @misses of @Register<i64> : !txn.module<"Register">
  
  txn.action_method @read(%addr: i32) -> i64 {
    // Extract tag from address
    %c12 = arith.constant 12 : i32
    %req_tag = arith.shrui %addr, %c12 : i32
    
    // Check if valid and tag matches
    %is_valid = txn.call @valid::@read() : () -> i1
    %stored_tag = txn.call @tag::@read() : () -> i32
    %tag_match = arith.cmpi eq, %req_tag, %stored_tag : i32
    %hit = arith.andi %is_valid, %tag_match : i1
    
    %data_val = txn.call @data::@read() : () -> i64
    %zero = arith.constant 0 : i64
    %result = arith.select %hit, %data_val, %zero : i64
    
    // Update statistics
    %h = txn.call @hits::@read() : () -> i64
    %m = txn.call @misses::@read() : () -> i64
    %one = arith.constant 1 : i64
    
    scf.if %hit {
      %h_new = arith.addi %h, %one : i64
      txn.call @hits::@write(%h_new) : (i64) -> ()
    } else {
      %m_new = arith.addi %m, %one : i64
      txn.call @misses::@write(%m_new) : (i64) -> ()
    }
    
    txn.return %result : i64
  }
  
  txn.action_method @write(%addr: i32, %value: i64) {
    %c12 = arith.constant 12 : i32
    %req_tag = arith.shrui %addr, %c12 : i32
    
    // Write allocate policy
    txn.call @tag::@write(%req_tag) : (i32) -> ()
    txn.call @data::@write(%value) : (i64) -> ()
    %true = arith.constant true
    txn.call @valid::@write(%true) : (i1) -> ()
    txn.call @dirty::@write(%true) : (i1) -> ()
    
    txn.yield
  }
  
  txn.value_method @get_hit_rate() -> i32 {
    %h = txn.call @hits::@read() : () -> i64
    %m = txn.call @misses::@read() : () -> i64
    %total = arith.addi %h, %m : i64
    
    // Calculate percentage (simplified integer math)
    %c100 = arith.constant 100 : i64
    %h_scaled = arith.muli %h, %c100 : i64
    %rate = arith.divui %h_scaled, %total : i64
    %rate_i32 = arith.trunci %rate : i64 to i32
    
    txn.return %rate_i32 : i32
  }
  
  txn.schedule [@read, @write, @get_hit_rate] {
    performance_targets = {
      "read_latency" = 1 : i32,
      "write_latency" = 1 : i32
    }
  }
}