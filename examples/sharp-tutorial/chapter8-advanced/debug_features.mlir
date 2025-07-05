// Module with extensive debugging
txn.module @DebugExample {
  %state = txn.instance @state of @Register<i32> : !txn.module<"Register">
  
  // Performance counters
  %method_calls = txn.instance @calls of @Register<i64> : !txn.module<"Register">
  %cycle_count = txn.instance @cycles of @Register<i64> : !txn.module<"Register">
  
  txn.action_method @process(%input: i32) {
    // Increment call counter
    %calls = txn.call @method_calls::@read() : () -> i64
    %one = arith.constant 1 : i64
    %new_calls = arith.addi %calls, %one : i64
    txn.call @method_calls::@write(%new_calls) : (i64) -> ()
    
    // Actual processing
    %current = txn.call @state::@read() : () -> i32
    %result = arith.addi %current, %input : i32
    txn.call @state::@write(%result) : (i32) -> ()
    
    txn.yield
  }
  
  // Cycle counter rule
  txn.rule @count_cycles {
    %c = txn.call @cycle_count::@read() : () -> i64
    %one = arith.constant 1 : i64
    %new_c = arith.addi %c, %one : i64
    txn.call @cycle_count::@write(%new_c) : (i64) -> ()
    txn.yield
  }
  
  txn.value_method @get_stats() -> (i64, i64) {
    %calls = txn.call @method_calls::@read() : () -> i64
    %cycles = txn.call @cycle_count::@read() : () -> i64
    txn.return %calls, %cycles : i64, i64
  }
  
  txn.schedule [@process, @get_stats, @count_cycles] {
    debug_level = 2 : i32,
    profile_enabled = true
  }
}