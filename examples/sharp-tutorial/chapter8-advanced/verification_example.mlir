// Module with formal properties
txn.module @SecureCounter {
  txn.instance @count of @Register<i32> 
  txn.instance @max of @Register<i32> 
  
  txn.action_method @set_max(%limit: i32) {
    txn.call @max::@write(%limit) : (i32) -> ()
    txn.yield
  }
  
  txn.action_method @increment() {
    %val = txn.call @count::@read() : () -> i32
    %limit = txn.call @max::@read() : () -> i32
    %cmp = arith.cmpi slt, %val, %limit : i32
    scf.if %cmp {
      %one = arith.constant 1 : i32
      %next = arith.addi %val, %one : i32
      txn.call @count::@write(%next) : (i32) -> ()
    }
    txn.yield
  }
  
  // Formal properties
  txn.property @never_exceed_max {
    %count_val = txn.call @count::@read() : () -> i32
    %max_val = txn.call @max::@read() : () -> i32
    %valid = arith.cmpi sle, %count_val, %max_val : i32
    txn.assert %valid : i1
  }
  
  txn.property @monotonic_increase {
    %old = txn.sample @count::@read() : () -> i32
    txn.call @this.increment() : () -> ()
    %new = txn.call @count::@read() : () -> i32
    %increased = arith.cmpi sge, %new, %old : i32
    txn.assert %increased : i1
  }
  
  txn.schedule [@set_max, @increment] {
    verification_depth = 20 : i32
  }
}

// Deadlock-free protocol verification
txn.module @Protocol {
  txn.instance @state of @Register<i8> 
  
  // State encoding
  %IDLE = arith.constant 0 : i8
  %REQ = arith.constant 1 : i8
  %ACK = arith.constant 2 : i8
  
  txn.action_method @request() {
    %s = txn.call @state::@read() : () -> i8
    %is_idle = arith.cmpi eq, %s, %IDLE : i8
    scf.if %is_idle {
      txn.call @state::@write(%REQ) : (i8) -> ()
    }
    txn.yield
  }
  
  txn.action_method @acknowledge() {
    %s = txn.call @state::@read() : () -> i8
    %is_req = arith.cmpi eq, %s, %REQ : i8
    scf.if %is_req {
      txn.call @state::@write(%ACK) : (i8) -> ()
    }
    txn.yield
  }
  
  txn.action_method @complete() {
    %s = txn.call @state::@read() : () -> i8
    %is_ack = arith.cmpi eq, %s, %ACK : i8
    scf.if %is_ack {
      txn.call @state::@write(%IDLE) : (i8) -> ()
    }
    txn.yield
  }
  
  // Liveness property
  txn.property @eventually_completes {
    txn.eventually {
      %s = txn.call @state::@read() : () -> i8
      %is_idle = arith.cmpi eq, %s, %IDLE : i8
      txn.assert %is_idle : i1
    }
  }
  
  txn.schedule [@request, @acknowledge, @complete]
}