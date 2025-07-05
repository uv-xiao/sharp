// Optimized pipeline with forwarding
txn.module @OptimizedPipeline {
  // Pipeline registers with bypass logic
  %stage1 = txn.instance @stage1 of @Register<i32> : !txn.module<"Register">
  %stage2 = txn.instance @stage2 of @Register<i32> : !txn.module<"Register">
  %stage3 = txn.instance @stage3 of @Register<i32> : !txn.module<"Register">
  
  // Forwarding paths
  %fwd1to3 = txn.instance @fwd1to3 of @Wire<i32> : !txn.module<"Wire">
  %fwd2to3 = txn.instance @fwd2to3 of @Wire<i32> : !txn.module<"Wire">
  
  txn.action_method @process(%use_fwd: i1) {
    scf.if %use_fwd {
      // Fast path with forwarding
      %s1 = txn.call @stage1::@read() : () -> i32
      %s2 = txn.call @stage2::@read() : () -> i32
      
      // Compute and forward
      %r1 = arith.muli %s1, %s1 : i32
      %r2 = arith.addi %s2, %r1 : i32
      
      txn.call @fwd1to3::@write(%r1) : (i32) -> ()
      txn.call @fwd2to3::@write(%r2) : (i32) -> ()
      txn.call @stage3::@write(%r2) : (i32) -> ()
    } else {
      // Normal pipeline advance
      %s1 = txn.call @stage1::@read() : () -> i32
      %s2 = txn.call @stage2::@read() : () -> i32
      
      txn.call @stage2::@write(%s1) : (i32) -> ()
      txn.call @stage3::@write(%s2) : (i32) -> ()
    }
    txn.yield
  }
  
  // Speculation and rollback
  txn.action_method @speculate(%pred: i32) {
    %backup = txn.call @stage3::@read() : () -> i32
    
    // Speculative execution
    %spec_result = arith.muli %pred, %pred : i32
    txn.call @stage3::@write(%spec_result) : (i32) -> ()
    
    // Validation
    %actual = txn.call @stage2::@read() : () -> i32
    %correct = arith.cmpi eq, %pred, %actual : i32
    
    // Rollback if mispredicted
    scf.if %correct {
      // Keep speculative result
    } else {
      txn.call @stage3::@write(%backup) : (i32) -> ()
    }
    txn.yield
  }
  
  txn.schedule [@process, @speculate] {
    optimization_hints = {
      "unroll_factor" = 2 : i32,
      "pipeline_depth" = 3 : i32,
      "enable_forwarding" = true
    }
  }
}

// Banking and parallelism
txn.module @BankedMemory {
  // 4-way banked memory for parallel access
  %bank0 = txn.instance @bank0 of @BRAM<i32> : !txn.module<"BRAM">
  %bank1 = txn.instance @bank1 of @BRAM<i32> : !txn.module<"BRAM">
  %bank2 = txn.instance @bank2 of @BRAM<i32> : !txn.module<"BRAM">
  %bank3 = txn.instance @bank3 of @BRAM<i32> : !txn.module<"BRAM">
  
  txn.action_method @parallel_read(%addr0: i32, %addr1: i32, %addr2: i32, %addr3: i32) 
      -> (i32, i32, i32, i32) {
    // All banks can be accessed in parallel
    %v0 = txn.call @bank0::@read(%addr0) : (i32) -> i32
    %v1 = txn.call @bank1::@read(%addr1) : (i32) -> i32
    %v2 = txn.call @bank2::@read(%addr2) : (i32) -> i32
    %v3 = txn.call @bank3::@read(%addr3) : (i32) -> i32
    
    txn.return %v0, %v1, %v2, %v3 : i32, i32, i32, i32
  }
  
  txn.action_method @streaming_write(%base: i32, %d0: i32, %d1: i32, %d2: i32, %d3: i32) {
    // Distribute writes across banks
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    
    %a0 = arith.addi %base, %c0 : i32
    %a1 = arith.addi %base, %c1 : i32
    %a2 = arith.addi %base, %c2 : i32
    %a3 = arith.addi %base, %c3 : i32
    
    txn.call @bank0::@write(%a0, %d0) : (i32, i32) -> ()
    txn.call @bank1::@write(%a1, %d1) : (i32, i32) -> ()
    txn.call @bank2::@write(%a2, %d2) : (i32, i32) -> ()
    txn.call @bank3::@write(%a3, %d3) : (i32, i32) -> ()
    
    txn.yield
  }
  
  txn.schedule [@parallel_read, @streaming_write]
}