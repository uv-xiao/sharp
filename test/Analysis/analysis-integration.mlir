// RUN: sharp-opt %s --sharp-infer-conflict-matrix --sharp-reachability-analysis --sharp-validate-method-attributes --sharp-pre-synthesis-check -allow-unregistered-dialect | FileCheck %s

// Comprehensive test for all analysis passes working together

// CHECK-LABEL: txn.module @IntegratedAnalysis
txn.module @IntegratedAnalysis {
  // Multiple primitives to create complex interactions
  %ctrl = txn.instance @ctrl of @Register<i8> : !txn.module<"Register">
  %data = txn.instance @data of @Register<i32> : !txn.module<"Register">
  %fifo1 = txn.instance @fifo1 of @FIFO<i32> : !txn.module<"FIFO">
  %fifo2 = txn.instance @fifo2 of @FIFO<i32> : !txn.module<"FIFO">
  %wire = txn.instance @wire of @Wire<i1> : !txn.module<"Wire">
  %mem = txn.instance @mem of @Memory<i32> : !txn.module<"Memory">
  
  // Value method for conflict-free computation
  txn.value_method @computeAddress(%base: i32, %offset: i32) -> i32 attributes {combinational} {
    %addr = arith.addi %base, %offset : i32
    %c255 = arith.constant 255 : i32
    %masked = arith.andi %addr, %c255 : i32
    txn.return %masked : i32
  }
  
  // Action with complex control flow for reachability analysis
  txn.action_method @producer(%value: i32) attributes {
    signal_prefix = "prod",
    enable_suffix = "_en",
    ready_suffix = "_rdy"
  } {
    // Read control register
    %ctrl_val = txn.call @ctrl::@read() : () -> i8
    
    %c0 = arith.constant 0 : i8
    %c1 = arith.constant 1 : i8
    %c2 = arith.constant 2 : i8
    
    %is_mode0 = arith.cmpi eq, %ctrl_val, %c0 : i8
    %is_mode1 = arith.cmpi eq, %ctrl_val, %c1 : i8
    %is_mode2 = arith.cmpi eq, %ctrl_val, %c2 : i8
    
    txn.if %is_mode0 {
      // Mode 0: Write to FIFO1
      %can_enq = txn.call @fifo1::@canEnq() : () -> i1
      txn.if %can_enq {
        txn.call @fifo1::@enq(%value) : (i32) -> ()
        txn.yield
      } else {
        txn.abort  // Abort if FIFO1 full
      }
      txn.yield
    } else {
      txn.if %is_mode1 {
        // Mode 1: Write to FIFO2 with transformation
        %c2_i32 = arith.constant 2 : i32
        %doubled = arith.muli %value, %c2_i32 : i32
        %can_enq = txn.call @fifo2::@canEnq() : () -> i1
        txn.if %can_enq {
          txn.call @fifo2::@enq(%doubled) : (i32) -> ()
          txn.yield
        } else {
          // Fallback to memory
          %c0_i32 = arith.constant 0 : i32
          txn.call @mem::@write(%c0_i32, %doubled) : (i32, i32) -> ()
          txn.yield
        }
        txn.yield
      } else {
        txn.if %is_mode2 {
          // Mode 2: Complex interaction
          %flag = txn.call @wire::@read() : () -> i1
          txn.if %flag {
            txn.call @data::@write(%value) : (i32) -> ()
            txn.yield
          } else {
            txn.abort  // Abort if flag not set
          }
          txn.yield
        } else {
          // Invalid mode
          txn.abort
        }
        txn.yield
      }
      txn.yield
    }
    txn.yield
  }
  
  // Consumer with inter-FIFO transfer
  txn.action_method @consumer() -> i32 attributes {
    signal_prefix = "cons",
    return_suffix = "_out"
  } {
    %can_deq1 = txn.call @fifo1::@canDeq() : () -> i1
    %can_enq2 = txn.call @fifo2::@canEnq() : () -> i1
    
    %both_ready = arith.andi %can_deq1, %can_enq2 : i1
    
    %result = txn.if %both_ready -> i32 {
      // Transfer from FIFO1 to FIFO2
      %fifo_data = txn.call @fifo1::@first() : () -> i32
      txn.call @fifo1::@deq() : () -> ()
      
      // Process data
      %c10 = arith.constant 10 : i32
      %processed = arith.addi %fifo_data, %c10 : i32
      
      txn.call @fifo2::@enq(%processed) : (i32) -> ()
      txn.yield %processed : i32
    } else {
      // Try direct read from data register
      %reg_data = txn.call @data::@read() : () -> i32
      %c0 = arith.constant 0 : i32
      %is_valid = arith.cmpi ne, %reg_data, %c0 : i32
      
      %result = txn.if %is_valid -> i32 {
        txn.yield %reg_data : i32
      } else {
        txn.abort  // No valid data available
      }
      txn.yield %result : i32
    }
    txn.return %result : i32
  }
  
  // Rule that calls other methods
  txn.rule @orchestrator {
    %flag = txn.call @wire::@read() : () -> i1
    
    txn.if %flag {
      // Produce data
      %c42 = arith.constant 42 : i32
      txn.call @producer(%c42) : (i32) -> ()
      
      // Try to consume
      %result = txn.call @consumer() : () -> i32
      
      // Store result
      %addr = txn.call @computeAddress(%result, %c42) : (i32, i32) -> i32
      txn.call @mem::@write(%addr, %result) : (i32, i32) -> ()
      txn.yield
    } else {
      // Flag not set
      txn.yield
    }
    txn.yield
  }
  
  // Partial schedule for conflict inference
  txn.schedule [@producer, @consumer, @orchestrator] {
    conflict_matrix = {
      // Explicitly specify some conflicts
      "producer,consumer" = 2 : i32,  // Both access FIFOs
      "orchestrator,producer" = 3 : i32  // orchestrator calls producer
      
      // Let inference determine:
      // - Self conflicts (all should be C)
      // - orchestrator vs consumer (should be C - orchestrator calls consumer)
      // - Transitive conflicts through resource access
    }
  }
}

// Check that all analyses complete successfully
// CHECK: txn.schedule [@producer, @consumer, @orchestrator] {
// CHECK-DAG: "producer,producer" = 2 : i32
// CHECK-DAG: "consumer,consumer" = 2 : i32
// CHECK-DAG: "orchestrator,orchestrator" = 2 : i32
// CHECK-DAG: "orchestrator,consumer" = 3 : i32

// Check reachability analysis runs without error
// TODO: Add specific guard checks when guard addition is implemented

// TODO: Add FIRRTL conversion tests when FIFO/Memory primitives support FIRRTL generation

// CHECK-LABEL: txn.module @ConflictInferenceEdgeCases
txn.module @ConflictInferenceEdgeCases {
  %shared = txn.instance @shared of @Register<i32> : !txn.module<"Register">
  
  // Simple actions that access same resource for conflict inference
  txn.action_method @read_action() {
    %val = txn.call @shared::@read() : () -> i32
    "test.use"(%val) : (i32) -> ()
    txn.yield
  }
  
  txn.action_method @write_action() {
    %c42 = arith.constant 42 : i32
    txn.call @shared::@write(%c42) : (i32) -> ()
    txn.yield
  }
  
  txn.action_method @read_write_action() {
    %val = txn.call @shared::@read() : () -> i32
    %inc = arith.addi %val, %val : i32
    txn.call @shared::@write(%inc) : (i32) -> ()
    txn.yield
  }
  
  // Inference should determine:
  // - read_action CF write_action (read SA write in Register)
  // - write_action C read_write_action (both write)
  // - All methods conflict with themselves
  txn.schedule [@read_action, @write_action, @read_write_action] {
    conflict_matrix = {
      // Let inference fill in the relationships
    }
  }
}

// CHECK: txn.schedule [@read_action, @write_action, @read_write_action] {
// CHECK-DAG: "read_action,read_action" = 2 : i32
// CHECK-DAG: "write_action,write_action" = 2 : i32
// CHECK-DAG: "read_write_action,read_write_action" = 2 : i32
// CHECK-DAG: "write_action,read_write_action" = 3 : i32
// CHECK-DAG: "read_action,write_action" = 3 : i32