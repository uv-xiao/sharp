// RUN: sharp-opt %s -allow-unregistered-dialect | FileCheck %s

// Comprehensive test for multi-cycle operations with proper syntax

// CHECK-LABEL: txn.module @MultiCycleProcessor
txn.module @MultiCycleProcessor {
  %pc = txn.instance @pc of @Register<i32> : index
  %state = txn.instance @state of @Register<i8> : index
  %mem = txn.instance @mem of @Memory<i32> : index
  %fifo = txn.instance @fifo of @FIFO<i32> : index
  
  // Simple value method
  txn.value_method @decode(%instr: i32) -> i8 attributes {combinational} {
    %c0 = arith.constant 0 : i32
    %mask8 = arith.constant 255 : i32
    %opcode_i32 = arith.andi %instr, %mask8 : i32
    %opcode = arith.trunci %opcode_i32 : i32 to i8
    txn.return %opcode : i8
  }
  
  // Action method with multi-cycle behavior
  txn.action_method @executeInstruction(%instr: i32) {
    %opcode = txn.call @decode(%instr) : (i32) -> i8
    
    // Check if it's a load instruction (opcode 1)
    %c1_i8 = arith.constant 1 : i8
    %is_load = arith.cmpi eq, %opcode, %c1_i8 : i8
    
    txn.if %is_load {
      // Multi-cycle load operation
      txn.future {
        // CHECK: txn.launch after 2
        %done1 = txn.launch after 2 {
          %addr = arith.constant 100 : i32
          %data = txn.call @mem::@read(%addr) : (i32) -> i32
          txn.call @fifo::@enq(%data) : (i32) -> ()
          txn.yield
        }
        txn.yield
      }
      txn.yield
    } else {
      // Single cycle operation
      %c42 = arith.constant 42 : i32
      txn.call @fifo::@enq(%c42) : (i32) -> ()
      txn.yield
    }
    txn.yield
  }
  
  // Rule with conditional multi-cycle operations
  txn.rule @processor {
    %current_state = txn.call @state::@read() : () -> i8
    %c0_i8 = arith.constant 0 : i8
    %c1_i8 = arith.constant 1 : i8
    %c2_i8 = arith.constant 2 : i8
    
    // State 0: Fetch
    %is_fetch = arith.cmpi eq, %current_state, %c0_i8 : i8
    txn.if %is_fetch {
      %pc_val = txn.call @pc::@read() : () -> i32
      
      // Multi-cycle memory fetch
      txn.future {
        // CHECK: txn.launch after 2
        %fetch_done = txn.launch after 2 {
          %instr = txn.call @mem::@read(%pc_val) : (i32) -> i32
          // Process instruction
          txn.call @executeInstruction(%instr) : (i32) -> ()
          txn.yield
        }
        txn.yield
      }
      
      // Update state
      txn.call @state::@write(%c1_i8) : (i8) -> ()
      txn.yield
    } else {
      %is_execute = arith.cmpi eq, %current_state, %c1_i8 : i8
      txn.if %is_execute {
        // Execute state - wait for completion
        %can_deq = txn.call @fifo::@canDeq() : () -> i1
        txn.if %can_deq {
          %result = txn.call @fifo::@deq() : () -> i32
          // Update PC
          %pc_val = txn.call @pc::@read() : () -> i32
          %c4 = arith.constant 4 : i32
          %next_pc = arith.addi %pc_val, %c4 : i32
          txn.call @pc::@write(%next_pc) : (i32) -> ()
          
          // Back to fetch
          txn.call @state::@write(%c0_i8) : (i8) -> ()
          txn.yield
        } else {
          // Stay in execute state
          txn.yield
        }
        txn.yield
      } else {
        // Invalid state - reset
        txn.call @state::@write(%c0_i8) : (i8) -> ()
        txn.yield
      }
      txn.yield
    }
    txn.yield
  }
  
  // Action with dynamic timing
  txn.action_method @dynamicOperation(%cond: i1) {
    txn.future {
      // Dynamic launch - waits until condition
      // CHECK: txn.launch until %{{.*}}
      %done = txn.launch until %cond {
        %c100 = arith.constant 100 : i32
        txn.call @fifo::@enq(%c100) : (i32) -> ()
        txn.yield
      }
      txn.yield
    }
    txn.yield
  }
  
  // Action with combined timing
  txn.action_method @combinedTiming(%cond: i1) {
    txn.future {
      // Wait for condition then 3 more cycles
      // CHECK: txn.launch until %{{.*}} after 3
      %done = txn.launch until %cond after 3 {
        %c200 = arith.constant 200 : i32
        txn.call @fifo::@enq(%c200) : (i32) -> ()
        txn.yield
      }
      txn.yield
    }
    txn.yield
  }
  
  // Nested multi-cycle operations
  txn.action_method @nestedLaunch() {
    txn.future {
      // Outer launch
      // CHECK: txn.launch after 1
      %outer_done = txn.launch after 1 {
        %data = txn.call @fifo::@deq() : () -> i32
        
        // Inner future with its own launch
        txn.future {
          // CHECK: txn.launch after 2
          %inner_done = txn.launch after 2 {
            %doubled = arith.muli %data, %data : i32
            txn.call @mem::@write(%data, %doubled) : (i32, i32) -> ()
            txn.yield
          }
          txn.yield
        }
        txn.yield
      }
      txn.yield
    }
    txn.yield
  }
  
  txn.schedule [@processor, @executeInstruction, @dynamicOperation, @combinedTiming, @nestedLaunch] {
    conflict_matrix = {
      "processor,processor" = 2 : i32,
      "executeInstruction,executeInstruction" = 2 : i32,
      "dynamicOperation,dynamicOperation" = 2 : i32,
      "combinedTiming,combinedTiming" = 2 : i32,
      "nestedLaunch,nestedLaunch" = 2 : i32,
      "processor,executeInstruction" = 0 : i32  // processor calls executeInstruction
    }
  }
}

// CHECK: txn.future
// CHECK: txn.launch