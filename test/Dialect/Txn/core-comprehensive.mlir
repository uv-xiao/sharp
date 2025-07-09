// RUN: sharp-opt %s -allow-unregistered-dialect | FileCheck %s
// RUN: not sharp-opt %s -allow-unregistered-dialect --verify-diagnostics 2>&1 | FileCheck %s --check-prefix=ERROR

// Comprehensive test for all core Txn dialect features in realistic scenarios

// CHECK-LABEL: txn.module @ProcessorCore
txn.module @ProcessorCore {
  // Parametric primitive instantiation with type arguments
  %pc = txn.instance @pc of @Register<i32> : !txn.module<"Register">
  %ir = txn.instance @ir of @Register<i32> : !txn.module<"Register">
  %alu_result = txn.instance @alu_result of @Wire<i32> : !txn.module<"Wire">
  %mem = txn.instance @mem of @Memory<i32> : !txn.module<"Memory">
  
  // Complex value method with multiple primitive calls
  txn.value_method @decode() -> (i32, i32, i32) attributes {combinational} {
    %instr = txn.call @ir::@read() : () -> i32
    
    // Extract opcode, rs1, rs2 using bit operations
    %c0 = arith.constant 0 : i32
    %c5 = arith.constant 5 : i32
    %c10 = arith.constant 10 : i32
    %c15 = arith.constant 15 : i32
    %mask5 = arith.constant 31 : i32  // 0x1F
    
    %opcode = arith.shrui %instr, %c0 : i32
    %opcode_masked = arith.andi %opcode, %mask5 : i32
    
    %rs1_shifted = arith.shrui %instr, %c5 : i32
    %rs1 = arith.andi %rs1_shifted, %mask5 : i32
    
    %rs2_shifted = arith.shrui %instr, %c10 : i32
    %rs2 = arith.andi %rs2_shifted, %mask5 : i32
    
    txn.return %opcode_masked, %rs1, %rs2 : i32, i32, i32
  }
  
  // Action method with conditional logic and abort
  txn.action_method @execute(%enable: i1) attributes {
    signal_prefix = "exec", 
    enable_suffix = "_en",
    ready_suffix = "_rdy"
  } {
    txn.if %enable {
      %opcode, %rs1, %rs2 = txn.call @decode() : () -> (i32, i32, i32)
      
      // Check for illegal instruction
      %c31 = arith.constant 31 : i32
      %is_illegal = arith.cmpi eq, %opcode, %c31 : i32
      txn.if %is_illegal {
        txn.abort  // Abort on illegal instruction
      } else {
        // ALU operation
        %c0 = arith.constant 0 : i32
        %is_add = arith.cmpi eq, %opcode, %c0 : i32
        txn.if %is_add {
          %a = txn.call @mem::@read(%rs1) : (i32) -> i32
          %b = txn.call @mem::@read(%rs2) : (i32) -> i32
          %sum = arith.addi %a, %b : i32
          txn.call @alu_result::@write(%sum) : (i32) -> ()
        } else {
          // Other opcodes not implemented
        }
      }
    } else {
      // Not enabled
    }
    txn.yield
  }
  
  // Rule with complex guard condition
  txn.rule @fetch {
    %pc_val = txn.call @pc::@read() : () -> i32
    %instr = txn.call @mem::@read(%pc_val) : (i32) -> i32
    
    // Check if instruction is valid (non-zero)
    %c0 = arith.constant 0 : i32
    %is_valid = arith.cmpi ne, %instr, %c0 : i32
    
    txn.if %is_valid {
      txn.call @ir::@write(%instr) : (i32) -> ()
      
      // Increment PC
      %c4 = arith.constant 4 : i32
      %next_pc = arith.addi %pc_val, %c4 : i32
      txn.call @pc::@write(%next_pc) : (i32) -> ()
    } else {
      // Halt on invalid instruction
      txn.abort
    }
    txn.yield
  }
  
  // Schedule with complex conflict matrix
  txn.schedule [@execute, @fetch] {
    conflict_matrix = {
      "execute,execute" = 2 : i32,  // C: Can't execute twice
      "fetch,fetch" = 2 : i32,      // C: Can't fetch twice
      "execute,fetch" = 0 : i32     // SB: Execute before fetch
    }
  }
}

// CHECK-LABEL: txn.module @FIFONetwork
txn.module @FIFONetwork {
  // Network of FIFOs with different types
  %data_fifo = txn.instance @data_fifo of @FIFO<i64> : !txn.module<"FIFO">
  %ctrl_fifo = txn.instance @ctrl_fifo of @FIFO<i1> : !txn.module<"FIFO">
  %status = txn.instance @status of @Register<i8> : !txn.module<"Register">
  
  // Producer rule with multi-cycle behavior
  txn.rule @producer {
    %can_enq = txn.call @data_fifo::@canEnq() : () -> i1
    %ctrl = txn.call @ctrl_fifo::@first() : () -> i1
    
    %both_ready = arith.andi %can_enq, %ctrl : i1
    
    txn.if %both_ready {
      // Generate data based on control
      %c42 = arith.constant 42 : i64
      %c100 = arith.constant 100 : i64
      %data = arith.select %ctrl, %c100, %c42 : i64
      
      // Enqueue and consume control
      txn.call @data_fifo::@enq(%data) : (i64) -> ()
      txn.call @ctrl_fifo::@deq() : () -> ()
      
      // Update status
      %c1 = arith.constant 1 : i8
      %old_status = txn.call @status::@read() : () -> i8
      %new_status = arith.addi %old_status, %c1 : i8
      txn.call @status::@write(%new_status) : (i8) -> ()
    } else {
      // Not ready
    }
    txn.yield
  }
  
  // Consumer with error handling
  txn.action_method @consume() -> i64 {
    %can_deq = txn.call @data_fifo::@canDeq() : () -> i1
    txn.if %can_deq {
      %data = txn.call @data_fifo::@first() : () -> i64
      txn.call @data_fifo::@deq() : () -> ()
      txn.return %data : i64
    } else {
      // Return error value if FIFO empty
      %error = arith.constant -1 : i64
      txn.return %error : i64
    }
  }
  
  txn.schedule [@producer, @consume] {
    conflict_matrix = {
      "producer,producer" = 2 : i32,
      "consume,consume" = 2 : i32,
      "producer,consume" = 3 : i32  // CF: Can run concurrently
    }
  }
}

// Test error cases
// ERROR-LABEL: txn.module @ErrorCases
txn.module @ErrorCases {
  %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
  
  // ERROR: txn.value_method @invalid_abort
  txn.value_method @invalid_abort() -> i32 {
    // expected-error@+1 {{abort not allowed in value method}}
    txn.abort
    %c0 = arith.constant 0 : i32
    txn.return %c0 : i32
  }
  
  // ERROR: txn.action_method @missing_yield
  txn.action_method @missing_yield() {
    %c0 = arith.constant 0 : i32
    // expected-error@+1 {{expected 'txn.yield' at end of action method}}
  }
  
  // ERROR: txn.schedule [@invalid_abort, @missing_yield]
  txn.schedule [@invalid_abort, @missing_yield] {
    // expected-error@+1 {{value method 'invalid_abort' cannot be in schedule}}
  }
}