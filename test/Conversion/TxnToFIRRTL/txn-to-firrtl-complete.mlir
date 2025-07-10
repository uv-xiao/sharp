// RUN: sharp-opt %s --convert-txn-to-firrtl | FileCheck %s
// RUN: sharp-opt %s --convert-txn-to-firrtl --firrtl-lower-types --firrtl-imconstprop --firrtl-remove-unused-ports | FileCheck %s --check-prefix=OPT

// Comprehensive test for TxnToFIRRTL conversion covering all features

// Define primitives used in tests
txn.primitive @Register type = "hw" interface = !txn.module<"Register"> {
  txn.fir_value_method @read() : () -> i32
  txn.fir_action_method @write() : (i32) -> ()
  txn.clock_by @clk
  txn.reset_by @rst
  txn.schedule [@write] {
    conflict_matrix = {
      "read,read" = 3 : i32,
      "read,write" = 3 : i32,
      "write,read" = 3 : i32,
      "write,write" = 2 : i32
    }
  }
} {firrtl.impl = "Register_impl"}

txn.primitive @Wire type = "hw" interface = !txn.module<"Wire"> {
  txn.fir_value_method @read() : () -> i1
  txn.fir_action_method @write() : (i1) -> ()
  txn.clock_by @clk
  txn.reset_by @rst
  txn.schedule [@write] {
    conflict_matrix = {
      "read,read" = 3 : i32,
      "read,write" = 3 : i32,
      "write,read" = 3 : i32,
      "write,write" = 2 : i32
    }
  }
} {firrtl.impl = "Wire_impl"}

txn.primitive @FIFO type = "hw" interface = !txn.module<"FIFO"> {
  txn.fir_value_method @canEnq() : () -> i1
  txn.fir_value_method @canDeq() : () -> i1
  txn.fir_value_method @first() : () -> i32
  txn.fir_action_method @enq() : (i32) -> ()
  txn.fir_action_method @deq() : () -> ()
  txn.clock_by @clk
  txn.reset_by @rst
  txn.schedule [@enq, @deq] {
    conflict_matrix = {
      "canEnq,canEnq" = 3 : i32,
      "canEnq,enq" = 3 : i32,
      "canEnq,deq" = 3 : i32,
      "canDeq,canDeq" = 3 : i32,
      "canDeq,enq" = 3 : i32,
      "canDeq,deq" = 3 : i32,
      "first,first" = 3 : i32,
      "first,enq" = 3 : i32,
      "first,deq" = 3 : i32,
      "enq,enq" = 2 : i32,
      "enq,deq" = 2 : i32,
      "deq,deq" = 2 : i32
    }
  }
} {firrtl.impl = "FIFO_impl"}

txn.primitive @Memory type = "hw" interface = !txn.module<"Memory"> {
  txn.fir_value_method @read() : (i32) -> i32
  txn.fir_action_method @write() : (i32, i32) -> ()
  txn.clock_by @clk
  txn.reset_by @rst
  txn.schedule [@write] {
    conflict_matrix = {
      "read,read" = 3 : i32,
      "read,write" = 3 : i32,
      "write,read" = 3 : i32,
      "write,write" = 2 : i32
    }
  }
} {firrtl.impl = "Memory_impl"}

// CHECK-LABEL: firrtl.circuit "CompleteTxnToFIRRTL"
txn.module @CompleteTxnToFIRRTL {
  // Various primitive instances with different types
  %ctrl = txn.instance @ctrl of @Register<i8> : !txn.module<"Register">
  %data = txn.instance @data of @Register<i64> : !txn.module<"Register">
  %status = txn.instance @status of @Wire<i1> : !txn.module<"Wire">
  %fifo = txn.instance @fifo of @FIFO<i32> : !txn.module<"FIFO">
  %mem = txn.instance @mem of @Memory<i32> : !txn.module<"Memory">
  
  // CHECK: firrtl.module @CompleteTxnToFIRRTL
  // CHECK-DAG: %clock = firrtl.input "clock"
  // CHECK-DAG: %reset = firrtl.input "reset"
  
  // Value method with parameters - becomes combinational module
  // CHECK: firrtl.module private @getValue
  // CHECK-NOT: %clock
  // CHECK-NOT: %reset
  // CHECK: %base = firrtl.input "base"
  // CHECK: %offset = firrtl.input "offset"
  // CHECK: %result = firrtl.output "result"
  txn.value_method @getValue(%base: i32, %offset: i32) -> i32 attributes {combinational} {
    %sum = arith.addi %base, %offset : i32
    %c10 = arith.constant 10 : i32
    %scaled = arith.muli %sum, %c10 : i32
    txn.return %scaled : i32
  }
  
  // Action method with complex control flow and abort
  // CHECK: %processDataEN = firrtl.input "processDataEN"
  // CHECK: %processData_value = firrtl.input "processData_value"
  // CHECK: %processDataRDY = firrtl.output "processDataRDY"
  // CHECK: %processData_result = firrtl.output "processData_result"
  txn.action_method @processData(%value: i32) -> i32 attributes {
    signal_prefix = "proc",
    enable_suffix = "_en",
    ready_suffix = "_rdy",
    return_suffix = "_out"
  } {
    %c0 = arith.constant 0 : i32
    %c100 = arith.constant 100 : i32
    
    // Check bounds
    %is_zero = arith.cmpi eq, %value, %c0 : i32
    txn.if %is_zero {
      txn.abort  // Abort on zero
    } else {
      // Continue execution
    }
    
    %too_large = arith.cmpi sgt, %value, %c100 : i32
    txn.if %too_large {
      txn.abort  // Abort on overflow
    } else {
      // Continue execution
    }
    
    // Read control register
    %ctrl_val = txn.call @ctrl::@read() : () -> i8
    %c1_i8 = arith.constant 1 : i8
    %is_mode1 = arith.cmpi eq, %ctrl_val, %c1_i8 : i8
    
    %result = txn.if %is_mode1 -> i32 {
      // Mode 1: FIFO operation
      %can_enq = txn.call @fifo::@canEnq() : () -> i1
      %result = txn.if %can_enq -> i32 {
        txn.call @fifo::@enq(%value) : (i32) -> ()
        txn.yield %value : i32
      } else {
        // FIFO full, try memory
        %c0_addr = arith.constant 0 : i32
        txn.call @mem::@write(%c0_addr, %value) : (i32, i32) -> ()
        %negated = arith.subi %c0, %value : i32
        txn.yield %negated : i32
      }
      txn.yield %result : i32
    } else {
      // Mode 0: Direct register write
      %extended = arith.extsi %value : i32 to i64
      txn.call @data::@write(%extended) : (i64) -> ()
      txn.yield %value : i32
    }
    txn.return %result : i32
  }
  
  // Rule with complex guard and nested calls
  // CHECK: %complexRule_wf = firrtl.node
  // CHECK: firrtl.when %complexRule_wf
  txn.rule @complexRule {
    // Read multiple values
    %data_val = txn.call @data::@read() : () -> i64
    %can_deq = txn.call @fifo::@canDeq() : () -> i1
    
    %c0 = arith.constant 0 : i64
    %data_valid = arith.cmpi ne, %data_val, %c0 : i64
    %both_ready = arith.andi %data_valid, %can_deq : i1
    
    txn.if %both_ready {
      // Get FIFO data
      %fifo_val = txn.call @fifo::@first() : () -> i32
      txn.call @fifo::@deq() : () -> ()
      
      // Compute address
      %truncated = arith.trunci %data_val : i64 to i32
      %addr = txn.call @getValue(%truncated, %fifo_val) : (i32, i32) -> i32
      
      // Process data
      %result = txn.call @processData(%fifo_val) : (i32) -> i32
      
      // Write result
      %c255 = arith.constant 255 : i32
      %masked_addr = arith.andi %addr, %c255 : i32
      txn.call @mem::@write(%masked_addr, %result) : (i32, i32) -> ()
      
      // Update status
      %c_true = arith.constant true
      txn.call @status::@write(%c_true) : (i1) -> ()
      txn.yield
    } else {
      txn.yield
    }
    txn.yield
  }
  
  // Simple rule to test SB/SA relationships
  // CHECK: %producerRule_wf = firrtl.node
  txn.rule @producerRule {
    %c42 = arith.constant 42 : i32
    %extended = arith.extsi %c42 : i32 to i64
    txn.call @data::@write(%extended) : (i64) -> ()
    txn.yield
  }
  
  // CHECK: %consumerRule_wf = firrtl.node
  txn.rule @consumerRule {
    %val = txn.call @data::@read() : () -> i64
    %c0 = arith.constant 0 : i64
    %is_ready = arith.cmpi ne, %val, %c0 : i64
    
    txn.if %is_ready {
      %c_false = arith.constant false
      txn.call @status::@write(%c_false) : (i1) -> ()
      txn.yield
    } else {
      txn.yield
    }
    txn.yield
  }
  
  // Action method that calls value method
  // CHECK: %computeAction_value1 = firrtl.input
  // CHECK: %computeAction_value2 = firrtl.input
  // CHECK: %computeActionRDY = firrtl.output
  txn.action_method @computeAction(%value1: i32, %value2: i32) {
    %result = txn.call @getValue(%value1, %value2) : (i32, i32) -> i32
    
    // Store in memory
    %c10 = arith.constant 10 : i32
    txn.call @mem::@write(%c10, %result) : (i32, i32) -> ()
    txn.yield
  }
  
  // Schedule with complete conflict matrix
  txn.schedule [@processData, @complexRule, @producerRule, @consumerRule, @computeAction] {
    conflict_matrix = {
      // Self conflicts
      "processData,processData" = 2 : i32,
      "complexRule,complexRule" = 2 : i32,
      "producerRule,producerRule" = 2 : i32,
      "consumerRule,consumerRule" = 2 : i32,
      "computeAction,computeAction" = 2 : i32,
      
      // Sequential relationships
      "producerRule,consumerRule" = 0 : i32,  // SB: producer before consumer
      "consumerRule,producerRule" = 1 : i32,  // SA: consumer after producer
      
      // Conflicts
      "processData,complexRule" = 2 : i32,    // Both access FIFO and memory
      "complexRule,computeAction" = 2 : i32,  // Both access memory
      "processData,computeAction" = 2 : i32,  // Both access memory
      
      // Complex relationships
      "processData,producerRule" = 2 : i32,   // Both might write data
      "processData,consumerRule" = 2 : i32,   // processData reads ctrl, consumer reads data
      "complexRule,producerRule" = 2 : i32,   // Both access data register
      "complexRule,consumerRule" = 2 : i32,   // Both access data and status
      "producerRule,computeAction" = 3 : i32, // CF: Different resources
      "consumerRule,computeAction" = 3 : i32  // CF: Different resources
    }
  }
}

// Check will-fire generation
// CHECK: %processData_will_fire = firrtl.and %processDataEN
// CHECK: %complexRule_conflict_free = firrtl.and
// CHECK: %producerRule_conflict_free = firrtl.and
// CHECK: %consumerRule_conflict_free = firrtl.and

// Check reach_abort calculation
// CHECK: %processData_reach_abort = firrtl.or
// CHECK: %complexRule_reach_abort = firrtl.or

// Check conflict_inside for rules
// CHECK: firrtl.when %complexRule_wf
// CHECK: %complexRule_processData_conflict_inside = firrtl.or

// OPT-LABEL: firrtl.module @CompleteTxnToFIRRTL
// OPT: firrtl.connect

// Test instance generation
// CHECK-LABEL: firrtl.module @WithInstances
txn.module @WithInstances {
  %sub = txn.instance @sub of @SubModule : !txn.module<"SubModule">
  
  // CHECK: firrtl.instance sub @SubModule
  // CHECK: firrtl.connect %sub.clock, %clock
  // CHECK: firrtl.connect %sub.reset, %reset
  
  txn.action_method @callSub(%x: i32) -> i32 {
    // CHECK: firrtl.connect %sub.processEN
    // CHECK: firrtl.connect %sub.process_value
    %result = txn.call @sub::@process(%x) : (i32) -> i32
    txn.return %result : i32
  }
  
  txn.schedule [@callSub] {
    conflict_matrix = {
      "callSub,callSub" = 2 : i32
    }
  }
}

// CHECK-LABEL: firrtl.module private @SubModule
txn.module @SubModule {
  %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
  
  // CHECK: %processEN = firrtl.input
  // CHECK: %process_value = firrtl.input
  // CHECK: %processRDY = firrtl.output
  // CHECK: %process_result = firrtl.output
  txn.action_method @process(%value: i32) -> i32 {
    txn.call @reg::@write(%value) : (i32) -> ()
    %doubled = arith.muli %value, %value : i32
    txn.return %doubled : i32
  }
  
  txn.schedule [@process] {
    conflict_matrix = {
      "process,process" = 2 : i32
    }
  }
}