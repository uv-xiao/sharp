// RUN: sharp-opt %s --convert-txn-to-firrtl | FileCheck %s
// RUN: sharp-opt %s --convert-txn-to-func | FileCheck %s --check-prefix=FUNC

// Comprehensive test for abort propagation through multiple levels

// Define primitives used in tests
txn.primitive @Register type = "hw" interface = index {
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

txn.primitive @FIFO type = "hw" interface = index {
  txn.fir_value_method @canEnq() : () -> i1
  txn.fir_action_method @enq() : (i32) -> ()
  txn.clock_by @clk
  txn.reset_by @rst
  txn.schedule [@enq] {
    conflict_matrix = {
      "canEnq,canEnq" = 3 : i32,
      "canEnq,enq" = 3 : i32,
      "enq,canEnq" = 3 : i32,
      "enq,enq" = 2 : i32
    }
  }
} {firrtl.impl = "FIFO_impl"}

txn.primitive @Memory type = "hw" interface = index {
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

txn.module @AbortPropagation {
  %reg = txn.instance @reg of @Register<i32> : index
  %reg2 = txn.instance @reg2 of @Register<i32> : index
  %wire = txn.instance @wire of @Wire<i32> : index
  
  // Deepest level - multiple abort conditions
  txn.action_method @level3(%x: i32, %y: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    
    // Check x != 0
    %x_zero = arith.cmpi eq, %x, %c0 : i32
    txn.if %x_zero {
      txn.abort  // Abort path 1
    } else {
      // Continue
    }
    
    // Check y != 0
    %y_zero = arith.cmpi eq, %y, %c0 : i32
    txn.if %y_zero {
      txn.abort  // Abort path 2
    } else {
      // Continue
    }
    
    // Safe division
    %quotient = arith.divsi %x, %y : i32
    
    // Check result bounds
    %c100 = arith.constant 100 : i32
    %too_large = arith.cmpi sgt, %quotient, %c100 : i32
    txn.if %too_large {
      txn.abort  // Abort path 3
    } else {
      // Continue
    }
    
    txn.return %quotient : i32
  }
  
  // Middle level - conditional calls to level3
  txn.action_method @level2(%a: i32) -> i32 {
    %c10 = arith.constant 10 : i32
    %is_small = arith.cmpi slt, %a, %c10 : i32
    
    %result = txn.if %is_small -> i32 {
      // Path 1: might abort through level3
      %c2 = arith.constant 2 : i32
      %result = txn.call @level3(%a, %c2) : (i32, i32) -> i32
      txn.yield %result : i32
    } else {
      // Path 2: different parameters to level3
      %c20 = arith.constant 20 : i32
      %is_medium = arith.cmpi slt, %a, %c20 : i32
      
      %result = txn.if %is_medium -> i32 {
        %result = txn.call @level3(%c20, %a) : (i32, i32) -> i32
        txn.yield %result : i32
      } else {
        // Path 3: direct abort
        txn.abort
      }
      txn.yield %result : i32
    }
    txn.return %result : i32
  }
  
  // Top level - multiple paths with different abort conditions
  txn.action_method @level1(%input: i32) {
    // Check wire status first
    %can_enq = txn.call @wire::@read() : () -> i1
    %not_can_enq = arith.xori %can_enq, %can_enq : i1
    txn.if %not_can_enq {
      txn.abort  // Abort if wire is false
    } else {
      // Continue
    }
    
    // Complex control flow
    %c0 = arith.constant 0 : i32
    %c5 = arith.constant 5 : i32
    %c15 = arith.constant 15 : i32
    
    %is_negative = arith.cmpi slt, %input, %c0 : i32
    txn.if %is_negative {
      // Negative path - always aborts
      txn.abort
    } else {
      %is_small = arith.cmpi slt, %input, %c5 : i32
      txn.if %is_small {
        // Small values - process directly
        txn.call @reg2::@write(%input) : (i32) -> ()
        txn.yield
      } else {
        %is_medium = arith.cmpi slt, %input, %c15 : i32
        txn.if %is_medium {
          // Medium values - call level2 (might abort)
          %result = txn.call @level2(%input) : (i32) -> i32
          txn.call @reg2::@write(%result) : (i32) -> ()
          txn.yield
        } else {
          // Large values - complex processing
          %processed = txn.call @complexProcess(%input) : (i32) -> i32
          txn.call @reg2::@write(%processed) : (i32) -> ()
          txn.yield
        }
        txn.yield
      }
      txn.yield
    }
    txn.yield
  }
  
  // Method with abort in loop-like recursion
  txn.action_method @recursiveAbort(%n: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c10 = arith.constant 10 : i32
    
    // Base case
    %is_zero = arith.cmpi eq, %n, %c0 : i32
    txn.if %is_zero {
      txn.abort
    } else {
      // Continue
      txn.yield
    }
    
    // Check bounds
    %too_large = arith.cmpi sgt, %n, %c10 : i32
    txn.if %too_large {
      txn.abort  // Abort on overflow
    } else {
      // Continue
      txn.yield
    }
    
    // Recursive call
    %n_minus_1 = arith.subi %n, %c1 : i32
    %rec_result = txn.call @recursiveAbort(%n_minus_1) : (i32) -> i32
    
    // Post-processing that might abort
    %sum = arith.addi %rec_result, %n : i32
    %c50 = arith.constant 50 : i32
    %sum_overflow = arith.cmpi sgt, %sum, %c50 : i32
    txn.if %sum_overflow {
      txn.abort  // Abort on sum overflow
    } else {
      // Continue
      txn.yield
    }
    
    txn.return %sum : i32
  }
  
  // Complex processing with multiple abort paths
  txn.action_method @complexProcess(%val: i32) -> i32 {
    // Read from memory
    %c0 = arith.constant 0 : i32
    %addr = arith.andi %val, %c0 : i32
    %mem_val = txn.call @wire::@read() : () -> i32
    
    // Check memory value
    %mem_valid = arith.cmpi ne, %mem_val, %c0 : i32
    %result = txn.if %mem_valid -> i32 {
      // Try recursive processing
      %c5 = arith.constant 5 : i32
      %small_val = arith.remsi %val, %c5 : i32
      %result = txn.call @recursiveAbort(%small_val) : (i32) -> i32
      
      // Final validation
      %c3 = arith.constant 3 : i32
      %is_multiple_of_3 = arith.remsi %result, %c3 : i32
      %not_multiple = arith.cmpi ne, %is_multiple_of_3, %c0 : i32
      txn.if %not_multiple {
        txn.abort  // Abort if not multiple of 3
      } else {
        // Continue
        txn.yield
      }
      
      txn.yield %result : i32
    } else {
      txn.abort  // Abort on invalid memory
    }
    txn.return %result : i32
  }
  
  // Rule that calls methods with abort
  txn.rule @abortingRule {
    %reg_val = txn.call @reg::@read() : () -> i32
    
    %c0 = arith.constant 0 : i32
    %c30 = arith.constant 30 : i32
    
    %in_range = arith.cmpi slt, %reg_val, %c30 : i32
    %positive = arith.cmpi sgt, %reg_val, %c0 : i32
    %valid = arith.andi %in_range, %positive : i1
    
    txn.if %valid {
      // This call chain can abort in many ways
      txn.call @level1(%reg_val) : (i32) -> ()
      
      // Only executed if level1 doesn't abort
      %c1 = arith.constant 1 : i32
      %inc = arith.addi %reg_val, %c1 : i32
      txn.call @reg::@write(%inc) : (i32) -> ()
      txn.yield
    } else {
      // Continue
      txn.yield
    }
    txn.yield
  }
  
  txn.schedule [@level1, @level2, @level3, @recursiveAbort, @complexProcess, @abortingRule] {
    conflict_matrix = {
      // All self-conflicts
      "level1,level1" = 2 : i32,
      "level2,level2" = 2 : i32,
      "level3,level3" = 2 : i32,
      "recursiveAbort,recursiveAbort" = 2 : i32,
      "complexProcess,complexProcess" = 2 : i32,
      "abortingRule,abortingRule" = 2 : i32,
      
      // Method calls create conflicts
      "level1,level2" = 2 : i32,
      "level1,level3" = 2 : i32,
      "level2,level3" = 2 : i32,
      "complexProcess,recursiveAbort" = 2 : i32,
      "abortingRule,level1" = 2 : i32,
      
      // Resource conflicts
      "level1,abortingRule" = 2 : i32,  // Both use FIFO
      "complexProcess,abortingRule" = 2 : i32  // Both use memory/reg
    }
  }
}

// CHECK-LABEL: firrtl.module @AbortPropagation

// Check reach_abort calculation for each method
// CHECK: %level3_reach_abort = firrtl.or
// CHECK: %level2_reach_abort = firrtl.or
// CHECK: %level1_reach_abort = firrtl.or
// CHECK: %recursiveAbort_reach_abort = firrtl.or
// CHECK: %complexProcess_reach_abort = firrtl.or

// Check will-fire includes abort conditions
// CHECK: %level1_no_abort = firrtl.not %level1_reach_abort
// CHECK: %level1_can_fire = firrtl.and %level1_guard, %level1_no_abort
// CHECK: %level1_will_fire = firrtl.and %level1EN, %level1_can_fire

// Check abort propagation in rule
// CHECK: %abortingRule_reach_abort = firrtl.or
// CHECK: %abortingRule_will_fire = firrtl.and

// FUNC mode checks abort propagation in functional conversion
// FUNC-LABEL: func.func @level3(%arg0: i32, %arg1: i32) -> (i32, i1)
// FUNC: %[[ABORT1:.*]] = arith.cmpi eq, %arg0
// FUNC: %[[ABORT2:.*]] = arith.cmpi eq, %arg1
// FUNC: %[[ABORT3:.*]] = arith.cmpi sgt
// FUNC: %[[ANY_ABORT:.*]] = arith.ori %[[ABORT1]], %[[ABORT2]]
// FUNC: %[[FINAL_ABORT:.*]] = arith.ori %[[ANY_ABORT]], %[[ABORT3]]
// FUNC: return %{{.*}}, %[[FINAL_ABORT]]

// FUNC-LABEL: func.func @level2(%arg0: i32) -> (i32, i1)
// FUNC: %[[CALL_RESULT:.*]]:2 = call @level3
// FUNC: %[[ABORT_PROP:.*]] = arith.ori
// FUNC: return %{{.*}}, %[[ABORT_PROP]]

// FUNC-LABEL: func.func @level1(%arg0: i32) -> i1
// FUNC: %[[CALL2:.*]]:2 = call @level2
// FUNC: %[[CALL_COMPLEX:.*]]:2 = call @complexProcess
// FUNC: %[[ABORT_FINAL:.*]] = arith.ori
// FUNC: return %[[ABORT_FINAL]]