// RUN: sharp-opt %s -allow-unregistered-dialect | FileCheck %s
// RUN: not sharp-opt %s -allow-unregistered-dialect --verify-diagnostics 2>&1 | FileCheck %s --check-prefix=ERROR

// Comprehensive test for control flow edge cases and complex guard conditions

// CHECK-LABEL: txn.module @NestedControlFlow
txn.module @NestedControlFlow {
  %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
  %wire = txn.instance @wire of @Wire<i32> : !txn.module<"Wire">
  %fifo = txn.instance @fifo of @FIFO<i32> : !txn.module<"FIFO">
  
  // Deep nesting with multiple abort paths
  txn.action_method @deeplyNested(%a: i32, %b: i32, %c: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c100 = arith.constant 100 : i32
    
    // First level: check a > 0
    %a_positive = arith.cmpi sgt, %a, %c0 : i32
    txn.if %a_positive {
      // Second level: check b < 10
      %b_small = arith.cmpi slt, %b, %c10 : i32
      txn.if %b_small {
        // Third level: check c != 0
        %c_nonzero = arith.cmpi ne, %c, %c0 : i32
        txn.if %c_nonzero {
          // Deeply nested computation
          %sum = arith.addi %a, %b : i32
          %result = arith.divsi %sum, %c : i32
          
          // Fourth level: bounds check
          %in_bounds = arith.cmpi slt, %result, %c100 : i32
          txn.if %in_bounds {
            txn.call @reg::@write(%result) : (i32) -> ()
            txn.return %result : i32
          } else {
            // Overflow abort
            txn.abort
          }
        } else {
          // Division by zero abort
          txn.abort
        }
      } else {
        // Try alternative path
        %can_enq = txn.call @fifo::@canEnq() : () -> i1
        txn.if %can_enq {
          txn.call @fifo::@enq(%b) : (i32) -> ()
          txn.return %b : i32
        } else {
          // FIFO full abort
          txn.abort
        }
      }
    } else {
      // Negative input abort
      txn.abort
    }
  }
  
  // Multiple abort paths with different conditions
  txn.rule @complexAbortLogic {
    %val = txn.call @reg::@read() : () -> i32
    %c0 = arith.constant 0 : i32
    %c5 = arith.constant 5 : i32
    %c10 = arith.constant 10 : i32
    %c15 = arith.constant 15 : i32
    
    // Multiple conditions that could lead to abort
    %is_zero = arith.cmpi eq, %val, %c0 : i32
    %is_five = arith.cmpi eq, %val, %c5 : i32
    %is_ten = arith.cmpi eq, %val, %c10 : i32
    %is_fifteen = arith.cmpi eq, %val, %c15 : i32
    
    // Complex abort logic
    txn.if %is_zero {
      txn.abort  // Abort on zero
    } else {
      txn.if %is_five {
        // Try recovery
        %can_deq = txn.call @fifo::@canDeq() : () -> i1
        txn.if %can_deq {
          %data = txn.call @fifo::@first() : () -> i32
          txn.call @fifo::@deq() : () -> ()
          txn.call @wire::@write(%data) : (i32) -> ()
        } else {
          txn.abort  // Abort if can't recover
        }
      } else {
        txn.if %is_ten {
          // Conditional abort based on wire value
          %wire_val = txn.call @wire::@read() : () -> i32
          %wire_ok = arith.cmpi sgt, %wire_val, %c0 : i32
          txn.if %wire_ok {
            txn.call @reg::@write(%wire_val) : (i32) -> ()
          } else {
            txn.abort  // Abort on invalid wire value
          }
        } else {
          txn.if %is_fifteen {
            // Always succeed for fifteen
            %c20 = arith.constant 20 : i32
            txn.call @reg::@write(%c20) : (i32) -> ()
          } else {
            // Default case - complex computation
            %doubled = arith.muli %val, %c5 : i32
            %in_range = arith.cmpi slt, %doubled, %c100 : i32
            txn.if %in_range {
              txn.call @wire::@write(%doubled) : (i32) -> ()
            } else {
              txn.abort  // Abort on overflow
            }
          }
        }
      }
    }
    txn.yield
  }
  
  // Nested method calls with conditional aborts
  txn.action_method @callChain(%depth: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    
    %at_bottom = arith.cmpi eq, %depth, %c0 : i32
    txn.if %at_bottom {
      // Base case - might abort
      %val = txn.call @reg::@read() : () -> i32
      %is_valid = arith.cmpi sgt, %val, %c0 : i32
      txn.if %is_valid {
        txn.return %val : i32
      } else {
        txn.abort
      }
    } else {
      // Recursive case
      %next_depth = arith.subi %depth, %c1 : i32
      %result = txn.call @callChain(%next_depth) : (i32) -> i32
      
      // Might abort based on recursive result
      %c10 = arith.constant 10 : i32
      %too_large = arith.cmpi sgt, %result, %c10 : i32
      txn.if %too_large {
        txn.abort
      } else {
        %incremented = arith.addi %result, %c1 : i32
        txn.return %incremented : i32
      }
    }
    } else {
      // Continue execution
    }
  }
  
  // Complex guard conditions
  txn.rule @guardedRule {
    %v1 = txn.call @reg::@read() : () -> i32
    %v2 = txn.call @wire::@read() : () -> i32
    %can_deq = txn.call @fifo::@canDeq() : () -> i1
    
    // Complex guard: (v1 > v2) && ((v1 + v2 < 100) || can_deq)
    %c100 = arith.constant 100 : i32
    %v1_gt_v2 = arith.cmpi sgt, %v1, %v2 : i32
    %sum = arith.addi %v1, %v2 : i32
    %sum_ok = arith.cmpi slt, %sum, %c100 : i32
    %sum_or_fifo = arith.ori %sum_ok, %can_deq : i1
    %guard = arith.andi %v1_gt_v2, %sum_or_fifo : i1
    
    txn.if %guard {
      // Complex action with multiple branches
      txn.if %can_deq {
        %data = txn.call @fifo::@first() : () -> i32
        txn.call @fifo::@deq() : () -> ()
        %new_val = arith.addi %v1, %data : i32
        txn.call @reg::@write(%new_val) : (i32) -> ()
      } else {
        // Alternative path
        %diff = arith.subi %v1, %v2 : i32
        txn.call @wire::@write(%diff) : (i32) -> ()
      }
    } else {
      // Guard not satisfied
    }
    txn.yield
  }
  
  txn.schedule [@deeplyNested, @complexAbortLogic, @callChain, @guardedRule] {
    conflict_matrix = {
      "deeplyNested,deeplyNested" = 2 : i32,
      "complexAbortLogic,complexAbortLogic" = 2 : i32,
      "callChain,callChain" = 2 : i32,
      "guardedRule,guardedRule" = 2 : i32,
      "deeplyNested,complexAbortLogic" = 2 : i32,  // Both access reg
      "deeplyNested,callChain" = 2 : i32,         // Both access reg
      "complexAbortLogic,guardedRule" = 2 : i32,  // Both access multiple resources
      "callChain,guardedRule" = 2 : i32,          // Both access reg
      "deeplyNested,guardedRule" = 2 : i32,       // Both access reg
      "complexAbortLogic,callChain" = 2 : i32     // Both access reg
    }
  }
}

// Test error cases for control flow
// ERROR-LABEL: txn.module @ControlFlowErrors
txn.module @ControlFlowErrors {
  // ERROR: txn.action_method @unreachableCode
  txn.action_method @unreachableCode() -> i32 {
    txn.abort
    // expected-warning@+1 {{unreachable code after abort}}
    %c0 = arith.constant 0 : i32
    txn.return %c0 : i32
  }
  
  // ERROR: txn.action_method @missingElseBranch
  txn.action_method @missingElseBranch(%cond: i1) -> i32 {
    txn.if %cond {
      %c1 = arith.constant 1 : i32
      txn.return %c1 : i32
    }
    // expected-error@+1 {{action method must return or abort on all paths}}
  }
  
  // ERROR: txn.rule @abortInRule
  txn.rule @abortInRule {
    // Rules should use txn.yield, not txn.abort
    // expected-error@+1 {{rules cannot use abort}}
    txn.abort
  }
}