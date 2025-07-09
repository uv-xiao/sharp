// RUN: sharp-opt %s --sharp-reachability-analysis | FileCheck %s
// RUN: sharp-opt %s --sharp-reachability-analysis --convert-txn-to-firrtl | FileCheck %s --check-prefix=FIRRTL

// Comprehensive test for reachability analysis with deep nesting and conditional aborts

// CHECK-LABEL: txn.module @DeepReachability
txn.module @DeepReachability {
  %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
  %fifo = txn.instance @fifo of @FIFO<i32> : !txn.module<"FIFO">
  %wire = txn.instance @wire of @Wire<i1> : !txn.module<"Wire">
  %mem = txn.instance @mem of @Memory<i32> : !txn.module<"Memory">
  
  // Action method with conditional calls and aborts (no return value)
  txn.action_method @level1(%x: i32) {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    
    %is_zero = arith.cmpi eq, %x, %c0 : i32
    txn.if %is_zero {
      txn.abort  // Abort path 1
    } else {
      %is_small = arith.cmpi slt, %x, %c10 : i32
      txn.if %is_small {
        // CHECK: txn.call @level2(%{{.*}}) guard(%{{.*}})
        txn.call @level2(%x) : (i32) -> ()
      } else {
        // CHECK: txn.call @level3(%{{.*}}) guard(%{{.*}})
        txn.call @level3(%x) : (i32) -> ()
      }
    }
    txn.yield
  }
  
  txn.action_method @level2(%y: i32) {
    %c5 = arith.constant 5 : i32
    %is_five = arith.cmpi eq, %y, %c5 : i32
    
    txn.if %is_five {
      // CHECK: txn.call @fifo::@canEnq() guard(%{{.*}})
      %can_enq = txn.call @fifo::@canEnq() : () -> i1
      txn.if %can_enq {
        // CHECK: txn.call @fifo::@enq(%{{.*}}) guard(%{{.*}})
        txn.call @fifo::@enq(%y) : (i32) -> ()
        txn.call @reg::@write(%y) : (i32) -> ()
      } else {
        txn.abort  // Abort path 2
      }
    } else {
      // CHECK: txn.call @level4(%{{.*}}) guard(%{{.*}})
      %doubled = arith.muli %y, %c5 : i32
      txn.call @level4(%doubled) : (i32) -> ()
    }
    txn.yield
  }
  
  txn.action_method @level3(%z: i32) {
    // CHECK: txn.call @wire::@read() guard(%{{.*}})
    %flag = txn.call @wire::@read() : () -> i1
    
    txn.if %flag {
      // CHECK: txn.call @reg::@read() guard(%{{.*}})
      %old_val = txn.call @reg::@read() : () -> i32
      %sum = arith.addi %old_val, %z : i32
      
      %c100 = arith.constant 100 : i32
      %overflow = arith.cmpi sgt, %sum, %c100 : i32
      
      txn.if %overflow {
        txn.abort  // Abort path 3
      } else {
        // CHECK: txn.call @reg::@write(%{{.*}}) guard(%{{.*}})
        txn.call @reg::@write(%sum) : (i32) -> ()
      }
    } else {
      // No flag - do nothing
    }
    txn.yield
  }
  
  txn.action_method @level4(%w: i32) {
    %c0 = arith.constant 0 : i32
    %c3 = arith.constant 3 : i32
    
    // Modulo check
    %remainder = arith.remsi %w, %c3 : i32
    %is_multiple = arith.cmpi eq, %remainder, %c0 : i32
    
    txn.if %is_multiple {
      // CHECK: txn.call @mem::@write(%{{.*}}, %{{.*}}) guard(%{{.*}})
      txn.call @mem::@write(%c0, %w) : (i32, i32) -> ()
    } else {
      // Recursive call with guard
      %c1 = arith.constant 1 : i32
      %next = arith.subi %w, %c1 : i32
      
      %is_positive = arith.cmpi sgt, %next, %c0 : i32
      txn.if %is_positive {
        // CHECK: txn.call @level4(%{{.*}}) guard(%{{.*}})
        txn.call @level4(%next) : (i32) -> ()
      } else {
        txn.abort  // Abort path 4
      }
    }
    txn.yield
  }
  
  // Value method to check state
  txn.value_method @checkState() -> i1 {
    %val = txn.call @reg::@read() : () -> i32
    %c50 = arith.constant 50 : i32
    %is_ready = arith.cmpi slt, %val, %c50 : i32
    txn.return %is_ready : i1
  }
  
  // Rule with nested calls that may abort
  txn.rule @complexChain {
    // CHECK: txn.call @checkState() guard(%{{.*}})
    %ready = txn.call @checkState() : () -> i1
    
    txn.if %ready {
      %c7 = arith.constant 7 : i32
      // CHECK: txn.call @level1(%{{.*}}) guard(%{{.*}})
      txn.call @level1(%c7) : (i32) -> ()
      
      // If we reach here, no abort occurred
      %c1 = arith.constant true
      // CHECK: txn.call @wire::@write(%{{.*}}) guard(%{{.*}})
      txn.call @wire::@write(%c1) : (i1) -> ()
    } else {
      // Not ready
    }
    txn.yield
  }
  
  // Another rule with abort conditions
  txn.rule @abortingRule {
    // CHECK: txn.call @fifo::@canDeq() guard(%{{.*}})
    %can_deq = txn.call @fifo::@canDeq() : () -> i1
    
    txn.if %can_deq {
      // CHECK: txn.call @fifo::@deq() guard(%{{.*}})
      %data = txn.call @fifo::@deq() : () -> i32
      
      %c15 = arith.constant 15 : i32
      %is_special = arith.cmpi eq, %data, %c15 : i32
      
      txn.if %is_special {
        txn.abort  // Special value causes abort
      } else {
        // CHECK: txn.call @level1(%{{.*}}) guard(%{{.*}})
        txn.call @level1(%data) : (i32) -> ()
      }
    } else {
      // Nothing to dequeue
    }
    txn.yield
  }
  
  txn.schedule [@complexChain, @abortingRule] {
    conflict_matrix = {
      "complexChain,complexChain" = 2 : i32,
      "abortingRule,abortingRule" = 2 : i32,
      "complexChain,abortingRule" = 2 : i32  // Both may access reg/fifo
    }
  }
}

// FIRRTL: firrtl.module @DeepReachability
// FIRRTL: %reach_abort = firrtl.or