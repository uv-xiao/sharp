// RUN: sharp-opt %s --convert-txn-to-firrtl | FileCheck %s --check-prefix=DYNAMIC
// RUN: sharp-opt %s --convert-txn-to-firrtl=mode=static | FileCheck %s --check-prefix=STATIC

// Comprehensive test for will-fire generation in all modes

txn.module @WillFireModes {
  %reg1 = txn.instance @reg1 of @Register<i32> : index
  %reg2 = txn.instance @reg2 of @Register<i32> : index
  %wire = txn.instance @wire of @Wire<i1> : index
  
  // Action with conditional abort - tests reach_abort calculation
  txn.action_method @conditionalAbort(%x: i32) {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    
    // First check - abort if zero
    %is_zero = arith.cmpi eq, %x, %c0 : i32
    txn.if %is_zero {
      txn.abort  // First abort condition
    } else {
      // Continue execution
    }
    
    // If we reach here, x is not zero
    %too_large = arith.cmpi sgt, %x, %c10 : i32
    txn.if %too_large {
      // Value is > 10, write to reg1
      txn.call @reg1::@write(%x) : (i32) -> ()
      txn.yield
    } else {
      // Value is in range [1, 10], write to reg2
      txn.call @reg2::@write(%x) : (i32) -> ()
      txn.yield
    }
    txn.yield
  }
  
  // Another action with abort based on wire value
  txn.action_method @wireAbort(%y: i32) {
    %flag = txn.call @wire::@read() : () -> i1
    txn.if %flag {
      txn.call @reg1::@write(%y) : (i32) -> ()
      txn.yield
    } else {
      txn.abort  // Abort if flag is false
    }
    txn.yield
  }
  
  // Rule with guard
  txn.rule @guardedRule {
    %val1 = txn.call @reg1::@read() : () -> i32
    %val2 = txn.call @reg2::@read() : () -> i32
    %guard = txn.call @wire::@read() : () -> i1
    
    // Simple guard check
    txn.if %guard {
      %sum = arith.addi %val1, %val2 : i32
      txn.call @reg2::@write(%sum) : (i32) -> ()
      txn.yield
    } else {
      // Do nothing if wire is false
      txn.yield
    }
    txn.yield
  }
  
  // Rule with abort inside
  txn.rule @abortingRule {
    %val = txn.call @reg1::@read() : () -> i32
    %c100 = arith.constant 100 : i32
    %overflow = arith.cmpi sgt, %val, %c100 : i32
    
    txn.if %overflow {
      txn.abort
    } else {
      // Continue if no overflow
    }
    
    // Only execute if no overflow
    %c1 = arith.constant 1 : i32
    %inc = arith.addi %val, %c1 : i32
    txn.call @reg1::@write(%inc) : (i32) -> ()
    txn.yield
  }
  
  // Simple conflicting actions
  txn.action_method @writer1() {
    %c42 = arith.constant 42 : i32
    txn.call @reg1::@write(%c42) : (i32) -> ()
    txn.yield
  }
  
  txn.action_method @writer2() {
    %c100 = arith.constant 100 : i32
    txn.call @reg1::@write(%c100) : (i32) -> ()
    txn.yield
  }
  
  // Rule that's conflict-free with some actions
  txn.rule @independentRule {
    %val = txn.call @reg2::@read() : () -> i32
    %c1 = arith.constant 1 : i32
    %inc = arith.addi %val, %c1 : i32
    txn.call @reg2::@write(%inc) : (i32) -> ()
    txn.yield
  }
  
  txn.schedule [@conditionalAbort, @wireAbort, @guardedRule, @abortingRule, @writer1, @writer2, @independentRule] {
    conflict_matrix = {
      // Self conflicts
      "conditionalAbort,conditionalAbort" = 2 : i32,
      "wireAbort,wireAbort" = 2 : i32,
      "guardedRule,guardedRule" = 2 : i32,
      "abortingRule,abortingRule" = 2 : i32,
      "writer1,writer1" = 2 : i32,
      "writer2,writer2" = 2 : i32,
      "independentRule,independentRule" = 2 : i32,
      
      // Conflicts
      "writer1,writer2" = 2 : i32,              // Both write reg1
      "conditionalAbort,writer1" = 2 : i32,     // Both access reg1
      "conditionalAbort,writer2" = 2 : i32,     // Both access reg1
      "wireAbort,writer1" = 2 : i32,           // Both write reg1
      "wireAbort,writer2" = 2 : i32,           // Both write reg1
      "wireAbort,guardedRule" = 2 : i32,       // wireAbort writes reg1, guardedRule reads reg1
      "guardedRule,writer1" = 2 : i32,          // guardedRule reads reg1
      "guardedRule,writer2" = 2 : i32,          // guardedRule reads reg1
      "abortingRule,writer1" = 2 : i32,         // Both access reg1
      "abortingRule,writer2" = 2 : i32,         // Both access reg1
      "independentRule,guardedRule" = 2 : i32,  // Both access reg2
      
      // Sequential relationships
      "writer1,guardedRule" = 0 : i32,          // writer1 before guardedRule
      "writer2,guardedRule" = 1 : i32,          // writer2 after guardedRule
      
      // Conflict-free
      "independentRule,writer1" = 3 : i32,      // Different registers
      "independentRule,writer2" = 3 : i32,      // Different registers
      "independentRule,wireAbort" = 3 : i32,    // Different resources
      "conditionalAbort,independentRule" = 3 : i32  // conditionalAbort uses reg1/reg2, but CF
    }
  }
}

// DYNAMIC mode checks - default behavior
// DYNAMIC: firrtl.module @WillFireModes

// conditionalAbort will-fire and reach_abort
// DYNAMIC: %conditionalAbort_reach_abort = firrtl.or
// DYNAMIC: %conditionalAbort_no_conflict = firrtl.and
// DYNAMIC: %conditionalAbort_guard = firrtl.or %conditionalAbortRDY
// DYNAMIC: %conditionalAbort_no_abort = firrtl.not %conditionalAbort_reach_abort
// DYNAMIC: %conditionalAbort_can_fire = firrtl.and %conditionalAbort_guard, %conditionalAbort_no_abort
// DYNAMIC: %conditionalAbort_will_fire = firrtl.and %conditionalAbortEN, %conditionalAbort_can_fire

// wireAbort with simple abort condition
// DYNAMIC: %wireAbort_reach_abort = firrtl.or
// DYNAMIC: %wireAbort_will_fire = firrtl.and

// guardedRule with guard check
// DYNAMIC: %guardedRule_guard = 
// DYNAMIC: %guardedRule_will_fire = firrtl.and

// abortingRule with abort inside
// DYNAMIC: %abortingRule_reach_abort = firrtl.or
// DYNAMIC: %abortingRule_will_fire = firrtl.and

// writer1 and writer2 conflict
// DYNAMIC: %writer1_conflicts_writer2 = firrtl.or %writer2_wf
// DYNAMIC: %writer1_no_conflict = firrtl.and
// DYNAMIC: %writer2_conflicts_writer1 = firrtl.or %writer1_wf
// DYNAMIC: %writer2_no_conflict = firrtl.and

// independentRule is conflict-free with many
// DYNAMIC: %independentRule_no_conflict = firrtl.and

// STATIC mode checks - conservative conflict resolution
// STATIC: firrtl.module @WillFireModes

// In static mode, conflicts are resolved more conservatively
// STATIC: %conditionalAbort_will_fire = firrtl.and %conditionalAbortEN
// STATIC-NOT: %conditionalAbort_reach_abort
// STATIC: %guardedRule_will_fire = firrtl.and
// STATIC: %abortingRule_will_fire = firrtl.and
// STATIC: %writer1_will_fire = firrtl.and %writer1EN
// STATIC: %writer2_will_fire = firrtl.and %writer2EN

// Static mode uses simpler conflict checking
// STATIC: %writer1_no_conflict = firrtl.and
// STATIC: %writer2_no_conflict = firrtl.and

