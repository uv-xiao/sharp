// RUN: sharp-opt %s -sharp-validate-action-calls -split-input-file -verify-diagnostics

// Test 1: Valid - actions calling value methods
txn.module @ValidActionCalls {
  %reg = txn.instance @reg of @Register<i32> : index
  
  txn.value_method @getValue() -> i32 {
    %v = txn.call @reg::@read() : () -> i32
    txn.return %v : i32
  }
  
  txn.action_method @setValue(%v: i32) {
    txn.call @reg::@write(%v) : (i32) -> ()
    txn.return
  }
  
  txn.rule @incrementRule {
    // Valid: rule calling value method
    %v = txn.call @getValue() : () -> i32
    %one = arith.constant 1 : i32
    %next = arith.addi %v, %one : i32
    // Valid: rule calling instance action method
    txn.call @reg::@write(%next) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@setValue, @incrementRule]
}

// -----

// Test 2: Invalid - rule calling action method in same module
txn.module @RuleCallingActionMethod {
  %reg = txn.instance @reg of @Register<i32> : index
  
  txn.action_method @setValue(%v: i32) {
    txn.call @reg::@write(%v) : (i32) -> ()
    txn.return
  }
  
  txn.rule @badRule {
    %v = arith.constant 42 : i32
    // expected-error@+1 {{action 'badRule' cannot call action 'setValue' in the same module}}
    txn.call @setValue(%v) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@setValue, @badRule]
}

// -----

// Test 3: Invalid - action method calling another action method
txn.module @ActionMethodCallingActionMethod {
  %reg = txn.instance @reg of @Register<i32> : index
  
  txn.action_method @increment() {
    %v = txn.call @reg::@read() : () -> i32
    %one = arith.constant 1 : i32
    %next = arith.addi %v, %one : i32
    // expected-error@+1 {{action 'increment' cannot call action 'setValue' in the same module}}
    txn.call @setValue(%next) : (i32) -> ()
    txn.return
  }
  
  txn.action_method @setValue(%v: i32) {
    txn.call @reg::@write(%v) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@increment, @setValue]
}

// -----

// Test 4: Invalid - action method calling a rule (rules are actions too)
txn.module @ActionMethodCallingRule {
  %reg = txn.instance @reg of @Register<i32> : index
  
  txn.rule @autoUpdate {
    %v = arith.constant 0 : i32
    txn.call @reg::@write(%v) : (i32) -> ()
    txn.return
  }
  
  txn.action_method @reset() {
    // expected-error@+1 {{action 'reset' cannot call action 'autoUpdate' in the same module}}
    txn.call @autoUpdate() : () -> ()
    txn.return
  }
  
  txn.schedule [@autoUpdate, @reset]
}

// -----

// Test 5: Valid - actions calling instance methods (both value and action)
txn.module @ValidInstanceCalls {
  %counter = txn.instance @counter of @Counter : index
  %display = txn.instance @display of @Display : index
  
  txn.rule @updateDisplay {
    // Valid: calling instance value method
    %v = txn.call @counter::@getValue() : () -> i32
    // Valid: calling instance action method
    txn.call @display::@show(%v) : (i32) -> ()
    txn.return
  }
  
  txn.action_method @incrementAndDisplay() {
    // Valid: calling instance action methods
    txn.call @counter::@increment() : () -> ()
    %v = txn.call @counter::@getValue() : () -> i32
    txn.call @display::@show(%v) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@updateDisplay, @incrementAndDisplay]
}

// -----

// Test 6: Complex case - nested calls
txn.module @NestedCalls {
  %reg = txn.instance @reg of @Register<i32> : index
  
  txn.value_method @compute() -> i32 {
    %v = txn.call @reg::@read() : () -> i32
    %two = arith.constant 2 : i32
    %result = arith.muli %v, %two : i32
    txn.return %result : i32
  }
  
  txn.action_method @process() {
    %v = txn.call @compute() : () -> i32  // Valid: calling value method
    %c0 = arith.constant 0 : i32
    %cond = arith.cmpi ne, %v, %c0 : i32
    txn.if %cond {
      // expected-error@+1 {{action 'process' cannot call action 'store' in the same module}}
      txn.call @store(%v) : (i32) -> ()
      txn.yield
    } else {
      txn.yield
    }
    txn.return
  }
  
  txn.action_method @store(%v: i32) {
    txn.call @reg::@write(%v) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@process, @store]
}