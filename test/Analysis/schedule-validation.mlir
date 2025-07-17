// RUN: sharp-opt %s -sharp-validate-schedule -split-input-file -verify-diagnostics

// Test 1: Valid schedule with only actions
txn.module @ValidSchedule {
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
    %v = txn.call @getValue() : () -> i32
    %one = arith.constant 1 : i32
    %next = arith.addi %v, %one : i32
    txn.call @setValue(%next) : (i32) -> ()
    txn.yield
  }
  
  // Valid: only actions in schedule
  txn.schedule [@setValue, @incrementRule]
}

// -----

// Test 2: Invalid schedule with value method
txn.module @InvalidScheduleWithValueMethod {
  %reg = txn.instance @reg of @Register<i32> : index
  
  txn.value_method @getValue() -> i32 {
    %v = txn.call @reg::@read() : () -> i32
    txn.return %v : i32
  }
  
  txn.action_method @setValue(%v: i32) {
    txn.call @reg::@write(%v) : (i32) -> ()
    txn.return
  }
  
  // expected-error@+1 {{value method 'getValue' cannot be in schedule}}
  txn.schedule [@getValue, @setValue]
}

// -----

// Test 3: Schedule with non-existent action
txn.module @ScheduleWithNonExistentAction {
  %reg = txn.instance @reg of @Register<i32> : index
  
  txn.action_method @setValue(%v: i32) {
    txn.call @reg::@write(%v) : (i32) -> ()
    txn.return
  }
  
  // expected-error@+1 {{scheduled action 'nonExistent' not found in module}}
  txn.schedule [@setValue, @nonExistent]
}

// -----

// Test 4: Empty schedule is valid
txn.module @EmptySchedule {
  %reg = txn.instance @reg of @Register<i32> : index
  
  txn.value_method @getValue() -> i32 {
    %v = txn.call @reg::@read() : () -> i32
    txn.return %v : i32
  }
  
  // Valid: empty schedule
  txn.schedule []
}

// -----

// Test 5: Module with empty schedule is valid
txn.module @EmptySchedule2 {
  %reg = txn.instance @reg of @Register<i32> : index
  
  txn.value_method @getValue() -> i32 {
    %v = txn.call @reg::@read() : () -> i32
    txn.return %v : i32
  }
  
  // No schedule operation - modules must have a schedule
  txn.schedule []
}

// -----

// Test 6: Schedule with multiple value methods
txn.module @MultipleValueMethodsInSchedule {
  %reg = txn.instance @reg of @Register<i32> : index
  
  txn.value_method @getValue1() -> i32 {
    %v = txn.call @reg::@read() : () -> i32
    txn.return %v : i32
  }
  
  txn.value_method @getValue2() -> i32 {
    %v = txn.call @reg::@read() : () -> i32
    %two = arith.constant 2 : i32
    %result = arith.muli %v, %two : i32
    txn.return %result : i32
  }
  
  txn.action_method @setValue(%v: i32) {
    txn.call @reg::@write(%v) : (i32) -> ()
    txn.return
  }
  
  // expected-error@+2 {{value method 'getValue1' cannot be in schedule}}
  // expected-error@+1 {{value method 'getValue2' cannot be in schedule}}
  txn.schedule [@getValue1, @setValue, @getValue2]
}