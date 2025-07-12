// Module with incomplete schedule for testing schedule completeness validation
txn.module @IncompleteScheduleExample {
  // State elements
  %data = txn.instance @data of @Register<i32> : !txn.module<"Register">
  %flag = txn.instance @flag of @Register<i1> : !txn.module<"Register">
  
  // Action method 1
  txn.action_method @processData(%value: i32) {
    %current = txn.call @data::@read() : () -> i32
    %sum = arith.addi %current, %value : i32
    txn.call @data::@write(%sum) : (i32) -> ()
    txn.yield
  }
  
  // Action method 2
  txn.action_method @updateFlag() {
    %data_val = txn.call @data::@read() : () -> i32
    %zero = arith.constant 0 : i32
    %is_positive = arith.cmpi sgt, %data_val, %zero : i32
    txn.call @flag::@write(%is_positive) : (i1) -> ()
    txn.yield
  }
  
  // Rule (action 3)
  txn.rule @defaultRule {
    %flag_val = txn.call @flag::@read() : () -> i1
    txn.if %flag_val {
      %zero = arith.constant 0 : i32
      txn.call @data::@write(%zero) : (i32) -> ()
      txn.yield
    } else {
      txn.yield
    }
    txn.yield
  }
  
  // Value method (should NOT be in schedule)
  txn.value_method @getCurrentData() -> i32 {
    %result = txn.call @data::@read() : () -> i32
    txn.return %result : i32
  }
  
  // INCOMPLETE SCHEDULE: Missing @updateFlag and @defaultRule
  // This will fail general-check schedule completeness validation
  txn.schedule [@processData] {
    conflict_matrix = {
      "processData,processData" = 2 : i32,  // C (self-conflict)
      "processData,defaultRule" = 0 : i32  // SB
    }
  }
}

// Module with complete schedule for comparison
txn.module @CompleteScheduleExample {
  %data = txn.instance @data of @Register<i32> : !txn.module<"Register">
  
  txn.action_method @increment() {
    %current = txn.call @data::@read() : () -> i32
    %one = arith.constant 1 : i32
    %new_val = arith.addi %current, %one : i32
    txn.call @data::@write(%new_val) : (i32) -> ()
    txn.yield
  }
  
  txn.action_method @reset() {
    %zero = arith.constant 0 : i32
    txn.call @data::@write(%zero) : (i32) -> ()
    txn.yield
  }
  
  // COMPLETE SCHEDULE: Includes all actions
  txn.schedule [@increment, @reset] {
    conflict_matrix = {
      "increment,increment" = 2 : i32,  // C
      "increment,reset" = 2 : i32,      // C
      "reset,increment" = 2 : i32,      // C
      "reset,reset" = 2 : i32           // C
    }
  }
}