// A module with various analysis challenges
txn.module @ComplexModule {
  // State elements
  %data = txn.instance @data of @Register<i32> : !txn.module<"Register">
  %flag = txn.instance @flag of @Register<i1> : !txn.module<"Register">
  %temp = txn.instance @temp of @Wire<i32> : !txn.module<"Wire">
  
  // Action that reads and writes same register
  txn.action_method @readModifyWrite(%delta: i32) {
    %current = txn.call @data::@read() : () -> i32
    %new = arith.addi %current, %delta : i32
    txn.call @data::@write(%new) : (i32) -> ()
    txn.yield
  }
  
  // Action that might have conflicts
  txn.action_method @conditionalUpdate(%cond: i1, %value: i32) {
    %flag_val = txn.call @flag::@read() : () -> i1
    %should_update = arith.andi %cond, %flag_val : i1
    // In real hardware, would use conditional logic
    txn.call @data::@write(%value) : (i32) -> ()
    txn.yield
  }
  
  // Value method using wire
  txn.value_method @getProcessed() -> i32 {
    %data_val = txn.call @data::@read() : () -> i32
    %two = arith.constant 2 : i32
    %doubled = arith.muli %data_val, %two : i32
    txn.call @temp::@write(%doubled) : (i32) -> ()
    %result = txn.call @temp::@read() : () -> i32
    txn.return %result : i32
  }
  
  // Action with potential combinational loop
  txn.action_method @updateFlag() {
    %data_val = txn.call @data::@read() : () -> i32
    %zero = arith.constant 0 : i32
    %is_zero = arith.cmpi eq, %data_val, %zero : i32
    txn.call @flag::@write(%is_zero) : (i1) -> ()
    txn.yield
  }
  
  // Partial schedule - let inference complete it
  txn.schedule [@readModifyWrite, @conditionalUpdate, @getProcessed, @updateFlag] {
    conflict_matrix = {
      // Only specify some conflicts
      "readModifyWrite,conditionalUpdate" = 2 : i32  // C
    }
  }
}