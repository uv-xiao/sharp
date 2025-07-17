// A module with various analysis challenges
txn.module @ComplexModule attributes {top} {
  // State elements
  txn.instance @data of @Register<i32> 
  txn.instance @flag of @Register<i1> 
  txn.instance @temp of @Wire<i32> 
  
  // Action that reads and writes same register
  txn.action_method @readModifyWrite(%delta: i32) {
    %current = txn.call @data::@read() : () -> i32
    %new = arith.addi %current, %delta : i32
    txn.call @data::@write(%new) : (i32) -> ()
    txn.yield
  }
  
  // Action with conditional execution and potential abort
  txn.action_method @conditionalUpdate(%cond: i1, %value: i32) {
    %flag_val = txn.call @flag::@read() : () -> i1
    %should_update = arith.andi %cond, %flag_val : i1
    
    txn.if %should_update {
      // Only write if conditions are met
      txn.call @data::@write(%value) : (i32) -> ()
      txn.yield
    } else {
      // Abort if conditions not satisfied
      txn.abort
    }
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
  
  // Action with nested conditionals and multiple abort paths
  txn.action_method @updateFlag() {
    %data_val = txn.call @data::@read() : () -> i32
    %zero = arith.constant 0 : i32
    %max_val = arith.constant 100 : i32
    
    %is_zero = arith.cmpi eq, %data_val, %zero : i32
    %is_too_large = arith.cmpi sgt, %data_val, %max_val : i32
    
    txn.if %is_too_large {
      // Invalid data - abort immediately
      txn.abort
    } else {
      txn.if %is_zero {
        // Set flag to true for zero value
        %true = arith.constant true
        txn.call @flag::@write(%true) : (i1) -> ()
        txn.yield
      } else {
        // Check if value is positive
        %is_positive = arith.cmpi sgt, %data_val, %zero : i32
        txn.if %is_positive {
          // Set flag to false for positive value
          %false = arith.constant false
          txn.call @flag::@write(%false) : (i1) -> ()
          txn.yield
        } else {
          // Negative value - abort
          txn.abort
        }
        txn.yield
      }
      txn.yield
    }
    txn.yield
  }
  
  // Action that demonstrates complex reachability analysis
  txn.action_method @complexProcessor(%enable: i1, %threshold: i32) {
    %flag_val = txn.call @flag::@read() : () -> i1
    %data_val = txn.call @data::@read() : () -> i32
    
    txn.if %enable {
      txn.if %flag_val {
        // Path 1: Both enable and flag are true
        %exceeds_threshold = arith.cmpi sgt, %data_val, %threshold : i32
        txn.if %exceeds_threshold {
          // Reachable only when: enable AND flag AND (data > threshold)
          %doubled = arith.muli %data_val, %data_val : i32
          txn.call @data::@write(%doubled) : (i32) -> ()
          txn.yield
        } else {
          // Reachable only when: enable AND flag AND (data <= threshold)
          txn.call @temp::@write(%data_val) : (i32) -> ()
          txn.yield
        }
        txn.yield
      } else {
        // Path 2: Enable true but flag false - conditional abort
        %is_negative = arith.cmpi slt, %data_val, %threshold : i32
        txn.if %is_negative {
          // Abort if data is negative when flag is false
          txn.abort
        } else {
          // Safe path when enable=true, flag=false, data>=0
          %incremented = arith.addi %data_val, %threshold : i32
          txn.call @data::@write(%incremented) : (i32) -> ()
          txn.yield
        }
        txn.yield
      }
      txn.yield
    } else {
      // Path 3: Enable false - always abort
      txn.abort
    }
    txn.yield
  }
  
  // Partial schedule - let inference complete it (note: getProcessed is a value method and not scheduled)
  txn.schedule [@readModifyWrite, @conditionalUpdate, @updateFlag, @complexProcessor] {
    conflict_matrix = {
      // Only specify some conflicts
      "readModifyWrite,conditionalUpdate" = 2 : i32  // C
    }
  }
}