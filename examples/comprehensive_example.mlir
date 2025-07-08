// Comprehensive example showcasing Sharp's features
// This demonstrates:
// - Primitive usage with automatic type inference
// - Conflict matrix inference
// - Reachability analysis
// - FIRRTL conversion
// - Verilog export

// Define Register primitive with conflict matrix
txn.primitive @Register type = "hw" interface = !txn.module<"Register"> {
  txn.fir_value_method @read() : () -> i32
  txn.fir_action_method @write() : (i32) -> ()
  txn.clock_by @clk
  txn.reset_by @rst
  txn.schedule [@write] {
    conflict_matrix = {
      "read,read" = 3 : i32,    // CF
      "read,write" = 3 : i32,   // CF
      "write,read" = 3 : i32,   // CF
      "write,write" = 2 : i32   // C - only one write per cycle
    }
  }
} {firrtl.impl = "Register_impl"}

// Main module: a configurable up/down counter
txn.module @ConfigurableCounter {
  // State registers (type inferred from primitive methods)
  %value = txn.instance @value of @Register : !txn.module<"Register">
  %step = txn.instance @step of @Register : !txn.module<"Register">
  %direction = txn.instance @direction of @Register : !txn.module<"Register">  // 0=up, 1=down
  
  // Value method: read current count
  txn.value_method @getValue() -> i32 attributes {always_ready, timing = "combinational"} {
    %val = txn.call @value::@read() : () -> i32
    txn.return %val : i32
  }
  
  // Value method: check if counting up
  txn.value_method @isCountingUp() -> i1 attributes {timing = "combinational"} {
    %dir = txn.call @direction::@read() : () -> i32
    %zero = arith.constant 0 : i32
    %is_up = arith.cmpi eq, %dir, %zero : i32
    txn.return %is_up : i1
  }
  
  // Action: count (up or down based on direction)
  txn.action_method @count() attributes {timing = "static(1)"} {
    %current = txn.call @value::@read() : () -> i32
    %step_val = txn.call @step::@read() : () -> i32
    %dir = txn.call @direction::@read() : () -> i32
    %zero = arith.constant 0 : i32
    %is_up = arith.cmpi eq, %dir, %zero : i32
    
    txn.if %is_up {
      %new = arith.addi %current, %step_val : i32
      txn.call @value::@write(%new) : (i32) -> ()
      txn.yield
    } else {
      %new = arith.subi %current, %step_val : i32
      txn.call @value::@write(%new) : (i32) -> ()
      txn.yield
    }
    txn.return
  }
  
  // Action: set step size
  txn.action_method @setStep(%new_step: i32) attributes {timing = "static(1)"} {
    txn.call @step::@write(%new_step) : (i32) -> ()
    txn.return
  }
  
  // Action: toggle direction
  txn.action_method @toggleDirection() attributes {timing = "static(1)"} {
    %dir = txn.call @direction::@read() : () -> i32
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %is_up = arith.cmpi eq, %dir, %zero : i32
    
    txn.if %is_up {
      txn.call @direction::@write(%one) : (i32) -> ()
      txn.yield
    } else {
      txn.call @direction::@write(%zero) : (i32) -> ()
      txn.yield
    }
    txn.return
  }
  
  // Action: reset all state
  txn.action_method @reset() attributes {timing = "static(1)"} {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    txn.call @value::@write(%zero) : (i32) -> ()
    txn.call @step::@write(%one) : (i32) -> ()
    txn.call @direction::@write(%zero) : (i32) -> ()
    txn.return
  }
  
  // Rule: auto-increment when value reaches 100 in up mode
  txn.rule @auto_reset {
    %val = txn.call @getValue() : () -> i32
    %hundred = arith.constant 100 : i32
    %at_limit = arith.cmpi sge, %val, %hundred : i32
    %is_up = txn.call @isCountingUp() : () -> i1
    %should_reset = arith.andi %at_limit, %is_up : i1
    
    txn.if %should_reset {
      txn.call @reset() : () -> ()
      txn.yield
    } else {
      txn.yield
    }
    
    txn.return
  }
  
  // Schedule with partial conflict matrix (will be completed by analysis)
  txn.schedule [@count, @setStep, @toggleDirection, @reset, @auto_reset] {
    conflict_matrix = {
      // Specify key conflicts, let inference handle the rest
      "count,reset" = 2 : i32,         // C
      "setStep,reset" = 2 : i32,       // C
      "toggleDirection,reset" = 2 : i32  // C
    }
  }
}