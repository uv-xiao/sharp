// Example demonstrating the replacement of value methods with arguments using txn.func
// According to the execution model, value methods should not take arguments since their
// results must be constant during one cycle.

// BEFORE: Using value methods with arguments (not recommended)
// txn.module @CounterBefore {
//   %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
//   
//   // This violates the execution model - value methods shouldn't have arguments
//   txn.value_method @read_add(%offset: i32) -> i32 {
//     %val = txn.call @reg::@read() : () -> i32
//     %result = arith.addi %val, %offset : i32
//     txn.return %result : i32
//   }
// }

// AFTER: Using txn.func for combinational logic with arguments
txn.module @CounterAfter {
  %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
  
  // Pure function for combinational logic
  txn.func @add_offset(%val: i32, %offset: i32) -> i32 {
    %result = arith.addi %val, %offset : i32
    txn.return %result : i32
  }
  
  // Value method with no arguments - constant during cycle
  txn.value_method @read() -> i32 {
    %val = txn.call @reg::@read() : () -> i32
    txn.return %val : i32
  }
  
  // Action method that uses the function
  txn.action_method @increment_by(%offset: i32) {
    %current = txn.call @reg::@read() : () -> i32
    %new_val = txn.func_call @add_offset(%current, %offset) : (i32, i32) -> i32
    txn.call @reg::@write(%new_val) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@increment_by]
}

// More complex example with ALU operations
txn.module @ALU {
  // Pure functions for ALU operations
  txn.func @add(%a: i32, %b: i32) -> i32 {
    %result = arith.addi %a, %b : i32
    txn.return %result : i32
  }
  
  txn.func @sub(%a: i32, %b: i32) -> i32 {
    %result = arith.subi %a, %b : i32
    txn.return %result : i32
  }
  
  txn.func @mul(%a: i32, %b: i32) -> i32 {
    %result = arith.muli %a, %b : i32
    txn.return %result : i32
  }
  
  txn.func @and(%a: i32, %b: i32) -> i32 {
    %result = arith.andi %a, %b : i32
    txn.return %result : i32
  }
  
  txn.func @or(%a: i32, %b: i32) -> i32 {
    %result = arith.ori %a, %b : i32
    txn.return %result : i32
  }
  
  txn.func @xor(%a: i32, %b: i32) -> i32 {
    %result = arith.xori %a, %b : i32
    txn.return %result : i32
  }
  
  txn.func @select_op(%op: i3, %a: i32, %b: i32) -> i32 {
    %c0 = arith.constant 0 : i3
    %c1 = arith.constant 1 : i3
    %c2 = arith.constant 2 : i3
    %c3 = arith.constant 3 : i3
    %c4 = arith.constant 4 : i3
    
    %is_add = arith.cmpi eq, %op, %c0 : i3
    %is_sub = arith.cmpi eq, %op, %c1 : i3
    %is_mul = arith.cmpi eq, %op, %c2 : i3
    %is_and = arith.cmpi eq, %op, %c3 : i3
    %is_or = arith.cmpi eq, %op, %c4 : i3
    
    %add_result = txn.func_call @add(%a, %b) : (i32, i32) -> i32
    %sub_result = txn.func_call @sub(%a, %b) : (i32, i32) -> i32
    %mul_result = txn.func_call @mul(%a, %b) : (i32, i32) -> i32
    %and_result = txn.func_call @and(%a, %b) : (i32, i32) -> i32
    %or_result = txn.func_call @or(%a, %b) : (i32, i32) -> i32
    %xor_result = txn.func_call @xor(%a, %b) : (i32, i32) -> i32
    
    // Chain of selects to pick the right result
    %r1 = arith.select %is_add, %add_result, %sub_result : i32
    %r2 = arith.select %is_sub, %r1, %mul_result : i32
    %r3 = arith.select %is_mul, %r2, %and_result : i32
    %r4 = arith.select %is_and, %r3, %or_result : i32
    %r5 = arith.select %is_or, %r4, %xor_result : i32
    
    txn.return %r5 : i32
  }
  
  // Storage for inputs and outputs
  %op_reg = txn.instance @op_reg of @Register<i3> : !txn.module<"Register">
  %a_reg = txn.instance @a_reg of @Register<i32> : !txn.module<"Register">
  %b_reg = txn.instance @b_reg of @Register<i32> : !txn.module<"Register">
  %result_reg = txn.instance @result_reg of @Register<i32> : !txn.module<"Register">
  
  // Value method to read current result
  txn.value_method @get_result() -> i32 {
    %result = txn.call @result_reg::@read() : () -> i32
    txn.return %result : i32
  }
  
  // Action to compute and store result
  txn.action_method @compute() {
    %op = txn.call @op_reg::@read() : () -> i3
    %a = txn.call @a_reg::@read() : () -> i32
    %b = txn.call @b_reg::@read() : () -> i32
    
    %result = txn.func_call @select_op(%op, %a, %b) : (i3, i32, i32) -> i32
    
    txn.call @result_reg::@write(%result) : (i32) -> ()
    txn.return
  }
  
  // Action methods to set inputs
  txn.action_method @set_operation(%op: i3) {
    txn.call @op_reg::@write(%op) : (i3) -> ()
    txn.return
  }
  
  txn.action_method @set_operands(%a: i32, %b: i32) {
    txn.call @a_reg::@write(%a) : (i32) -> ()
    txn.call @b_reg::@write(%b) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@compute, @set_operation, @set_operands]
}

// Run the inlining pass to see the effect:
// sharp-opt examples/value-method-replacement.mlir -sharp-inline-functions