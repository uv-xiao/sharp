// Module with combinational loop through wires
txn.module @LoopExample {
  %wire_a = txn.instance @wire_a of @Wire<i32> : !txn.module<"Wire">
  %wire_b = txn.instance @wire_b of @Wire<i32> : !txn.module<"Wire">
  
  // Creates a->b->a loop
  txn.rule @loop_rule_a {
    %b_val = txn.call @wire_b::@read() : () -> i32
    %one = arith.constant 1 : i32
    %inc = arith.addi %b_val, %one : i32
    txn.call @wire_a::@write(%inc) : (i32) -> ()
    txn.yield
  }
  
  txn.rule @loop_rule_b {
    %a_val = txn.call @wire_a::@read() : () -> i32
    %two = arith.constant 2 : i32
    %double = arith.muli %a_val, %two : i32
    txn.call @wire_b::@write(%double) : (i32) -> ()
    txn.yield
  }
  
  txn.schedule [@loop_rule_a, @loop_rule_b]
}