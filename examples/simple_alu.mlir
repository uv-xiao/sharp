// Simple ALU example without primitives
// This demonstrates pure combinational logic
txn.module @SimpleALU {
  // Value method: add two numbers
  txn.value_method @add(%a: i32, %b: i32) -> i32 attributes {timing = "combinational"} {
    %sum = arith.addi %a, %b : i32
    txn.return %sum : i32
  }
  
  // Value method: subtract
  txn.value_method @sub(%a: i32, %b: i32) -> i32 attributes {timing = "combinational"} {
    %diff = arith.subi %a, %b : i32
    txn.return %diff : i32
  }
  
  // Value method: multiply
  txn.value_method @mul(%a: i32, %b: i32) -> i32 attributes {timing = "combinational"} {
    %prod = arith.muli %a, %b : i32
    txn.return %prod : i32
  }
  
  // Value method: bitwise AND
  txn.value_method @and(%a: i32, %b: i32) -> i32 attributes {timing = "combinational"} {
    %result = arith.andi %a, %b : i32
    txn.return %result : i32
  }
  
  // Value method: bitwise OR
  txn.value_method @or(%a: i32, %b: i32) -> i32 attributes {timing = "combinational"} {
    %result = arith.ori %a, %b : i32
    txn.return %result : i32
  }
  
  // Value method: bitwise XOR
  txn.value_method @xor(%a: i32, %b: i32) -> i32 attributes {timing = "combinational"} {
    %result = arith.xori %a, %b : i32
    txn.return %result : i32
  }
  
  // Value method: compare equal
  txn.value_method @eq(%a: i32, %b: i32) -> i1 attributes {timing = "combinational"} {
    %result = arith.cmpi eq, %a, %b : i32
    txn.return %result : i1
  }
  
  // Value method: compare greater than
  txn.value_method @gt(%a: i32, %b: i32) -> i1 attributes {timing = "combinational"} {
    %result = arith.cmpi sgt, %a, %b : i32
    txn.return %result : i1
  }
  
  // No schedule needed - all value methods are combinational
  txn.schedule []
}