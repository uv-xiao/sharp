// RUN: sharp-opt %s --sharp-simulate="mode=translation" | FileCheck %s

// Define primitives
txn.primitive @Register type = "hw" interface = index {
  txn.fir_value_method @read() : () -> i32
  txn.fir_action_method @write() : (i32) -> ()
  txn.clock_by @clk
  txn.reset_by @rst
  txn.schedule [@write] {
    conflict_matrix = {
      "read,read" = 3 : i32,
      "read,write" = 3 : i32,
      "write,read" = 3 : i32,
      "write,write" = 2 : i32
    }
  }
} {firrtl.impl = "Register_impl"}

txn.module @MultiCycleCounter {
  %count = txn.instance @count of @Register : index
  
  txn.action_method @increment() {
    // Per-cycle action
    %v = txn.call @count::@read() : () -> i32
    %one = arith.constant 1 : i32
    %next = arith.addi %v, %one : i32
    txn.call @count::@write(%next) : (i32) -> ()
    
    // Multi-cycle actions
    txn.future {
      // Static launch - must succeed after 3 cycles
      %done1 = txn.launch after 3 {
        %v2 = txn.call @count::@read() : () -> i32
        %two = arith.constant 2 : i32
        %next2 = arith.addi %v2, %two : i32
        txn.call @count::@write(%next2) : (i32) -> ()
        txn.yield
      }
      
      // Dynamic launch - depends on done1
      %done2 = txn.launch until %done1 {
        %v3 = txn.call @count::@read() : () -> i32
        %three = arith.constant 3 : i32
        %next3 = arith.addi %v3, %three : i32
        txn.call @count::@write(%next3) : (i32) -> ()
        txn.yield
      }
      
      // Combined launch
      %done3 = txn.launch until %done2 after 1 {
        %v4 = txn.call @count::@read() : () -> i32
        %four = arith.constant 4 : i32
        %next4 = arith.addi %v4, %four : i32
        txn.call @count::@write(%next4) : (i32) -> ()
        txn.yield
      }
    }
    txn.return
  }
  
  txn.schedule [@increment]
}

// CHECK: class MultiCycleCounterModule : public MultiCycleSimModule {
// CHECK: MultiCycleCounterModule() : MultiCycleSimModule
// CHECK: registerMultiCycleAction("increment"
// CHECK: std::unique_ptr<MultiCycleExecution> increment_multicycle()
// CHECK: launch->hasStaticLatency = true;
// CHECK: launch->latency = 3;
// CHECK: launch->conditionName = "increment_launch_0";
// CHECK: launch->conditionName = "increment_launch_1";