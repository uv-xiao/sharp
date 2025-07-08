// RUN: sharp-opt %s --sharp-simulate="mode=translation" | FileCheck %s

// Define primitives used in tests
txn.primitive @Register type = "hw" interface = !txn.module<"Register"> {
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

// Boolean register primitive
txn.primitive @RegisterBool type = "hw" interface = !txn.module<"RegisterBool"> {
  txn.fir_value_method @read() : () -> i1
  txn.fir_action_method @write() : (i1) -> ()
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
} {firrtl.impl = "RegisterBool_impl"}

// Test multi-cycle rules
txn.module @MultiCycleRuleTest {
  %counter = txn.instance @counter of @Register : !txn.module<"Register">
  %status = txn.instance @status of @RegisterBool : !txn.module<"RegisterBool">
  
  txn.rule @incrementRule {
    %enabled = txn.call @status::@read() : () -> i1
    
    // Per-cycle: increment counter if enabled
    txn.if %enabled {
      %val = txn.call @counter::@read() : () -> i32
      %one = arith.constant 1 : i32
      %next = arith.addi %val, %one : i32
      txn.call @counter::@write(%next) : (i32) -> ()
      txn.yield
    } else {
      txn.yield
    }
    
    // Multi-cycle: disable after 3 cycles
    txn.future {
      %done = txn.launch after 3 {
        %false = arith.constant 0 : i1
        txn.call @status::@write(%false) : (i1) -> ()
        txn.yield
      }
    }
  }
  
  txn.schedule [@incrementRule]
}

// CHECK: class MultiCycleRuleTestModule : public MultiCycleSimModule
// CHECK: registerRule("incrementRule"
// CHECK: registerMultiCycleAction("incrementRule"
// CHECK: std::unique_ptr<MultiCycleExecution> incrementRule_multicycle()
// CHECK: Future block launches
// CHECK: launch->hasStaticLatency = true;
// CHECK: launch->latency = 3;
// CHECK: status_data = _launch_0;