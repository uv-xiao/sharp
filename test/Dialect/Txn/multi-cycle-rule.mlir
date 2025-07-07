// RUN: sharp-opt %s --sharp-simulate="mode=translation" | FileCheck %s

// Test multi-cycle rules
txn.module @MultiCycleRuleTest {
  %counter = txn.instance @counter of @Register<i32> : !txn.module<"Register">
  %status = txn.instance @status of @Register<i1> : !txn.module<"Register">
  
  txn.rule @incrementRule attributes {multicycle = true} {
    %enabled = txn.call @status::@read() : () -> i1
    txn.if %enabled {
      // Per-cycle: increment counter
      %val = txn.call @counter::@read() : () -> i32
      %one = arith.constant 1 : i32
      %next = arith.addi %val, %one : i32
      txn.call @counter::@write(%next) : (i32) -> ()
      
      txn.future {
        // Disable after 3 cycles
        %done = txn.launch after 3 {
          %false = arith.constant false : i1
          txn.call @status::@write(%false) : (i1) -> ()
          txn.yield
        }
      }
    }
    txn.yield
  }
  
  txn.schedule [@incrementRule]
}

// CHECK: class MultiCycleRuleTestModule : public MultiCycleSimModule
// CHECK: registerRule("incrementRule"
// CHECK: registerMultiCycleAction("incrementRule"
// CHECK: std::unique_ptr<MultiCycleExecution> incrementRule_multicycle()
// CHECK: counter_data = _2;
// CHECK: launch->hasStaticLatency = true;
// CHECK: launch->latency = 3;
// CHECK: int64_t _launch_0 = 0;
// CHECK: status_data = _launch_0;