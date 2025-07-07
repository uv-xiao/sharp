// RUN: sharp-opt %s --sharp-simulate="mode=translation" | FileCheck %s

// Test static latency launches
txn.module @StaticLaunchTest {
  %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
  
  txn.action_method @delayedWrite(%value: i32) attributes {multicycle = true} {
    // Immediate write
    txn.call @reg::@write(%value) : (i32) -> ()
    
    txn.future {
      // Write 10 after 2 cycles
      %done1 = txn.launch after 2 {
        %ten = arith.constant 10 : i32
        txn.call @reg::@write(%ten) : (i32) -> ()
        txn.yield
      }
      
      // Write 20 after 5 cycles
      %done2 = txn.launch after 5 {
        %twenty = arith.constant 20 : i32
        txn.call @reg::@write(%twenty) : (i32) -> ()
        txn.yield
      }
    }
    txn.return
  }
  
  txn.schedule [@delayedWrite]
}

// CHECK: class StaticLaunchTestModule : public MultiCycleSimModule
// CHECK: registerMultiCycleAction("delayedWrite"
// CHECK: std::unique_ptr<MultiCycleExecution> delayedWrite_multicycle(int64_t arg0)
// CHECK: reg_data = arg0;
// CHECK: launch->hasStaticLatency = true;
// CHECK: launch->latency = 2;
// CHECK: int64_t _launch_0 = 10;
// CHECK: reg_data = _launch_0;
// CHECK: launch->hasStaticLatency = true;
// CHECK: launch->latency = 5;
// CHECK: int64_t _launch_0 = 20;
// CHECK: reg_data = _launch_0;