// RUN: sharp-opt %s --sharp-simulate="mode=translation" | FileCheck %s

// Define FIFO primitive
txn.primitive @FIFO type = "hw" interface = !txn.module<"FIFO"> {
  txn.fir_value_method @notEmpty() : () -> i1
  txn.fir_value_method @notFull() : () -> i1
  txn.fir_action_method @enqueue() : (i32) -> ()
  txn.fir_action_method @dequeue() : () -> i32
  txn.clock_by @clk
  txn.reset_by @rst
  txn.schedule [@enqueue, @dequeue] {
    conflict_matrix = {
      "enqueue,dequeue" = 2 : i32,
      "dequeue,enqueue" = 2 : i32,
      "enqueue,enqueue" = 2 : i32,
      "dequeue,dequeue" = 2 : i32
    }
  }
} {firrtl.impl = "FIFO_impl"}

// Test dynamic dependency launches
txn.module @DynamicLaunchTest {
  %fifo = txn.instance @fifo of @FIFO : !txn.module<"FIFO">
  
  txn.action_method @sequentialOps(%v1: i32, %v2: i32, %v3: i32) {
    txn.future {
      // First enqueue
      %done1 = txn.launch after 1 {
        txn.call @fifo::@enqueue(%v1) : (i32) -> ()
        txn.yield
      }
      
      // Second enqueue depends on first
      %done2 = txn.launch until %done1 {
        txn.call @fifo::@enqueue(%v2) : (i32) -> ()
        txn.yield
      }
      
      // Third enqueue depends on second
      %done3 = txn.launch until %done2 {
        txn.call @fifo::@enqueue(%v3) : (i32) -> ()
        txn.yield
      }
    }
    txn.return
  }
  
  txn.schedule [@sequentialOps]
}

// CHECK: class DynamicLaunchTestModule : public MultiCycleSimModule
// CHECK: std::unique_ptr<MultiCycleExecution> sequentialOps_multicycle(int64_t arg0, int64_t arg1, int64_t arg2)
// CHECK: launch->hasStaticLatency = true;
// CHECK: launch->latency = 1;
// CHECK: fifo_queue.push();
// CHECK: launch->conditionName = "sequentialOps_launch_0";
// CHECK: fifo_queue.push();
// CHECK: launch->conditionName = "sequentialOps_launch_1";
// CHECK: fifo_queue.push();