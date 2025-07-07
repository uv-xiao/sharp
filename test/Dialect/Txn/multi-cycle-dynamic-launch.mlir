// RUN: sharp-opt %s --sharp-simulate="mode=translation" | FileCheck %s

// Test dynamic dependency launches
txn.module @DynamicLaunchTest {
  %fifo = txn.instance @fifo of @FIFO<i32> : !txn.module<"FIFO">
  
  txn.action_method @sequentialOps(%v1: i32, %v2: i32, %v3: i32) attributes {multicycle = true} {
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
// CHECK-DAG: launch->hasStaticLatency = true;
// CHECK-DAG: launch->latency = 1;
// CHECK-DAG: fifo_queue.push(arg0);
// CHECK-DAG: launch->conditionName = "sequentialOps_launch_0";
// CHECK-DAG: fifo_queue.push(arg1);
// CHECK-DAG: launch->conditionName = "sequentialOps_launch_1";
// CHECK-DAG: fifo_queue.push(arg2);