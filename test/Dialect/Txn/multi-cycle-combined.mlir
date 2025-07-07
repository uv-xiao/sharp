// RUN: sharp-opt %s --sharp-simulate="mode=translation" | FileCheck %s

// Test combined static latency + dynamic dependency
txn.module @CombinedLaunchTest {
  %mem = txn.instance @mem of @Memory<i32> : !txn.module<"Memory">
  
  txn.action_method @complexSequence() attributes {multicycle = true} {
    txn.future {
      // Phase 1: Write to address 0 after 2 cycles
      %phase1 = txn.launch after 2 {
        %addr0 = arith.constant 0 : i10
        %val100 = arith.constant 100 : i32
        txn.call @mem::@write(%addr0, %val100) : (i10, i32) -> ()
        txn.yield
      }
      
      // Phase 2: After phase1, wait 3 more cycles then write to address 1
      %phase2 = txn.launch until %phase1 after 3 {
        %addr1 = arith.constant 1 : i10
        %val200 = arith.constant 200 : i32
        txn.call @mem::@write(%addr1, %val200) : (i10, i32) -> ()
        txn.yield
      }
      
      // Phase 3: After phase2, immediately write to address 2
      %phase3 = txn.launch until %phase2 {
        %addr2 = arith.constant 2 : i10
        %val300 = arith.constant 300 : i32
        txn.call @mem::@write(%addr2, %val300) : (i10, i32) -> ()
        txn.yield
      }
    }
    txn.return
  }
  
  txn.schedule [@complexSequence]
}

// CHECK: class CombinedLaunchTestModule : public MultiCycleSimModule
// CHECK: launch->hasStaticLatency = true;
// CHECK: launch->latency = 2;
// CHECK-NOT: launch->conditionName
// CHECK: launch->hasStaticLatency = true;
// CHECK: launch->latency = 3;
// CHECK: launch->conditionName = "complexSequence_launch_0";
// CHECK-NOT: launch->hasStaticLatency
// CHECK: launch->conditionName = "complexSequence_launch_1";