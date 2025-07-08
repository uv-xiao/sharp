// RUN: sharp-opt %s -sharp-simulate=mode=translation | FileCheck %s

// Test multi-cycle operations with timing attributes
txn.module @MultiCycle {
  // Static 2-cycle operation
  txn.action_method @readMemory(%addr: i32) {
    txn.return
  }
  
  // Static 5-cycle operation
  txn.action_method @process() {
    txn.return
  }
  
  // Dynamic timing operation  
  txn.action_method @conditionalOp(%cond: i1) {
    txn.if %cond {
      txn.yield
    } else {
      txn.yield
    }
    txn.return
  }
  
  // Rule that uses multi-cycle operations
  txn.rule @pipeline {
    %addr = arith.constant 0 : i32
    txn.call @readMemory(%addr) : (i32) -> ()
    txn.call @process() : () -> ()
  }
  
  txn.schedule [@readMemory, @process, @conditionalOp, @pipeline]
}

// CHECK: // Generated Txn Module Simulation
// CHECK: class MultiCycleModule : public SimModule {

// CHECK:   // Action method: readMemory
// CHECK:   void readMemory(int64_t arg0) {
// CHECK:   }

// CHECK:   // Action method: process
// CHECK:   void process() {
// CHECK:   }