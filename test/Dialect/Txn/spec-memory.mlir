// RUN: sharp-opt %s | FileCheck %s

// Test SpecMemory primitive (placeholder for future implementation)

// This test documents the expected syntax for SpecMemory when implemented
// Currently using placeholder operations

// CHECK-LABEL: txn.module @MemoryExample
txn.module @MemoryExample {
  // Placeholder for future SpecMemory implementation
  
  txn.action_method @write(%addr: i32, %data: i32) {
    // Future: txn.mem_write %mem[%addr], %data
    txn.return
  }
  
  txn.value_method @read(%addr: i32) -> i32 {
    // Future: %val = txn.mem_read %mem[%addr]
    %placeholder = arith.constant 0 : i32
    txn.return %placeholder : i32
  }
  
  txn.schedule [@write] {
    conflict_matrix = {
      "write,read" = 2 : i32  // Conflict between write and read
    }
  }
}