// RUN: sharp-opt %s | FileCheck %s

// CHECK-LABEL: txn.module @SpecMemoryTest
txn.module @SpecMemoryTest {
  // CHECK: txn.instance @mem of @SpecMemory<i32>
  %mem = txn.instance @mem of @SpecMemory<i32> : !txn.module<"SpecMemory">
  
  // Test read method with dynamic timing
  txn.value_method @read_test(%addr: i32) -> i32 {
    // CHECK: txn.call @mem::@read(%{{.*}}) : (i32) -> i32
    %val = txn.call @mem::@read(%addr) : (i32) -> i32
    txn.return %val : i32
  }
  
  // Test write method
  txn.action_method @write_test(%addr: i32, %data: i32) {
    // CHECK: txn.call @mem::@write(%{{.*}}, %{{.*}}) : (i32, i32) -> ()
    txn.call @mem::@write(%addr, %data) : (i32, i32) -> ()
    txn.return
  }
  
  // Test setLatency method
  txn.action_method @set_latency_test(%latency: i32) {
    // CHECK: txn.call @mem::@setLatency(%{{.*}}) : (i32) -> ()
    txn.call @mem::@setLatency(%latency) : (i32) -> ()
    txn.return
  }
  
  // Test getLatency method
  txn.value_method @get_latency_test() -> i32 {
    // CHECK: txn.call @mem::@getLatency() : () -> i32
    %latency = txn.call @mem::@getLatency() : () -> i32
    txn.return %latency : i32
  }
  
  // Test clear method
  txn.action_method @clear_test() {
    // CHECK: txn.call @mem::@clear() : () -> ()
    txn.call @mem::@clear() : () -> ()
    txn.return
  }
  
  // CHECK: txn.schedule
  txn.schedule [@write_test, @set_latency_test, @clear_test]
}

// CHECK-LABEL: txn.module @SpecMemoryLatencyTest
txn.module @SpecMemoryLatencyTest {
  %fast_mem = txn.instance @fast_mem of @SpecMemory<i64> : !txn.module<"SpecMemory">
  %slow_mem = txn.instance @slow_mem of @SpecMemory<i64> : !txn.module<"SpecMemory">
  
  // Initialize memories with different latencies
  txn.action_method @init() {
    %lat1 = arith.constant 1 : i32
    %lat10 = arith.constant 10 : i32
    txn.call @fast_mem::@setLatency(%lat1) : (i32) -> ()
    txn.call @slow_mem::@setLatency(%lat10) : (i32) -> ()
    txn.return
  }
  
  // Cache-like behavior: check fast memory first, then slow
  txn.action_method @read_hierarchical(%addr: i32) -> i64 {
    // In a real implementation, we'd check if data is in fast_mem
    // For this test, we just demonstrate reading from both
    %fast_data = txn.call @fast_mem::@read(%addr) : (i32) -> i64
    %zero = arith.constant 0 : i64
    %is_miss = arith.cmpi eq, %fast_data, %zero : i64
    
    %result = scf.if %is_miss -> i64 {
      // Read from slow memory
      %slow_data = txn.call @slow_mem::@read(%addr) : (i32) -> i64
      // Write to fast memory (cache fill)
      txn.call @fast_mem::@write(%addr, %slow_data) : (i32, i64) -> ()
      scf.yield %slow_data : i64
    } else {
      scf.yield %fast_data : i64
    }
    
    txn.return %result : i64
  }
  
  // Write-through: write to both memories
  txn.action_method @write_through(%addr: i32, %data: i64) {
    txn.call @fast_mem::@write(%addr, %data) : (i32, i64) -> ()
    txn.call @slow_mem::@write(%addr, %data) : (i32, i64) -> ()
    txn.return
  }
  
  txn.schedule [@init, @read_hierarchical, @write_through]
}

// CHECK-LABEL: txn.module @SpecMemoryBurstAccess
txn.module @SpecMemoryBurstAccess {
  %mem = txn.instance @mem of @SpecMemory<i32> : !txn.module<"SpecMemory">
  
  // Burst write: write 4 consecutive addresses
  txn.action_method @burst_write(%base_addr: i32, %d0: i32, %d1: i32, %d2: i32, %d3: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    
    %a0 = arith.addi %base_addr, %c0 : i32
    %a1 = arith.addi %base_addr, %c1 : i32
    %a2 = arith.addi %base_addr, %c2 : i32
    %a3 = arith.addi %base_addr, %c3 : i32
    
    txn.call @mem::@write(%a0, %d0) : (i32, i32) -> ()
    txn.call @mem::@write(%a1, %d1) : (i32, i32) -> ()
    txn.call @mem::@write(%a2, %d2) : (i32, i32) -> ()
    txn.call @mem::@write(%a3, %d3) : (i32, i32) -> ()
    
    txn.return
  }
  
  // Burst read: read 4 consecutive addresses
  txn.value_method @burst_read(%base_addr: i32) -> (i32, i32, i32, i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    
    %a0 = arith.addi %base_addr, %c0 : i32
    %a1 = arith.addi %base_addr, %c1 : i32
    %a2 = arith.addi %base_addr, %c2 : i32
    %a3 = arith.addi %base_addr, %c3 : i32
    
    %d0 = txn.call @mem::@read(%a0) : (i32) -> i32
    %d1 = txn.call @mem::@read(%a1) : (i32) -> i32
    %d2 = txn.call @mem::@read(%a2) : (i32) -> i32
    %d3 = txn.call @mem::@read(%a3) : (i32) -> i32
    
    txn.return %d0, %d1, %d2, %d3 : i32, i32, i32, i32
  }
  
  txn.schedule [@burst_write]
}