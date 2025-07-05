// RUN: sharp-opt %s | FileCheck %s

// CHECK-LABEL: txn.module @MemoryTest
txn.module @MemoryTest {
  // CHECK: %mem = txn.instance @mem of @Memory<i32>
  %mem = txn.instance @mem of @Memory<i32> : !txn.module<"Memory">
  
  // Test read method
  txn.value_method @read_test(%addr: i32) -> i32 {
    // CHECK: txn.call @mem::@read(%{{.*}}) : (i32) -> i32
    %val = txn.call @mem::@read(%addr) : (i32) -> i32
    txn.return %val : i32
  }
  
  // Test write method
  txn.action_method @write_test(%addr: i32, %data: i32) {
    // CHECK: txn.call @mem::@write(%{{.*}}, %{{.*}}) : (i32, i32) -> ()
    txn.call @mem::@write(%addr, %data) : (i32, i32) -> ()
    txn.yield
  }
  
  // Test clear method
  txn.action_method @clear_test() {
    // CHECK: txn.call @mem::@clear() : () -> ()
    txn.call @mem::@clear() : () -> ()
    txn.yield
  }
  
  // Test read-modify-write pattern
  txn.action_method @increment(%addr: i32) {
    %val = txn.call @mem::@read(%addr) : (i32) -> i32
    %one = arith.constant 1 : i32
    %new_val = arith.addi %val, %one : i32
    txn.call @mem::@write(%addr, %new_val) : (i32, i32) -> ()
    txn.yield
  }
  
  // CHECK: txn.schedule
  txn.schedule [@read_test, @write_test, @clear_test, @increment]
}

// CHECK-LABEL: txn.module @MultiPortMemoryUsage
txn.module @MultiPortMemoryUsage {
  %mem1 = txn.instance @mem1 of @Memory<i64> : !txn.module<"Memory">
  %mem2 = txn.instance @mem2 of @Memory<i64> : !txn.module<"Memory">
  
  // Copy data from one memory to another
  txn.action_method @copy(%src_addr: i32, %dst_addr: i32) {
    %data = txn.call @mem1::@read(%src_addr) : (i32) -> i64
    txn.call @mem2::@write(%dst_addr, %data) : (i32, i64) -> ()
    txn.yield
  }
  
  // Swap data between memories
  txn.action_method @swap(%addr1: i32, %addr2: i32) {
    %data1 = txn.call @mem1::@read(%addr1) : (i32) -> i64
    %data2 = txn.call @mem2::@read(%addr2) : (i32) -> i64
    txn.call @mem1::@write(%addr1, %data2) : (i32, i64) -> ()
    txn.call @mem2::@write(%addr2, %data1) : (i32, i64) -> ()
    txn.yield
  }
  
  txn.schedule [@copy, @swap]
}