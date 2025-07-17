// RUN: sharp-opt --convert-txn-to-firrtl %s | FileCheck %s

// Test nested module hierarchy with multiple instances
txn.module @Counter {
  txn.value_method @get() -> i32 {
    %c0 = arith.constant 0 : i32
    txn.return %c0 : i32
  }
  
  txn.action_method @inc() {
    txn.return
  }
  
  txn.schedule [@inc] {
    conflict_matrix = {
      "inc,inc" = 2 : i32
    }
  }
}

txn.module @DualCounter {
  %low = txn.instance @low of @Counter : index
  %high = txn.instance @high of @Counter : index
  
  txn.action_method @increment() {
    // Increment low counter
    txn.call @low::@inc() : () -> ()
    
    // Check if we need to increment high
    %low_val = txn.call @low::@get() : () -> i32
    %c10 = arith.constant 10 : i32
    %overflow = arith.cmpi eq, %low_val, %c10 : i32
    
    txn.if %overflow {
      txn.call @high::@inc() : () -> ()
      txn.yield
    } else {
      txn.yield
    }
    
    txn.return
  }
  
  txn.value_method @read() -> i32 {
    %high_val = txn.call @high::@get() : () -> i32
    %low_val = txn.call @low::@get() : () -> i32
    %c10 = arith.constant 10 : i32
    %high_mult = arith.muli %high_val, %c10 : i32
    %total = arith.addi %high_mult, %low_val : i32
    txn.return %total : i32
  }
  
  txn.schedule [@increment] {
    conflict_matrix = {
      "increment,increment" = 2 : i32
    }
  }
}

txn.module @System {
  %counter1 = txn.instance @counter1 of @DualCounter : index
  %counter2 = txn.instance @counter2 of @DualCounter : index
  
  txn.rule @sync_counters {
    %val1 = txn.call @counter1::@read() : () -> i32
    %val2 = txn.call @counter2::@read() : () -> i32
    %diff = arith.subi %val1, %val2 : i32
    %c5 = arith.constant 5 : i32
    %need_sync = arith.cmpi sgt, %diff, %c5 : i32
    
    txn.if %need_sync {
      txn.call @counter2::@increment() : () -> ()
      txn.yield
    } else {
      txn.yield
    }
    
    txn.return
  }
  
  txn.schedule [@sync_counters] {
    conflict_matrix = {}
  }
}

// CHECK: module {
// CHECK-NEXT: firrtl.circuit "System" {

// Check all modules exist in order
// CHECK-DAG: firrtl.module @Counter
// CHECK-DAG: firrtl.module @DualCounter
// CHECK-DAG: firrtl.module @System

// Check instances are created
// CHECK-DAG: firrtl.instance low interesting_name @Counter
// CHECK-DAG: firrtl.instance high interesting_name @Counter
// CHECK-DAG: firrtl.instance counter1 interesting_name @DualCounter
// CHECK-DAG: firrtl.instance counter2 interesting_name @DualCounter

// Check key operations in the conversion
// Method calls to instances use connect
// CHECK-DAG: firrtl.connect %low_incEN
// CHECK-DAG: firrtl.connect %high_incEN
// CHECK-DAG: firrtl.connect %counter2_incrementEN

// Arithmetic operations are converted
// CHECK-DAG: firrtl.mul
// CHECK-DAG: firrtl.add
// CHECK-DAG: firrtl.sub
// CHECK-DAG: firrtl.gt

// When blocks for actions
// CHECK-DAG: firrtl.when %increment_wf
// CHECK-DAG: firrtl.when %sync_counters_wf