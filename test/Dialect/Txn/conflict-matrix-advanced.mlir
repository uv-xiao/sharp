// RUN: sharp-opt %s --sharp-infer-conflict-matrix -allow-unregistered-dialect | FileCheck %s
// RUN: sharp-opt %s --sharp-infer-conflict-matrix --sharp-validate-conflicts -allow-unregistered-dialect | FileCheck %s --check-prefix=VALIDATE

// Advanced conflict matrix test with all relation types and complex inference

// CHECK-LABEL: txn.module @ConflictInferenceTest
txn.module @ConflictInferenceTest {
  // Multiple primitive instances to create complex conflicts
  %reg1 = txn.instance @reg1 of @Register<i32> : !txn.module<"Register">
  %reg2 = txn.instance @reg2 of @Register<i32> : !txn.module<"Register">
  %wire = txn.instance @wire of @Wire<i32> : !txn.module<"Wire">
  %fifo = txn.instance @fifo of @FIFO<i32> : !txn.module<"FIFO">
  
  // Actions with varying conflict relationships
  txn.rule @r1 {
    %v = txn.call @reg1::@read() : () -> i32
    txn.call @reg1::@write(%v) : (i32) -> ()  // Read-write same register
    txn.yield
  }
  
  txn.rule @r2 {
    %v = txn.call @reg2::@read() : () -> i32
    txn.call @wire::@write(%v) : (i32) -> ()  // Read reg2, write wire
    txn.yield
  }
  
  txn.rule @r3 {
    %v = txn.call @wire::@read() : () -> i32
    txn.call @reg2::@write(%v) : (i32) -> ()  // Read wire, write reg2
    txn.yield
  }
  
  txn.action_method @enqueue_data(%data: i32) {
    %can_enq = txn.call @fifo::@canEnq() : () -> i1
    txn.if %can_enq {
      txn.call @fifo::@enq(%data) : (i32) -> ()
      txn.yield
    } else {
      txn.abort  // Abort if FIFO full
    }
    txn.yield
  }
  
  txn.action_method @dequeue_data() -> i32 {
    %can_deq = txn.call @fifo::@canDeq() : () -> i1
    %result = txn.if %can_deq -> i32 {
      %data = txn.call @fifo::@first() : () -> i32
      txn.call @fifo::@deq() : () -> ()
      txn.yield %data : i32
    } else {
      txn.abort  // Abort if FIFO empty
    }
    txn.return %result : i32
  }
  
  txn.rule @r4 {
    // Complex rule calling multiple methods
    %v1 = txn.call @reg1::@read() : () -> i32
    %v2 = txn.call @reg2::@read() : () -> i32
    %sum = arith.addi %v1, %v2 : i32
    
    // Conditional writes create complex conflicts
    %c10 = arith.constant 10 : i32
    %cond = arith.cmpi sgt, %sum, %c10 : i32
    txn.if %cond {
      txn.call @enqueue_data(%sum) : (i32) -> ()
      txn.yield
    } else {
      txn.call @wire::@write(%sum) : (i32) -> ()
      txn.yield
    }
    txn.yield
  }
  
  // Partial conflict matrix - inference should complete it
  txn.schedule [@r1, @r2, @r3, @enqueue_data, @dequeue_data, @r4] {
    conflict_matrix = {
      // Explicitly specified conflicts
      "r1,r2" = 3 : i32,  // CF: Different registers
      "r2,r3" = 0 : i32,  // SB: r2 writes wire, r3 reads it
      "enqueue_data,dequeue_data" = 2 : i32  // C: FIFO operations conflict
      
      // Let inference handle:
      // - Self conflicts (all should be C)
      // - r1,r3 (should be CF - different resources)
      // - r1,r4 (should be C - both access reg1)
      // - r2,r4 (should be CF or depends on wire usage)
      // - r3,r4 (complex - depends on conditionals)
      // - r4 with enqueue/dequeue (conditional conflicts)
    }
  }
}

// CHECK: txn.schedule [@r1, @r2, @r3, @enqueue_data, @dequeue_data, @r4] {
// CHECK-DAG: "r1,r1" = 2 : i32
// CHECK-DAG: "r2,r2" = 2 : i32
// CHECK-DAG: "r3,r3" = 2 : i32
// CHECK-DAG: "r4,r4" = 2 : i32
// CHECK-DAG: "enqueue_data,enqueue_data" = 2 : i32
// CHECK-DAG: "dequeue_data,dequeue_data" = 2 : i32
// CHECK-DAG: "r1,r4" = 2 : i32
// CHECK-DAG: "r4,enqueue_data" = 2 : i32

// VALIDATE-LABEL: Conflict validation passed

// CHECK-LABEL: txn.module @TransitiveConflicts
txn.module @TransitiveConflicts {
  %shared = txn.instance @shared of @Register<i64> : !txn.module<"Register">
  
  // Create a chain of dependencies
  txn.action_method @a1() {
    %c1 = arith.constant 1 : i64
    txn.call @shared::@write(%c1) : (i64) -> ()
    txn.yield
  }
  
  txn.action_method @a2() {
    txn.call @a1() : () -> ()  // a2 calls a1
    txn.yield
  }
  
  txn.action_method @a3() {
    txn.call @a2() : () -> ()  // a3 calls a2 which calls a1
    txn.yield
  }
  
  txn.action_method @b1() {
    %v = txn.call @shared::@read() : () -> i64
    txn.yield
  }
  
  txn.action_method @b2() {
    txn.call @b1() : () -> ()  // b2 calls b1
    txn.yield
  }
  
  // Inference should determine transitive conflicts:
  // a1 C b1 (write C read on shared)
  // Therefore: a2 C b1, a3 C b1, a1 C b2, a2 C b2, a3 C b2
  txn.schedule [@a1, @a2, @a3, @b1, @b2] {
    conflict_matrix = {
      // Only specify the base conflict
      "a1,b1" = 2 : i32  // C: write conflicts with read
    }
  }
}

// CHECK: txn.schedule [@a1, @a2, @a3, @b1, @b2] {
// CHECK-DAG: "a1,b1" = 2 : i32
// CHECK-DAG: "a2,b1" = 2 : i32
// CHECK-DAG: "a3,b1" = 2 : i32
// CHECK-DAG: "a1,b2" = 2 : i32
// CHECK-DAG: "a2,b2" = 2 : i32
// CHECK-DAG: "a3,b2" = 2 : i32

// Test complex scheduling with all conflict types
// CHECK-LABEL: txn.module @AllConflictTypes
txn.module @AllConflictTypes {
  %r = txn.instance @r of @Register<i32> : !txn.module<"Register">
  
  // Sequential before (SB)
  txn.rule @writer {
    %c42 = arith.constant 42 : i32
    txn.call @r::@write(%c42) : (i32) -> ()
    txn.yield
  }
  
  txn.rule @reader {
    %v = txn.call @r::@read() : () -> i32
    // Use the value to prevent optimization
    "test.use"(%v) : (i32) -> ()
    txn.yield
  }
  
  // Conflict (C)
  txn.rule @writer2 {
    %c100 = arith.constant 100 : i32
    txn.call @r::@write(%c100) : (i32) -> ()
    txn.yield
  }
  
  // Conflict-free (CF)
  txn.rule @independent {
    %c1 = arith.constant 1 : i32
    "test.independent"(%c1) : (i32) -> ()
    txn.yield
  }
  
  txn.schedule [@writer, @reader, @writer2, @independent] {
    conflict_matrix = {
      // All self-conflicts
      "writer,writer" = 2 : i32,
      "reader,reader" = 2 : i32,
      "writer2,writer2" = 2 : i32,
      "independent,independent" = 2 : i32,
      
      // Sequencing
      "writer,reader" = 0 : i32,   // SB: Write before read
      "reader,writer" = 1 : i32,   // SA: Read after write (inverse)
      
      // Conflicts
      "writer,writer2" = 2 : i32,  // C: Both write same register
      "writer2,writer" = 2 : i32,  // C: Symmetric
      "reader,writer2" = 2 : i32,  // C: Read conflicts with different write
      "writer2,reader" = 2 : i32,  // C: Symmetric
      
      // Conflict-free
      "independent,writer" = 3 : i32,
      "independent,reader" = 3 : i32,
      "independent,writer2" = 3 : i32,
      "writer,independent" = 3 : i32,
      "reader,independent" = 3 : i32,
      "writer2,independent" = 3 : i32
    }
  }
}

// Validate all relationships are properly encoded
// VALIDATE: Schedule validation completed successfully