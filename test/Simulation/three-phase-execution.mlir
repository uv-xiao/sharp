// RUN: sharp-opt %s --sharp-simulate | FileCheck %s
// RUN: sharp-opt %s --sharp-simulate=mode=rtl | FileCheck %s --check-prefix=RTL
// RUN: sharp-opt %s --sharp-simulate=mode=jit | FileCheck %s --check-prefix=JIT

// Comprehensive test for three-phase execution model

// CHECK-LABEL: @ThreePhaseExecution
txn.module @ThreePhaseExecution {
  %counter = txn.instance @counter of @Register<i32> : !txn.module<"Register">
  %cache = txn.instance @cache of @Register<i32> : !txn.module<"Register">
  %valid = txn.instance @valid of @Register<i1> : !txn.module<"Register">
  %fifo = txn.instance @fifo of @FIFO<i32> : !txn.module<"FIFO">
  %wire = txn.instance @wire of @Wire<i32> : !txn.module<"Wire">
  
  // Value method - should be evaluated once per cycle in value phase
  txn.value_method @computeNext() -> i32 attributes {combinational} {
    %current = txn.call @counter::@read() : () -> i32
    %c1 = arith.constant 1 : i32
    %next = arith.addi %current, %c1 : i32
    
    // Complex computation to test caching
    %c10 = arith.constant 10 : i32
    %mod = arith.remsi %next, %c10 : i32
    %squared = arith.muli %mod, %mod : i32
    
    txn.return %squared : i32
  }
  
  // Value method that depends on wire (combinational path)
  txn.value_method @processWire() -> i32 attributes {combinational} {
    %wire_val = txn.call @wire::@read() : () -> i32
    %c2 = arith.constant 2 : i32
    %doubled = arith.muli %wire_val, %c2 : i32
    txn.return %doubled : i32
  }
  
  // Action method - execution phase
  txn.action_method @increment() {
    // Value phase: compute next value (should use cached result)
    %next = txn.call @computeNext() : () -> i32
    
    // Execution phase: perform state updates
    txn.call @counter::@write(%next) : (i32) -> ()
    
    // Update cache with computed value
    txn.call @cache::@write(%next) : (i32) -> ()
    
    // Set valid flag
    %c_true = arith.constant true
    txn.call @valid::@write(%c_true) : (i1) -> ()
    
    txn.yield
  }
  
  // Rule with complex guard - tests guard evaluation in value phase
  txn.rule @producer {
    // Value phase: evaluate guards
    %is_valid = txn.call @valid::@read() : () -> i1
    %can_enq = txn.call @fifo::@canEnq() : () -> i1
    %both_ready = arith.andi %is_valid, %can_enq : i1
    
    txn.if %both_ready {
      // Value phase: get cached computation result
      %cached = txn.call @cache::@read() : () -> i32
      %processed = txn.call @processWire() : () -> i32
      %sum = arith.addi %cached, %processed : i32
      
      // Execution phase: perform actions
      txn.call @fifo::@enq(%sum) : (i32) -> ()
      
      // Clear valid flag
      %c_false = arith.constant false
      txn.call @valid::@write(%c_false) : (i1) -> ()
      txn.yield
    } else {
      // No action - wait for valid data or FIFO space
      txn.yield
    }
    txn.yield
  }
  
  // Consumer rule - tests FIFO dequeue in execution phase
  txn.rule @consumer {
    %can_deq = txn.call @fifo::@canDeq() : () -> i1
    
    txn.if %can_deq {
      // Value phase: peek at FIFO
      %data = txn.call @fifo::@first() : () -> i32
      
      // Execution phase: dequeue and process
      txn.call @fifo::@deq() : () -> ()
      
      // Write to wire (affects combinational paths)
      txn.call @wire::@write(%data) : (i32) -> ()
      
      // Complex processing
      %c100 = arith.constant 100 : i32
      %is_large = arith.cmpi sgt, %data, %c100 : i32
      txn.if %is_large {
        // Reset counter on large values
        %c0 = arith.constant 0 : i32
        txn.call @counter::@write(%c0) : (i32) -> ()
        txn.yield
      } else {
        // Keep counter value unchanged
        txn.yield
      }
      txn.yield
    } else {
      // No data to dequeue - skip
      txn.yield
    }
    txn.yield
  }
  
  // Action with abort - tests rollback in commit phase
  txn.action_method @conditionalUpdate(%threshold: i32) {
    %current = txn.call @counter::@read() : () -> i32
    
    %exceeds = arith.cmpi sgt, %current, %threshold : i32
    txn.if %exceeds {
      // This should trigger rollback
      txn.abort
    } else {
      // Continue with normal execution
      txn.yield
    }
    
    // These updates should only commit if no abort
    %c10 = arith.constant 10 : i32
    %new_val = arith.addi %current, %c10 : i32
    txn.call @counter::@write(%new_val) : (i32) -> ()
    
    txn.yield
  }
  
  // Test execution order within a cycle
  txn.rule @orderTest1 {
    %val = txn.call @counter::@read() : () -> i32
    %c1 = arith.constant 1 : i32
    %inc = arith.addi %val, %c1 : i32
    txn.call @wire::@write(%inc) : (i32) -> ()
    txn.yield
  }
  
  txn.rule @orderTest2 {
    // This should see the value written by orderTest1 if scheduled after
    %wire_val = txn.call @wire::@read() : () -> i32
    %c2 = arith.constant 2 : i32
    %is_even = arith.remsi %wire_val, %c2 : i32
    %c0 = arith.constant 0 : i32
    %even = arith.cmpi eq, %is_even, %c0 : i32
    
    txn.if %even {
      txn.call @increment() : () -> ()
      txn.yield
    } else {
      // Skip increment for odd values
      txn.yield
    }
    txn.yield
  }
  
  txn.schedule [@increment, @producer, @consumer, @conditionalUpdate, @orderTest1, @orderTest2] {
    conflict_matrix = {
      // Self conflicts
      "increment,increment" = 2 : i32,
      "producer,producer" = 2 : i32,
      "consumer,consumer" = 2 : i32,
      "conditionalUpdate,conditionalUpdate" = 2 : i32,
      "orderTest1,orderTest1" = 2 : i32,
      "orderTest2,orderTest2" = 2 : i32,
      
      // Sequential relationships for ordering
      "orderTest1,orderTest2" = 0 : i32,  // orderTest1 before orderTest2
      "producer,consumer" = 0 : i32,      // producer before consumer
      
      // Conflicts
      "increment,conditionalUpdate" = 2 : i32,  // Both write counter
      "increment,consumer" = 2 : i32,           // consumer might reset counter
      "conditionalUpdate,consumer" = 2 : i32,   // Both write counter
      "orderTest1,consumer" = 2 : i32,          // Both write wire
      "increment,orderTest2" = 2 : i32,         // orderTest2 calls increment
      
      // Some are conflict-free
      "producer,orderTest1" = 3 : i32,
      "producer,orderTest2" = 3 : i32
    }
  }
}

// CHECK: Simulation started
// CHECK: === Cycle 0 ===
// CHECK: Value Phase:
// CHECK-DAG: computeNext() = 
// CHECK-DAG: processWire() = 
// CHECK: Execution Phase:
// CHECK: Commit Phase:
// CHECK: State updates applied

// RTL mode specific checks
// RTL: RTL Simulation Mode
// RTL: Generating RTL events
// RTL: Clock edge processing

// JIT mode specific checks  
// JIT: JIT Compilation Mode
// JIT: Compiling value methods
// JIT: Executing compiled code

// Test module for phase separation
// CHECK-LABEL: @PhaseVisibility
txn.module @PhaseVisibility {
  %state = txn.instance @state of @Register<i32> : !txn.module<"Register">
  %temp = txn.instance @temp of @Wire<i32> : !txn.module<"Wire">
  
  // This tests that wire writes in execution phase are visible in same cycle
  txn.rule @writer {
    %c42 = arith.constant 42 : i32
    txn.call @temp::@write(%c42) : (i32) -> ()
    txn.yield
  }
  
  txn.rule @reader {
    %val = txn.call @temp::@read() : () -> i32
    %c0 = arith.constant 0 : i32
    %is_set = arith.cmpi ne, %val, %c0 : i32
    
    txn.if %is_set {
      txn.call @state::@write(%val) : (i32) -> ()
      txn.yield
    } else {
      // No update - wire value is zero
      txn.yield
    }
    txn.yield
  }
  
  // Sequential schedule ensures wire write is visible
  txn.schedule [@writer, @reader] {
    conflict_matrix = {
      "writer,writer" = 2 : i32,
      "reader,reader" = 2 : i32,
      "writer,reader" = 0 : i32  // writer before reader
    }
  }
}

// CHECK: @PhaseVisibility simulation
// CHECK: writer executes in phase
// CHECK: reader sees wire value = 42
// CHECK: state updated to 42