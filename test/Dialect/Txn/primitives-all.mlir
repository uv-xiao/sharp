// RUN: sharp-opt %s | FileCheck %s
// RUN: sharp-opt %s --sharp-validate-method-attributes | FileCheck %s --check-prefix=VALIDATE

// Comprehensive test for all primitive types and their interactions

// CHECK-LABEL: txn.module @AllPrimitives
txn.module @AllPrimitives {
  // Hardware primitives with various types
  %reg_i32 = txn.instance @reg_i32 of @Register<i32> : !txn.module<"Register">
  %reg_i64 = txn.instance @reg_i64 of @Register<i64> : !txn.module<"Register">
  %reg_i1 = txn.instance @reg_i1 of @Register<i1> : !txn.module<"Register">
  
  %wire_i32 = txn.instance @wire_i32 of @Wire<i32> : !txn.module<"Wire">
  %wire_vec = txn.instance @wire_vec of @Wire<vector<4xi32>> : !txn.module<"Wire">
  
  %fifo_i32 = txn.instance @fifo_i32 of @FIFO<i32> : !txn.module<"FIFO">
  %fifo_i64 = txn.instance @fifo_i64 of @FIFO<i64> : !txn.module<"FIFO">
  
  %mem_i32 = txn.instance @mem_i32 of @Memory<i32> : !txn.module<"Memory">
  %mem_i8 = txn.instance @mem_i8 of @Memory<i8> : !txn.module<"Memory">
  
  // Spec primitives
  %spec_fifo = txn.instance @spec_fifo of @SpecFIFO<i32> : !txn.module<"SpecFIFO">
  %spec_mem = txn.instance @spec_mem of @SpecMemory<i64> : !txn.module<"SpecMemory">
  
  // Complex interactions between primitives
  txn.action_method @transferData(%src_addr: i32, %dst_addr: i32) {
    // Read from memory
    %data = txn.call @mem_i32::@read(%src_addr) : (i32) -> i32
    
    // Check if we can enqueue
    %can_enq = txn.call @fifo_i32::@canEnq() : () -> i1
    txn.if %can_enq {
      // Enqueue to FIFO
      txn.call @fifo_i32::@enq(%data) : (i32) -> ()
      
      // Update wire with success status
      %c1 = arith.constant 1 : i32
      txn.call @wire_i32::@write(%c1) : (i32) -> ()
      txn.yield
    } else {
      // Write to alternative memory location
      txn.call @mem_i32::@write(%dst_addr, %data) : (i32, i32) -> ()
      
      // Update wire with failure status
      %c0 = arith.constant 0 : i32
      txn.call @wire_i32::@write(%c0) : (i32) -> ()
      txn.yield
    }
    txn.yield
  }
  
  // Test all FIFO operations
  txn.rule @fifoOperations {
    // Check both FIFOs
    %can_deq_32 = txn.call @fifo_i32::@canDeq() : () -> i1
    %can_enq_64 = txn.call @fifo_i64::@canEnq() : () -> i1
    
    %both_ready = arith.andi %can_deq_32, %can_enq_64 : i1
    
    txn.if %both_ready {
      // Transfer from 32-bit to 64-bit FIFO with extension
      %data_32 = txn.call @fifo_i32::@first() : () -> i32
      txn.call @fifo_i32::@deq() : () -> ()
      
      %data_64 = arith.extsi %data_32 : i32 to i64
      txn.call @fifo_i64::@enq(%data_64) : (i64) -> ()
      
      // Clear the FIFOs if requested
      %clear_flag = txn.call @reg_i1::@read() : () -> i1
      txn.if %clear_flag {
        txn.call @fifo_i32::@clear() : () -> ()
        txn.call @fifo_i64::@clear() : () -> ()
        txn.yield
      } else {
        // No clear needed
        txn.yield
      }
    } else {
      // Not ready
      txn.yield
    }
    txn.yield
  }
  
  // Test memory operations with different widths
  txn.action_method @memoryConversion(%addr: i32) -> i32 {
    // Read 8-bit value
    %val_i8 = txn.call @mem_i8::@read(%addr) : (i32) -> i8
    
    // Convert to 32-bit
    %val_i32 = arith.extui %val_i8 : i8 to i32
    
    // Store in register
    txn.call @reg_i32::@write(%val_i32) : (i32) -> ()
    
    // Also store in 32-bit memory
    %c0 = arith.constant 0 : i32
    txn.call @mem_i32::@write(%c0, %val_i32) : (i32, i32) -> ()
    
    txn.return %val_i32 : i32
  }
  
  // Test vector operations on wire
  txn.rule @vectorOperations {
    // Create a vector value
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %vec = vector.from_elements %c1, %c2, %c3, %c4 : vector<4xi32>
    
    // Write to vector wire
    txn.call @wire_vec::@write(%vec) : (vector<4xi32>) -> ()
    txn.yield
  }
  
  // Test spec primitives
  txn.action_method @specOperations(%enable: i1) {
    txn.if %enable {
      // SpecFIFO operations
      %can_enq = txn.call @spec_fifo::@canEnq() : () -> i1
      txn.if %can_enq {
        %c42 = arith.constant 42 : i32
        txn.call @spec_fifo::@enq(%c42) : (i32) -> ()
        
        // Reserve an entry
        %token = txn.call @spec_fifo::@reserve() : () -> i32
        
        // Commit later
        txn.call @spec_fifo::@commit(%token) : (i32) -> ()
        txn.yield
      } else {
        // Cannot enqueue
        txn.yield
      }
      
      // SpecMemory operations
      %c10 = arith.constant 10 : i32
      %c100 = arith.constant 100 : i64
      
      // Speculative write
      %write_token = txn.call @spec_mem::@specWrite(%c10, %c100) : (i32, i64) -> i32
      
      // Speculative read
      %spec_val = txn.call @spec_mem::@specRead(%c10) : (i32) -> i64
      
      // Decide whether to commit based on value
      %c50 = arith.constant 50 : i64
      %should_commit = arith.cmpi sgt, %spec_val, %c50 : i64
      
      txn.if %should_commit {
        txn.call @spec_mem::@commit(%write_token) : (i32) -> ()
        txn.yield
      } else {
        txn.call @spec_mem::@abort(%write_token) : (i32) -> ()
        txn.yield
      }
    } else {
      // Spec operations disabled
      txn.yield
    }
    txn.yield
  }
  
  // Test all register types
  txn.rule @registerTypes {
    // Read all register types
    %val_32 = txn.call @reg_i32::@read() : () -> i32
    %val_64 = txn.call @reg_i64::@read() : () -> i64
    %val_1 = txn.call @reg_i1::@read() : () -> i1
    
    // Perform computations
    %extended = arith.extsi %val_32 : i32 to i64
    %sum = arith.addi %val_64, %extended : i64
    
    // Conditional write based on boolean
    txn.if %val_1 {
      %truncated = arith.trunci %sum : i64 to i32
      txn.call @reg_i32::@write(%truncated) : (i32) -> ()
      txn.yield
    } else {
      txn.call @reg_i64::@write(%sum) : (i64) -> ()
      txn.yield
    }
    txn.yield
  }
  
  // Method with attributes for primitive calls
  txn.action_method @attributedMethod() attributes {
    signal_prefix = "prim",
    ready_suffix = "_rdy",
    enable_suffix = "_en"
  } {
    %data = txn.call @wire_i32::@read() : () -> i32
    %c0 = arith.constant 0 : i32
    %is_valid = arith.cmpi ne, %data, %c0 : i32
    
    txn.if %is_valid {
      // Chain of primitive operations
      txn.call @reg_i32::@write(%data) : (i32) -> ()
      %doubled = arith.muli %data, %data : i32
      txn.call @mem_i32::@write(%c0, %doubled) : (i32, i32) -> ()
      txn.yield
    } else {
      // Invalid data
      txn.yield
    }
    txn.yield
  }
  
  txn.schedule [@transferData, @fifoOperations, @memoryConversion, 
                @vectorOperations, @specOperations, @registerTypes, 
                @attributedMethod] {
    conflict_matrix = {
      // Self conflicts
      "transferData,transferData" = 2 : i32,
      "fifoOperations,fifoOperations" = 2 : i32,
      "memoryConversion,memoryConversion" = 2 : i32,
      "vectorOperations,vectorOperations" = 2 : i32,
      "specOperations,specOperations" = 2 : i32,
      "registerTypes,registerTypes" = 2 : i32,
      "attributedMethod,attributedMethod" = 2 : i32,
      
      // Cross-method conflicts based on shared resources
      "transferData,memoryConversion" = 2 : i32,     // Both use mem_i32
      "fifoOperations,transferData" = 2 : i32,       // Both use fifo_i32
      "registerTypes,attributedMethod" = 2 : i32,    // Both use reg_i32
      "vectorOperations,attributedMethod" = 2 : i32, // Both use wire_i32
      
      // Conflict-free pairs
      "specOperations,vectorOperations" = 3 : i32,   // Different resources
      "specOperations,registerTypes" = 3 : i32,      // Different resources
      "fifoOperations,memoryConversion" = 3 : i32    // Different resources
    }
  }
}

// VALIDATE: Method attribute validation passed

// CHECK: txn.instance @reg_i32 of @Register<i32>
// CHECK: txn.instance @reg_i64 of @Register<i64>
// CHECK: txn.instance @reg_i1 of @Register<i1>
// CHECK: txn.instance @wire_i32 of @Wire<i32>
// CHECK: txn.instance @wire_vec of @Wire<vector<4xi32>>
// CHECK: txn.instance @fifo_i32 of @FIFO<i32>
// CHECK: txn.instance @fifo_i64 of @FIFO<i64>
// CHECK: txn.instance @mem_i32 of @Memory<i32>
// CHECK: txn.instance @mem_i8 of @Memory<i8>
// CHECK: txn.instance @spec_fifo of @SpecFIFO<i32>
// CHECK: txn.instance @spec_mem of @SpecMemory<i64>

// CHECK: txn.action_method @transferData
// CHECK: txn.rule @fifoOperations
// CHECK: txn.action_method @memoryConversion
// CHECK: txn.rule @vectorOperations
// CHECK: txn.action_method @specOperations
// CHECK: txn.rule @registerTypes
// CHECK: txn.action_method @attributedMethod