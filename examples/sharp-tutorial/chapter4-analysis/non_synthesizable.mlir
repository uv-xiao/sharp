// Module with pre-synthesis violations for testing
txn.module @NonSynthesizable {
  // Violation 1: Using spec primitives (simulation-only)
  %spec_fifo = txn.instance @spec_fifo of @SpecFIFO<i32> : !txn.module<"SpecFIFO">
  %spec_mem = txn.instance @spec_mem of @SpecMemory<i32> : !txn.module<"SpecMemory">
  
  // Normal hardware primitive for comparison
  %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
  
  txn.action_method @useSpecPrimitives(%data: i32) {
    // This will fail synthesis - spec primitives cannot be synthesized
    txn.call @spec_fifo::@enqueue(%data) : (i32) -> ()
    %val = txn.call @spec_mem::@read(%data) : (i32) -> i32
    txn.call @reg::@write(%val) : (i32) -> ()
    txn.yield
  }
  
  txn.schedule [@useSpecPrimitives]
}

txn.module @DisallowedOperations {
  %reg = txn.instance @reg of @Register<f32> : !txn.module<"Register">
  
  txn.action_method @useFloatingPoint(%x: f32, %y: f32) {
    // Violation 2: Using disallowed operations (floating point not in allowlist)
    %sum = arith.addf %x, %y : f32
    %product = arith.mulf %x, %y : f32
    %sin_val = math.sin %x : f32  // math dialect not allowed
    
    txn.call @reg::@write(%sum) : (f32) -> ()
    txn.yield
  }
  
  txn.action_method @useDisallowedArith(%a: i32, %b: i32) {
    // Violation 3: Using disallowed arith operations
    %rem = arith.remsi %a, %b : i32  // remainder not in allowlist
    %div = arith.divsi %a, %b : i32  // division not in allowlist
    
    %result = arith.addi %rem, %div : i32
    %result_f32 = arith.sitofp %result : i32 to f32
    txn.call @reg::@write(%result_f32) : (f32) -> ()
    txn.yield
  }
  
  txn.schedule [@useFloatingPoint, @useDisallowedArith]
}

txn.module @HierarchicalViolation attributes {top} {
  // Violation 4: Instantiating non-synthesizable module
  %non_synth = txn.instance @child of @NonSynthesizable : !txn.module<"NonSynthesizable">
  %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
  
  txn.action_method @useNonSynthChild(%data: i32) {
    // This module becomes non-synthesizable because it uses NonSynthesizable
    txn.call @child::@useSpecPrimitives(%data) : (i32) -> ()
    txn.call @reg::@write(%data) : (i32) -> ()
    txn.yield
  }
  
  txn.schedule [@useNonSynthChild]
}