// RUN: sharp-opt %s --sharp-primitive-gen --sharp-infer-conflict-matrix --sharp-reachability-analysis --sharp-general-check -split-input-file -verify-diagnostics

// Test 1: Valid value method with no conflicts
txn.module @ValidValueMethod {
  %reg = txn.instance @reg of @Register<i32> : index
  
  // This value method has no conflicts specified, so it's valid
  txn.value_method @getValue() -> i32 {
    %v = txn.call @reg::@read() : () -> i32
    txn.return %v : i32
  }
  
  txn.action_method @setValue(%v: i32) {
    txn.call @reg::@write(%v) : (i32) -> ()
    txn.return
  }
  
  // No conflicts specified - all default to CF
  txn.schedule [@setValue]
}

// -----

// Test 2: Invalid value method with SB conflict
txn.module @ValueMethodWithSBConflict {
  %reg = txn.instance @reg of @Register<i32> : index
  
  // expected-error@+1 {{value method 'getValue' has non-CF conflict with action 'setValue' (SB (Sequence Before))}}
  txn.value_method @getValue() -> i32 {
    %v = txn.call @reg::@read() : () -> i32
    txn.return %v : i32
  }
  
  txn.action_method @setValue(%v: i32) {
    txn.call @reg::@write(%v) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@setValue] {
    conflict_matrix = {
      "getValue,setValue" = 0 : i32  // SB
    }
  }
}

// -----

// Test 3: Invalid value method with SA conflict
txn.module @ValueMethodWithSAConflict {
  %wire = txn.instance @wire of @Wire<i32> : index
  
  // expected-error@+1 {{value method 'readWire' has non-CF conflict with action 'writeWire' (SA (Sequence After))}}
  txn.value_method @readWire() -> i32 {
    %v = txn.call @wire::@read() : () -> i32
    txn.return %v : i32
  }
  
  txn.action_method @writeWire(%v: i32) {
    txn.call @wire::@write(%v) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@writeWire] {
    conflict_matrix = {
      "writeWire,readWire" = 1 : i32  // SA
    }
  }
}

// -----

// Test 4: Invalid value method with C conflict
txn.module @ValueMethodWithConflict {
  %reg = txn.instance @reg of @Register<i32> : index
  
  // expected-error@+1 {{value method 'compute' has non-CF conflict with action 'update' (C (Conflict))}}
  txn.value_method @compute() -> i32 {
    %v = txn.call @reg::@read() : () -> i32
    %two = arith.constant 2 : i32
    %result = arith.muli %v, %two : i32
    txn.return %result : i32
  }
  
  txn.action_method @update() {
    %v = txn.call @compute() : () -> i32
    txn.call @reg::@write(%v) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@update] {
    conflict_matrix = {
      "compute,update" = 2 : i32  // C
    }
  }
}

// -----

// Test 5: Valid module with explicit CF conflicts
txn.module @ExplicitCFConflicts {
  %reg = txn.instance @reg of @Register<i32> : index
  
  // This is valid - CF conflicts are allowed for value methods
  txn.value_method @getValue() -> i32 {
    %v = txn.call @reg::@read() : () -> i32
    txn.return %v : i32
  }
  
  txn.action_method @setValue(%v: i32) {
    txn.call @reg::@write(%v) : (i32) -> ()
    txn.return
  }
  
  txn.rule @incrementRule {
    %v = txn.call @getValue() : () -> i32
    %one = arith.constant 1 : i32
    %next = arith.addi %v, %one : i32
    txn.call @setValue(%next) : (i32) -> ()
    txn.yield
  }
  
  // Explicit CF relationships are fine
  txn.schedule [@setValue, @incrementRule] {
    conflict_matrix = {
      "getValue,setValue" = 3 : i32,  // CF
      "getValue,incrementRule" = 3 : i32  // CF
    }
  }
}

// -----

// Test 6: Multiple value methods with conflicts
txn.module @MultipleValueMethodsWithConflicts {
  %reg = txn.instance @reg of @Register<i32> : index
  
  // expected-error@+1 {{value method 'getValue1' has non-CF conflict with action 'setValue' (SB (Sequence Before))}}
  txn.value_method @getValue1() -> i32 {
    %v = txn.call @reg::@read() : () -> i32
    txn.return %v : i32
  }
  
  // expected-error@+1 {{value method 'getValue2' has non-CF conflict with action 'setValue' (C (Conflict))}}
  txn.value_method @getValue2() -> i32 {
    %v = txn.call @reg::@read() : () -> i32
    %two = arith.constant 2 : i32
    %result = arith.addi %v, %two : i32
    txn.return %result : i32
  }
  
  txn.action_method @setValue(%v: i32) {
    txn.call @reg::@write(%v) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@setValue] {
    conflict_matrix = {
      "getValue1,setValue" = 0 : i32,  // SB
      "getValue2,setValue" = 2 : i32  // C
    }
  }
}