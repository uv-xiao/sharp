// RUN: sharp-opt --sharp-detect-combinational-loops --split-input-file --verify-diagnostics %s

// Test combinational loop detection

// Direct loop through value methods
// expected-error@+1 {{Combinational loop detected}}
txn.module @DirectLoop {
  txn.value_method @getValue() -> i32 {
    %val = txn.call @self::@compute() : () -> i32
    txn.return %val : i32
  }
  
  txn.value_method @compute() -> i32 {
    %val = txn.call @self::@getValue() : () -> i32
    %c1 = arith.constant 1 : i32
    %result = arith.addi %val, %c1 : i32
    txn.return %result : i32
  }
  
  %self = txn.instance @self of @DirectLoop : !txn.module<"DirectLoop">
  txn.schedule [@getValue, @compute]
}

// -----

// Loop through multiple value methods
// expected-error@+1 {{Combinational loop detected}}
txn.module @MultiLoop {
  txn.value_method @getA() -> i32 {
    %val = txn.call @self::@getB() : () -> i32
    txn.return %val : i32
  }
  
  txn.value_method @getB() -> i32 {
    %val = txn.call @self::@getC() : () -> i32
    txn.return %val : i32
  }
  
  txn.value_method @getC() -> i32 {
    %val = txn.call @self::@getA() : () -> i32
    txn.return %val : i32
  }
  
  %self = txn.instance @self of @MultiLoop : !txn.module<"MultiLoop">
  txn.schedule [@getA, @getB, @getC]
}

// -----

// No loop - sequential through Register
txn.module @NoLoopRegister {
  %reg = txn.instance @reg of @Register : !txn.module<"Register">
  
  // This is OK - Register breaks combinational paths
  txn.value_method @getValue() -> i32 {
    %val = txn.call @reg::@read() : () -> i32
    txn.return %val : i32
  }
  
  txn.rule @update {
    %val = txn.call @self::@getValue() : () -> i32
    %c1 = arith.constant 1 : i32
    %inc = arith.addi %val, %c1 : i32
    txn.call @reg::@write(%inc) : (i32) -> ()
    txn.return
  }
  
  %self = txn.instance @self of @NoLoopRegister : !txn.module<"NoLoopRegister">
  txn.schedule [@getValue, @update]
}

// -----

// No loop - acyclic dependencies
txn.module @AcyclicDeps {
  %reg = txn.instance @state of @Register : !txn.module<"Register">
  
  txn.value_method @getBase() -> i32 {
    %val = txn.call @state::@read() : () -> i32
    txn.return %val : i32
  }
  
  txn.value_method @getDerived() -> i32 {
    %base = txn.call @self::@getBase() : () -> i32
    %c2 = arith.constant 2 : i32
    %result = arith.muli %base, %c2 : i32
    txn.return %result : i32
  }
  
  txn.value_method @getFinal() -> i32 {
    %derived = txn.call @self::@getDerived() : () -> i32
    %c10 = arith.constant 10 : i32
    %result = arith.addi %derived, %c10 : i32
    txn.return %result : i32
  }
  
  %self = txn.instance @self of @AcyclicDeps : !txn.module<"AcyclicDeps">
  txn.schedule [@getBase, @getDerived, @getFinal]
}
