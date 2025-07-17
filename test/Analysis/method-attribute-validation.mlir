// RUN: sharp-opt --sharp-primitive-gen --sharp-infer-conflict-matrix --sharp-reachability-analysis --sharp-general-check --sharp-pre-synthesis-check --split-input-file --verify-diagnostics %s

// Test method attribute validation

// Valid attributes with custom signal names
txn.module @ValidAttributes {
  %reg = txn.instance @state of @Register : index
  
  // Custom prefix and postfixes
  txn.value_method @getValue() -> i32 
    attributes {prefix = "get", result = "_data"} {
    %val = txn.call @state::@read() : () -> i32
    txn.return %val : i32
  }
  
  txn.action_method @reset() -> () 
    attributes {prefix = "do", ready = "_ready", enable = "_en"} {
    %c0 = arith.constant 0 : i32
    txn.call @state::@write(%c0) : (i32) -> ()
    txn.return
  }
  txn.schedule [@getValue, @reset]
}

// -----

// Signal name conflicts with module name
txn.module @NameConflict1 {
  // expected-error@+1 {{Method signal name conflicts with existing name: NameConflict1}}
  txn.value_method @method() -> i32 attributes {prefix = "NameConflict1", result = ""} {
    %c0 = arith.constant 0 : i32
    txn.return %c0 : i32
  }
  txn.schedule [@method]
}

// -----

// Signal name conflicts between methods
txn.module @NameConflict2 {
  txn.value_method @getValue() -> i32 attributes {prefix = "data", result = ""} {
    %c0 = arith.constant 0 : i32
    txn.return %c0 : i32
  }
  
  // expected-error@+2 {{Method name conflicts with existing name: data}}
  // expected-error@+1 {{Method signal name conflicts with existing name: data}}
  txn.action_method @setData(%val: i32) -> () attributes {prefix = "data", result = ""} {
    txn.return
  }
  txn.schedule [@getValue, @setData]
}

// -----

// Signal name conflicts with instance name
txn.module @NameConflict3 {
  %reg = txn.instance @control of @Register : index
  
  // expected-error@+2 {{Method name conflicts with existing name: control}}
  // expected-error@+1 {{Method enable signal conflicts with existing name: control}}
  txn.action_method @method() -> () attributes {prefix = "control", enable = ""} {
    txn.return
  }
  txn.schedule [@method]
}

// -----

// Invalid always_ready - method has conflicts
txn.module @InvalidAlwaysReady {
  %reg = txn.instance @state of @Register : index
  
  // expected-error@+1 {{Method marked always_ready but has potential conflicts}}
  txn.action_method @write(%val: i32) -> () attributes {always_ready} {
    txn.call @state::@write(%val) : (i32) -> ()
    txn.return
  }
  
  txn.rule @conflictingRule {
    %c42 = arith.constant 42 : i32
    txn.call @state::@write(%c42) : (i32) -> ()
    txn.return
  }
  
  // write conflicts with conflictingRule
  txn.schedule [@write, @conflictingRule] {
    conflict_matrix = {
      "write,conflictingRule" = 2 : i32  // C (conflict)
    }
  }
}

// -----

// Valid always_ready - no conflicts
txn.module @ValidAlwaysReady {
  %reg = txn.instance @state of @Register : index
  
  // This is OK - read has no conflicts
  txn.action_method @read() -> i32 attributes {always_ready} {
    %val = txn.call @state::@read() : () -> i32
    txn.return %val : i32
  }
  
  txn.rule @writer {
    %c42 = arith.constant 42 : i32
    txn.call @state::@write(%c42) : (i32) -> ()
    txn.return
  }
  
  // read is conflict-free with writer
  txn.schedule [@read, @writer] {
    conflict_matrix = {
      "read,writer" = 3 : i32  // CF (conflict-free)
    }
  }
}

// -----

// Invalid always_enable - has conditional callers
txn.module @InvalidAlwaysEnable {
  // expected-error@+1 {{Method marked always_enable but has conditional callers}}
  txn.action_method @conditionalMethod() -> () attributes {always_enable} {
    txn.return
  }
  txn.schedule [@conditionalMethod]
}

txn.module @Caller1 {
  %inst = txn.instance @m of @InvalidAlwaysEnable : index
  
  txn.rule @conditionalCaller {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cond = arith.cmpi "eq", %c0, %c1 : i32
    
    // Conditional call makes always_enable invalid
    txn.if %cond {
      txn.call @m::@conditionalMethod() : () -> ()
      txn.yield
    } else {
      txn.yield
    }
    txn.return
  }
  txn.schedule [@conditionalCaller]
}

// -----

// Valid always_enable - all callers unconditional
txn.module @ValidAlwaysEnable {
  txn.action_method @unconditionalMethod() -> () attributes {always_enable} {
    txn.return
  }
  txn.schedule [@unconditionalMethod]
}

txn.module @Caller2 {
  %inst = txn.instance @m of @ValidAlwaysEnable : index
  
  txn.rule @unconditionalCaller {
    // Always called - always_enable is valid
    txn.call @m::@unconditionalMethod() : () -> ()
    txn.return
  }
  txn.schedule [@unconditionalCaller]
}

// -----

// Multiple attribute combinations
txn.module @MultipleAttributes {
  %wire = txn.instance @w of @Wire : index
  
  // Valid: always_ready because no conflicts in schedule
  txn.action_method @alwaysMethod() -> () 
    attributes {always_ready, prefix = "am", enable = "_go"} {
    %c0 = arith.constant 0 : i32
    txn.call @w::@write(%c0) : (i32) -> ()
    txn.return
  }
  
  // No conflicts specified
  txn.schedule [@alwaysMethod] {}
}

// Primitive modules (simplified for testing)
txn.module @Register {
  txn.value_method @read() -> i32 {
    %0 = arith.constant 0 : i32
    txn.return %0 : i32
  }
  txn.action_method @write(%val: i32) -> () {
    txn.return
  }
  txn.schedule [@read, @write]
}

txn.module @Wire {
  txn.value_method @read() -> i32 {
    %0 = arith.constant 0 : i32
    txn.return %0 : i32
  }
  txn.action_method @write(%val: i32) -> () {
    txn.return
  }
  txn.schedule [@read, @write]
}