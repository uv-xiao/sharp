// Comprehensive example demonstrating txn.if, txn.call, and txn.abort
// for showcasing will-fire logic generation in different timing modes

// Conditional processor
txn.module @ConditionalLogic {
  txn.instance @counter of @Register<i32> 
  txn.instance @enabled of @Register<i1> 
  
  // Initialize system
  txn.action_method @initialize() {
    %zero = arith.constant 0 : i32
    %false = arith.constant false
    txn.call @counter::@write(%zero) : (i32) -> ()
    txn.call @enabled::@write(%false) : (i1) -> ()
    txn.return
  }
  
  // Enable processing
  txn.action_method @enable() {
    %true = arith.constant true
    txn.call @enabled::@write(%true) : (i1) -> ()
    txn.return
  }
  
  // Conditional increment - demonstrates txn.if with txn.abort
  txn.action_method @conditional_increment(%data: i32) {
    %is_enabled = txn.call @enabled::@read() : () -> i1
    %one = arith.constant 1 : i32
    %data_last_bit = arith.andi %data, %one : i32
    %is_odd = arith.cmpi eq, %data_last_bit, %one : i32
    txn.if %is_enabled {
      // Enabled: increment counter
      %current = txn.call @counter::@read() : () -> i32
      
      txn.if %is_odd {
        %new_val = arith.addi %current, %data : i32
        txn.call @counter::@write(%new_val) : (i32) -> ()
        txn.yield
      } else {
        txn.abort
      }
      txn.yield
    } else {
      // Not enabled: abort operation
      txn.abort
    }
    txn.return
  }

  txn.action_method @enforce_increment(%data: i32) {
    %current = txn.call @counter::@read() : () -> i32
    %new_val = arith.addi %current, %data : i32
    txn.call @counter::@write(%new_val) : (i32) -> ()
    txn.return
  }

  // Set zero to counter
  txn.action_method @set_zero() {
    %zero = arith.constant 0 : i32  
    txn.call @counter::@write(%zero) : (i32) -> ()
    txn.return
  }

  // Get current counter value
  txn.value_method @get_counter() -> i32 {
    %val = txn.call @counter::@read() : () -> i32
    txn.return %val : i32
  }
  
  // Check if enabled
  txn.value_method @is_enabled() -> i1 {
    %val = txn.call @enabled::@read() : () -> i1
    txn.return %val : i1
  }
  
  txn.schedule [@initialize, @enable, @conditional_increment, @enforce_increment, @set_zero] {
    conflict_matrix = {
    }
  }
}

// Multi-instance system demonstrating will-fire interactions
txn.module @ConditionalSystem attributes {top} {
  txn.instance @processor1 of @ConditionalLogic 
  txn.instance @processor2 of @ConditionalLogic 

  // Initialize system
  txn.action_method @init_system() {
    txn.call @processor1::@initialize() : () -> ()
    txn.call @processor2::@initialize() : () -> ()
    txn.return
  }
  
  // Enable both processors
  txn.action_method @enable_both() {
    txn.call @processor1::@enable() : () -> ()
    txn.call @processor2::@enable() : () -> ()
    txn.return
  }
  
  // Increment one processor
  txn.action_method @increment(%arb: i1) {
    %data = arith.constant 1 : i32
    txn.if %arb {
      txn.call @processor1::@conditional_increment(%data) : (i32) -> ()
      txn.yield
    } else {
      txn.call @processor2::@conditional_increment(%data) : (i32) -> ()
      txn.yield
    }
    txn.return
  }
  
  txn.schedule [@init_system, @enable_both, @increment] {
    conflict_matrix = {
    }
  }
}

// One-instance system demonstrating will-fire interactions
txn.module @OneInstanceSystem {
  txn.instance @processor of @ConditionalLogic 

  // Initialize system
  txn.action_method @init_system() {
    txn.call @processor::@initialize() : () -> ()
    txn.return
  }

  // Enable processor
  txn.action_method @enable() {
    txn.call @processor::@enable() : () -> ()
    txn.return
  }

  // Increment processor
  txn.action_method @increment(%data: i32) {
    %one = arith.constant 1 : i32
    %plus_1 = arith.addi %data, %one: i32
    txn.call @processor::@conditional_increment(%data) : (i32) -> ()
    txn.call @processor::@enforce_increment(%plus_1) : (i32) -> ()
    txn.return
  }

  txn.schedule [@init_system, @enable, @increment] {
    conflict_matrix = {
    }
  }
}